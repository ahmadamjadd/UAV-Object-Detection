"""Backend-managed local settings; credentials never reach the browser or exports.

``settings/tracking.json`` holds public destination details.
``settings/secrets.json`` (mode 0600) holds credentials keyed by destination.
Job specifications only ever name a destination ID.
"""

import json
import logging
import os
from pathlib import Path
import stat
import threading

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from foxtrot.persistence.files import sync_directory
from foxtrot.persistence.workspace import initialize_workspace


DESTINATION_ID = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
REDACTED = "***"


class TrackingDestination(BaseModel):
    """Public details. The tracking API URI stays separate from the browser URL."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    destination_id: str = Field(pattern=DESTINATION_ID)
    kind: str = Field(pattern=r"^(mlflow|local)$")
    tracking_uri: str = Field(min_length=1, max_length=2048)
    browser_url: str | None = Field(default=None, max_length=2048)
    experiment_name: str = Field(default="Foxtrot Detection", min_length=1, max_length=250)

    @field_validator("tracking_uri", "browser_url")
    @classmethod
    def no_embedded_credentials(cls, value: str | None) -> str | None:
        if value is not None and "@" in value.split("://", 1)[-1].split("/", 1)[0]:
            raise ValueError("Put credentials in the credential fields, not in the URL")
        return value


class DestinationUpdate(TrackingDestination):
    """Incoming form data; credentials are write-only."""

    username: str | None = Field(default=None, max_length=256)
    token: SecretStr | None = None


class SettingsStore:
    def __init__(self, workspace_root: str | Path, source_checkout: str | Path):
        root = initialize_workspace(workspace_root, source_checkout)
        self.directory = root / "settings"
        self.public_path = self.directory / "tracking.json"
        self.secret_path = self.directory / "secrets.json"
        self._lock = threading.Lock()

    def _read(self, path: Path) -> dict:
        if path.is_symlink():
            raise ValueError("Settings files must not be symbolic links")
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, value: dict, mode: int) -> None:
        """Replace atomically through a private temporary file."""
        if path.is_symlink():
            raise ValueError("Settings files must not be symbolic links")
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.unlink(missing_ok=True)
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(self.directory)

    def destinations(self) -> list[dict]:
        """Public view for the browser and exports: only whether credentials exist."""
        with self._lock:
            public, secrets = self._read(self.public_path), self._read(self.secret_path)
        return [
            {**TrackingDestination.model_validate(value).model_dump(),
             "has_credentials": bool(secrets.get(key))}
            for key, value in sorted(public.items())
        ]

    def save_destination(self, update: DestinationUpdate) -> dict:
        public_fields = TrackingDestination.model_fields.keys()
        with self._lock:
            public = self._read(self.public_path)
            public[update.destination_id] = update.model_dump(include=set(public_fields))
            self._write(self.public_path, public, 0o644)
            if update.username is not None or update.token is not None:
                secrets = self._read(self.secret_path)
                secrets[update.destination_id] = {
                    "username": update.username,
                    "token": update.token.get_secret_value() if update.token else None,
                }
                self._write(self.secret_path, secrets, 0o600)
        return next(d for d in self.destinations() if d["destination_id"] == update.destination_id)

    def credentials(self, destination_id: str) -> dict:
        """Backend-only access, e.g. for the V1 MLflow uploader."""
        with self._lock:
            return dict(self._read(self.secret_path).get(destination_id) or {})

    def secret_values(self) -> list[str]:
        if not self.secret_path.exists():
            return []
        info = self.secret_path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("Credential file must be a private regular file")
        with self._lock:
            values = self._read(self.secret_path).values()
        return [item["token"] for item in values if item.get("token")]


class RedactingFilter(logging.Filter):
    """Replace known credential values in log messages, including uvicorn's."""

    def __init__(self, store: SettingsStore):
        super().__init__()
        self.store = store

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = [value for value in self.store.secret_values() if len(value) >= 4]
        if secrets:
            message = record.getMessage()
            for value in secrets:
                message = message.replace(value, REDACTED)
            record.msg, record.args = message, None
        return True
