"""Identify an exact checkpoint within a job's output folder."""

from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CheckpointReference(BaseModel):
    """Record a checkpoint location and its expected content fingerprint."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )

    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_path")
    @classmethod
    def check_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)

        if (
            not value.strip()
            or path == PurePosixPath(".")
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or ":" in value
            or "\x00" in value
        ):
            raise ValueError(
                "Use a relative file path with forward slashes and no '..'"
            )

        return value