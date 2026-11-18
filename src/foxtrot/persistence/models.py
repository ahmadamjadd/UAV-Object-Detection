"""Pretrained model weights kept in the workspace, each with a recorded fingerprint.

``models/<model>.pt`` is the weights file and ``models/<model>.json`` records its
SHA-256, size and where it came from. A download is written to a temporary file
and only published when complete, so a half-downloaded file is never used.

    python -m foxtrot.persistence.models --workspace /abs/ws --source-checkout /abs/MlOps install yolov8n
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
from uuid import uuid4

from .files import sync_directory
from .workspace import initialize_workspace

# Official Ultralytics release files; the same ones YOLO("yolov8n.pt") would fetch.
WEIGHTS_SOURCES = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
}
MIN_BYTES = 100_000          # anything smaller is an error page, not weights
MAX_BYTES = 500_000_000      # refuse runaway downloads
TIMEOUT_SECONDS = 60


# Raised when weights cannot be installed or no longer match their record.
class WeightsError(ValueError):
    pass


# What the workspace knows about one installed weights file.
@dataclass(frozen=True)
class WeightsRecord:
    model_id: str
    sha256: str
    size_bytes: int
    source: str


# Return the models folder, refusing a redirected (symlinked) one.
def _models_directory(root: Path) -> Path:
    directory = root / "models"
    if directory.is_symlink() or not directory.is_dir():
        raise WeightsError("Workspace models folder is missing or unsafe")
    return directory


# Check the model name is one Foxtrot knows how to fetch.
def _check_model_id(model_id: str) -> None:
    if model_id not in WEIGHTS_SOURCES:
        raise WeightsError(f"Unknown model: {model_id}")


# SHA-256 of a file, read in chunks so large files use little memory.
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Read the saved record for a model, or None if it was never installed.
def read_weights_record(models_directory: Path, model_id: str) -> WeightsRecord | None:
    record = models_directory / f"{model_id}.json"
    if not record.is_file() or record.is_symlink():
        return None
    return WeightsRecord(**json.loads(record.read_text(encoding="utf-8")))


# Confirm an installed weights file still matches its record; returns its fingerprint.
def verify_weights(models_directory: Path, model_id: str) -> str:
    weights = models_directory / f"{model_id}.pt"
    record = read_weights_record(models_directory, model_id)
    if record is None or weights.is_symlink() or not weights.is_file():
        raise WeightsError(f"Pretrained weights for {model_id} are not installed")
    digest = _sha256(weights)
    if digest != record.sha256:
        raise WeightsError(f"{model_id}.pt changed after it was installed")
    return digest


# List every known model and whether its weights are installed and unchanged.
def list_weights(workspace_root: str | Path, source_checkout: str | Path) -> list[dict]:
    directory = _models_directory(initialize_workspace(workspace_root, source_checkout))
    result = []
    for model_id in sorted(WEIGHTS_SOURCES):
        record = read_weights_record(directory, model_id)
        try:
            verify_weights(directory, model_id)
            status = "installed"
        except WeightsError:
            status = "changed" if record else "missing"
        result.append({"model_id": model_id, "status": status,
                       "sha256": record.sha256 if record else None,
                       "size_bytes": record.size_bytes if record else None})
    return result


# Copy a local file or download the official file, check it, record it, then publish it.
# Installing again when the same file is already installed changes nothing.
def install_weights(
    workspace_root: str | Path, source_checkout: str | Path, model_id: str, *,
    source_file: str | Path | None = None, opener=urllib.request.urlopen,
) -> WeightsRecord:
    _check_model_id(model_id)
    directory = _models_directory(initialize_workspace(workspace_root, source_checkout))
    existing = read_weights_record(directory, model_id)
    if existing is not None:
        try:
            verify_weights(directory, model_id)
            return existing
        except WeightsError:
            pass  # changed or half-present: replace it with a fresh, verified copy
    temporary = directory / f".download-{uuid4().hex}.pt"
    record_temporary = directory / f".record-{uuid4().hex}.json"
    try:
        if source_file is not None:
            source = Path(source_file).resolve()
            if not source.is_file():
                raise WeightsError("Weights file not found")
            shutil.copyfile(source, temporary)
            origin = f"file://{source}"
        else:
            origin = WEIGHTS_SOURCES[model_id]
            with opener(origin, timeout=TIMEOUT_SECONDS) as response, temporary.open("xb") as stream:
                written = 0
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    written += len(chunk)
                    if written > MAX_BYTES:
                        raise WeightsError("Download is larger than any expected weights file")
                    stream.write(chunk)
        size = temporary.stat().st_size
        with temporary.open("rb") as stream:
            looks_like_pytorch = stream.read(2) == b"PK"  # PyTorch saves weights as a zip archive
        if size < MIN_BYTES or not looks_like_pytorch:
            raise WeightsError("The file is not a PyTorch weights file")
        record = WeightsRecord(model_id=model_id, sha256=_sha256(temporary), size_bytes=size, source=origin)
        record_temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        os.replace(temporary, directory / f"{model_id}.pt")
        os.replace(record_temporary, directory / f"{model_id}.json")
        sync_directory(directory)
        return record
    except OSError as error:
        raise WeightsError(f"Could not get weights ({type(error).__name__})") from error
    finally:
        temporary.unlink(missing_ok=True)
        record_temporary.unlink(missing_ok=True)


# Command line: install weights into a workspace, by download or from a local file.
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("action", choices=("install", "list"))
    parser.add_argument("model_id", nargs="?", default="yolov8n", choices=sorted(WEIGHTS_SOURCES))
    parser.add_argument("--from", dest="source_file", type=Path, help="Use this local .pt file instead of downloading")
    arguments = parser.parse_args(argv)
    try:
        if arguments.action == "list":
            print(json.dumps(list_weights(arguments.workspace, arguments.source_checkout), indent=2))
        else:
            record = install_weights(arguments.workspace, arguments.source_checkout, arguments.model_id,
                                     source_file=arguments.source_file)
            print(f"Installed {record.model_id}.pt ({record.size_bytes} bytes, sha256 {record.sha256[:12]}…)")
    except WeightsError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
