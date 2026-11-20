"""Bring in a dataset exactly as recorded at a Git/DVC revision (V4).

The member names a repository (a local Git+DVC folder or a Git URL), a revision
(commit, tag or branch) and the DVC-tracked folder. DVC downloads that version
into a hidden folder inside the workspace (the source repository and its cache
are never changed), the dataset is checked like any import, and it is published
with ``dvc.json`` recording the exact commit and DVC hash.

Runs in the generation runner process: DVC is heavy and may download a lot.
"""

from datetime import UTC, datetime
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from foxtrot.config.dataset import DatasetMetadata
from foxtrot.pipeline.contracts import DatasetInput, PipelineCancelled, PipelineFailure, ProgressReporter
from foxtrot.pipeline.dataset_validation import labels_directory, validate_dataset

from .datasets import DATA, MANIFEST, _dataset_directory, dataset_content_sha256
from .files import sync_directory, write_new_text
from .workspace import initialize_workspace

PROVENANCE = "dvc.json"
REPORT = "validation.json"


# What the member asks for: which repository, revision and folder, and how to register it.
class DvcImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1, max_length=4096)
    revision: str = Field(min_length=1, max_length=256, pattern=r"^[^\s\x00]+$")
    path: str = Field(min_length=1, max_length=1024)
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    class_names: tuple[str, ...] = Field(min_length=1, max_length=1000)
    role: Literal["training", "validation", "test"] = "training"
    train_images_ref: str = "train/images"
    validation_images_ref: str = "val/images"
    # Optional: log in to the DVC remote with a saved tracking destination's username/token
    # (DagsHub uses the same login for MLflow and DVC). Never stored in the dataset record.
    credentials_destination_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

    # A local absolute folder or an http(s)/ssh/git URL; never a relative path.
    @field_validator("repository")
    @classmethod
    def repository_location(cls, value: str) -> str:
        if not (value.startswith("/") or value.startswith(("https://", "http://", "ssh://", "git@"))):
            raise ValueError("Repository must be an absolute folder or a Git URL")
        return value

    # The DVC-tracked folder inside the repository, written with forward slashes.
    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("Path must be relative to the repository, without '..'")
        return value


# Read a DVC dataset's recorded origin, if it matches the manifest.
def load_dvc_provenance(dataset: DatasetInput) -> dict | None:
    path = Path(dataset.root).parent / PROVENANCE
    if path.is_symlink() or not path.is_file():
        return None
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metadata = dataset.metadata
    if not isinstance(provenance, dict) or (provenance.get("dataset_id"), provenance.get("content_sha256")) != (
            metadata.dataset_id, metadata.content_sha256):
        return None
    return provenance


# Download the requested folder at the requested revision into `target`; returns (commit, DVC md5).
def _download(request: DvcImportRequest, target: Path, reporter: ProgressReporter,
              credentials: dict | None) -> tuple[str, str | None]:
    from dvc.api import DVCFileSystem

    options = {}
    if credentials and credentials.get("username") and credentials.get("token"):
        options["remote_config"] = {"auth": "basic", "user": credentials["username"],
                                    "password": credentials["token"]}
    try:
        filesystem = DVCFileSystem(request.repository, rev=request.revision, **options)
        commit = filesystem.repo.get_rev()
        info = filesystem.info(request.path)
    except FileNotFoundError as error:
        raise PipelineFailure("dvc_path_missing", f"{request.path} does not exist at {request.revision}") from error
    except Exception as error:  # unknown revision, unreachable repository, not a DVC repository
        raise PipelineFailure("dvc_unavailable", f"Cannot open {request.revision} ({type(error).__name__}): "
                              f"{_safe(error, credentials)}") from error
    if info.get("type") != "directory":
        raise PipelineFailure("dvc_not_folder", f"{request.path} is not a folder")
    reporter.progress({"phase": f"downloading {request.path} at {commit[:8]}"})
    if reporter.wait_cancelled():
        raise PipelineCancelled()
    try:
        filesystem.get(request.path, str(target), recursive=True)
    except Exception as error:  # missing remote data, credentials, network
        raise PipelineFailure("dvc_download_failed", f"Download failed ({type(error).__name__}): "
                              f"{_safe(error, credentials)}") from error
    return commit, info.get("md5")


# An error message with any saved secret removed, short enough for the website.
def _safe(error: Exception, credentials: dict | None) -> str:
    text = str(error)
    for secret in (credentials or {}).values():
        if isinstance(secret, str) and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text[:200]


# Download, check and register one DVC dataset version. Nothing is published if any step fails.
def import_dvc_dataset(
    workspace_root: str | Path, source_checkout: str | Path, request: DvcImportRequest, reporter: ProgressReporter,
) -> tuple[DatasetInput, dict]:
    root = initialize_workspace(workspace_root, source_checkout)
    credentials = None
    if request.credentials_destination_id:
        from foxtrot.api.settings import SettingsStore
        credentials = SettingsStore(root, source_checkout).credentials(request.credentials_destination_id)
    directory = _dataset_directory(root, request.dataset_id)
    if directory.exists() or directory.is_symlink():
        raise PipelineFailure("dataset_exists", f"Dataset ID already used: {request.dataset_id}")
    if request.role == "training":
        splits = {"train": request.train_images_ref, "val": request.validation_images_ref}
    else:
        splits = {"evaluation": request.train_images_ref}
    DatasetMetadata(dataset_id=request.dataset_id, class_names=request.class_names, role=request.role,
                    train_images_ref=splits.get("train", request.train_images_ref),
                    validation_images_ref=splits.get("val", request.train_images_ref), content_sha256="0" * 64)
    staging = root / "datasets" / f".dvc-{uuid4().hex}"
    try:
        download = staging / "download"
        commit, dvc_md5 = _download(request, download, reporter, credentials)
        reporter.progress({"phase": "validating"})
        report = validate_dataset(download, request.class_names, splits,
                                  missing_labels="background" if request.role == "training" else "error")
        if not report.valid:
            first = report.errors[0]
            raise PipelineFailure("dvc_dataset_invalid", f"{report.error_count} problem(s), first: "
                                  f"{first.split} {first.file or ''} {first.message}".strip())
        data = staging / DATA
        for reference in dict.fromkeys(splits.values()):  # keep only the checked image/label folders
            for folder in (PurePosixPath(reference), PurePosixPath(labels_directory(Path(reference)).as_posix())):
                if (download / folder).is_dir():
                    (data / folder).parent.mkdir(parents=True, exist_ok=True)
                    (download / folder).rename(data / folder)
        shutil.rmtree(download)
        metadata = DatasetMetadata(
            dataset_id=request.dataset_id, class_names=request.class_names, role=request.role,
            train_images_ref=splits.get("train", request.train_images_ref),
            validation_images_ref=splits.get("val", request.train_images_ref),
            content_sha256=dataset_content_sha256(data), validation_status="valid",
        )
        provenance = {
            "format": 1, "source": "dvc", "dataset_id": request.dataset_id,
            "content_sha256": metadata.content_sha256, "repository": request.repository,
            "revision": request.revision, "commit": commit, "path": request.path, "dvc_md5": dvc_md5,
            "role": request.role, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        write_new_text(staging / MANIFEST, metadata.model_dump_json(indent=2))
        write_new_text(staging / PROVENANCE, json.dumps(provenance, indent=2))
        write_new_text(staging / REPORT, report.model_dump_json(indent=2))
        sync_directory(staging)
        staging.rename(directory)  # publish: complete or not at all
        sync_directory(directory.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DatasetInput(metadata=metadata, root=str(directory / DATA)), provenance
