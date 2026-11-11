"""Registered datasets inside the workspace, identified by content hash.

Layout: ``datasets/<dataset-id>/manifest.json`` plus ``datasets/<id>/data/``.
H6 checks structure and identity only. Image/label validation, registering
external folders, and generated-dataset caching are completed in V1/V2.
"""

import hashlib
from pathlib import Path, PurePosixPath
import re

from foxtrot.config.dataset import DatasetMetadata
from foxtrot.pipeline.contracts import DatasetInput

from .files import fingerprint, job_file, sync_directory, write_new_text
from .workspace import initialize_workspace


DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
MANIFEST = "manifest.json"
DATA = "data"


class DatasetUnavailable(ValueError):
    """The requested dataset is missing, unsafe, invalid, or changed."""


def _dataset_directory(root: Path, dataset_id: str) -> Path:
    # Same pattern as DatasetMetadata.dataset_id, so an ID can never be a path.
    if not isinstance(dataset_id, str) or not DATASET_ID.fullmatch(dataset_id):
        raise DatasetUnavailable("Invalid dataset ID")
    datasets = root / "datasets"
    if datasets.is_symlink() or not datasets.is_dir():
        raise DatasetUnavailable("Workspace datasets folder is missing or unsafe")
    return datasets / dataset_id


def _safe_directory(base: Path, relative: str) -> Path:
    """Follow a relative split reference without crossing a symlink."""
    path = base
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise DatasetUnavailable("Dataset paths must not traverse symbolic links")
    if not path.is_dir():
        raise DatasetUnavailable(f"Dataset split folder is missing: {relative}")
    return path


def dataset_content_sha256(data_root: Path) -> str:
    """Hash every file's relative path and content, in sorted order.

    Renaming, adding, removing, or changing a file changes the identity.
    Symbolic links are rejected so identity cannot depend on outside files.
    """
    if data_root.is_symlink() or not data_root.is_dir():
        raise DatasetUnavailable("Dataset data folder is missing or unsafe")
    listing = hashlib.sha256()
    for path in sorted(data_root.rglob("*"), key=lambda item: item.relative_to(data_root).as_posix()):
        if path.is_symlink():
            raise DatasetUnavailable("Datasets must not contain symbolic links")
        if path.is_file():
            digest, size = fingerprint(path)
            relative = path.relative_to(data_root).as_posix()
            listing.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))
    return listing.hexdigest()


def register_dataset(
    workspace_root: str | Path, source_checkout: str | Path, *, dataset_id: str,
    class_names: tuple[str, ...], train_images_ref: str, validation_images_ref: str,
) -> DatasetInput:
    """Record a dataset already placed in ``datasets/<id>/data``.

    The manifest is written once. The class mapping is the dataset's own; it is
    never forced to the default Foxtrot shapes.
    """
    root = initialize_workspace(workspace_root, source_checkout)
    directory = _dataset_directory(root, dataset_id)
    if directory.is_symlink() or not directory.is_dir():
        raise DatasetUnavailable("Place dataset files in the workspace before registering")
    data = directory / DATA
    for reference in (train_images_ref, validation_images_ref):
        _safe_directory(data, reference)
    metadata = DatasetMetadata(
        dataset_id=dataset_id, class_names=class_names,
        train_images_ref=train_images_ref, validation_images_ref=validation_images_ref,
        content_sha256=dataset_content_sha256(data), validation_status="pending",
    )
    try:
        write_new_text(directory / MANIFEST, metadata.model_dump_json(indent=2))
    except FileExistsError as error:
        raise DatasetUnavailable("Dataset is already registered") from error
    sync_directory(directory)
    return DatasetInput(metadata=metadata, root=str(data))


def list_datasets(
    workspace_root: str | Path, source_checkout: str | Path,
) -> tuple[DatasetMetadata, ...]:
    """Registered manifests for selection screens; content is verified at job launch."""
    root = initialize_workspace(workspace_root, source_checkout)
    datasets = root / "datasets"
    found = []
    for directory in sorted(datasets.iterdir()) if datasets.is_dir() else ():
        if directory.is_symlink() or not directory.is_dir() or not DATASET_ID.fullmatch(directory.name):
            continue
        try:
            metadata = DatasetMetadata.model_validate_json(job_file(directory, MANIFEST).read_bytes())
        except (ValueError, OSError):
            continue
        if metadata.dataset_id == directory.name:
            found.append(metadata)
    return tuple(found)


def load_dataset(
    workspace_root: str | Path, source_checkout: str | Path, dataset_id: str,
) -> DatasetInput:
    """Return a registered dataset only if its files still match the manifest."""
    root = initialize_workspace(workspace_root, source_checkout)
    directory = _dataset_directory(root, dataset_id)
    if directory.is_symlink() or not directory.is_dir():
        raise DatasetUnavailable(f"Unknown dataset: {dataset_id}")
    try:
        metadata = DatasetMetadata.model_validate_json(job_file(directory, MANIFEST).read_bytes())
    except (ValueError, OSError) as error:
        raise DatasetUnavailable("Dataset manifest is missing or invalid") from error
    if metadata.dataset_id != dataset_id:
        raise DatasetUnavailable("Dataset manifest belongs to a different dataset")
    if metadata.validation_status == "invalid":
        raise DatasetUnavailable("Dataset failed validation")
    data = directory / DATA
    for reference in (metadata.train_images_ref, metadata.validation_images_ref):
        _safe_directory(data, reference)
    if dataset_content_sha256(data) != metadata.content_sha256:
        raise DatasetUnavailable("Dataset files changed after registration")
    return DatasetInput(metadata=metadata, root=str(data))
