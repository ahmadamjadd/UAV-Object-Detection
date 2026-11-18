"""Bring an existing YOLO dataset from anywhere on this computer into the workspace.

The dataset is checked first, then copied into a temporary folder, fingerprinted,
and only then published under ``datasets/<id>/`` in one rename. A failed or
interrupted import never leaves something that looks like a usable dataset.
The original folder is only read, never changed.
"""

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from foxtrot.config.dataset import DatasetMetadata
from foxtrot.pipeline.contracts import DatasetInput
from foxtrot.pipeline.dataset_validation import DatasetReport, labels_directory, validate_dataset

from .datasets import DATA, MANIFEST, DatasetUnavailable, _dataset_directory, dataset_content_sha256
from .files import sync_directory, write_new_text
from .workspace import initialize_workspace

REPORT = "validation.json"
SKIPPED_SUFFIXES = {".cache"}  # YOLO rewrites these; they are not part of the data


# Copy one folder of images or labels, refusing symbolic links and skipping YOLO caches.
def _copy_folder(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for item in sorted(source.iterdir()):
        if item.is_symlink():
            raise DatasetUnavailable(f"Dataset must not contain symbolic links: {item.name}")
        if item.is_dir():
            _copy_folder(item, target / item.name)
        elif item.is_file() and item.suffix not in SKIPPED_SUFFIXES:
            shutil.copy2(item, target / item.name)


# Check, copy and register an external dataset; returns (dataset or None, check report).
# An invalid dataset is not imported at all: the report says what to fix.
def import_dataset(
    workspace_root: str | Path, source_checkout: str | Path, *, source: str | Path,
    dataset_id: str, class_names: tuple[str, ...],
    train_images_ref: str = "train/images", validation_images_ref: str = "val/images",
    missing_labels: Literal["background", "error"] = "background",
) -> tuple[DatasetInput | None, DatasetReport]:
    root = initialize_workspace(workspace_root, source_checkout)
    source = Path(source)
    if not source.is_absolute():
        raise DatasetUnavailable("Dataset folder must be an absolute path")
    source = source.resolve()
    if not source.is_dir():
        raise DatasetUnavailable("Dataset folder not found")
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise DatasetUnavailable("Dataset folder must be outside the Foxtrot workspace")
    directory = _dataset_directory(root, dataset_id)
    if directory.exists() or directory.is_symlink():
        raise DatasetUnavailable(f"Dataset ID already used: {dataset_id}")
    # Validates the names (non-blank, unique) before any work is done.
    DatasetMetadata(dataset_id=dataset_id, class_names=class_names, train_images_ref=train_images_ref,
                    validation_images_ref=validation_images_ref, content_sha256="0" * 64)

    splits = {"train": train_images_ref, "val": validation_images_ref}
    report = validate_dataset(source, class_names, splits, missing_labels=missing_labels)
    if not report.valid:
        return None, report

    staging = root / "datasets" / f".import-{uuid4().hex}"
    try:
        data = staging / DATA
        for reference in splits.values():
            images = PurePosixPath(reference)
            labels = PurePosixPath(labels_directory(Path(reference)).as_posix())
            _copy_folder(source / images, data / images)
            if (source / labels).is_dir():
                _copy_folder(source / labels, data / labels)
        metadata = DatasetMetadata(
            dataset_id=dataset_id, class_names=class_names,
            train_images_ref=train_images_ref, validation_images_ref=validation_images_ref,
            content_sha256=dataset_content_sha256(data), validation_status="valid",
        )
        write_new_text(staging / MANIFEST, metadata.model_dump_json(indent=2))
        write_new_text(staging / REPORT, json.dumps(
            {"source": str(source), "report": report.model_dump(mode="json")}, indent=2))
        sync_directory(staging)
        staging.rename(directory)  # publish: the dataset appears complete or not at all
        sync_directory(directory.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DatasetInput(metadata=metadata, root=str(directory / DATA)), report
