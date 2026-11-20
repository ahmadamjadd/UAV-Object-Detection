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


# ------------------------------------------------------------- V4: evaluation datasets

GROUPS_FILE = "groups.json"  # optional: {"image.jpg": "flight-3", ...} in the source folder


# Read the optional image -> flight/scene mapping; every image must have a non-blank group.
def _read_groups(source: Path, images: list[str]) -> dict[str, str] | None:
    path = source / GROUPS_FILE
    if not path.is_file():
        return None
    try:
        groups = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetUnavailable("groups.json is not valid JSON") from error
    if not isinstance(groups, dict) or not all(isinstance(v, str) and v.strip() for v in groups.values()):
        raise DatasetUnavailable("groups.json must map image file names to non-blank group names")
    missing = [name for name in images if name not in groups]
    if missing:
        raise DatasetUnavailable(f"groups.json has no group for {len(missing)} image(s), e.g. {missing[0]}")
    return {name: groups[name] for name in images}


# Copy the chosen images (and their labels) into a staged evaluation dataset and publish it.
def _publish_evaluation(root: Path, source: Path, images_ref: str, dataset_id: str, role: str,
                        class_names: tuple[str, ...], names: list[str], groups: dict[str, str] | None,
                        report: DatasetReport) -> DatasetInput:
    directory = _dataset_directory(root, dataset_id)
    if directory.exists() or directory.is_symlink():
        raise DatasetUnavailable(f"Dataset ID already used: {dataset_id}")
    staging = root / "datasets" / f".import-{uuid4().hex}"
    images = PurePosixPath(images_ref)
    labels = PurePosixPath(labels_directory(Path(images_ref)).as_posix())
    try:
        data = staging / DATA
        (data / "images").mkdir(parents=True)
        (data / "labels").mkdir()
        for name in names:
            shutil.copy2(source / images / name, data / "images" / name)
            label = source / labels / f"{Path(name).stem}.txt"
            if label.is_file():
                shutil.copy2(label, data / "labels" / label.name)
        if groups is not None:
            write_new_text(data / GROUPS_FILE, json.dumps({n: groups[n] for n in names}, indent=2, sort_keys=True))
        metadata = DatasetMetadata(
            dataset_id=dataset_id, class_names=class_names, train_images_ref="images",
            validation_images_ref="images", content_sha256=dataset_content_sha256(data),
            validation_status="valid", role=role,
        )
        write_new_text(staging / MANIFEST, metadata.model_dump_json(indent=2))
        write_new_text(staging / REPORT, json.dumps(
            {"source": str(source), "report": report.model_dump(mode="json")}, indent=2))
        sync_directory(staging)
        staging.rename(directory)
        sync_directory(directory.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DatasetInput(metadata=metadata, root=str(directory / DATA))


# Register labeled real images that only score models (never train them).
# role "validation" or "test"; with test_share > 0 the images are divided into
# "<id>-val" (validation) and "<id>-test" (test), keeping each flight/scene group
# from groups.json in one of them. Every image must have a label file.
def import_evaluation_dataset(
    workspace_root: str | Path, source_checkout: str | Path, *, source: str | Path, dataset_id: str,
    class_names: tuple[str, ...], role: Literal["validation", "test"] = "validation",
    images_ref: str = "images", test_share: float = 0.0, split_seed: int = 42,
) -> tuple[list[DatasetInput], DatasetReport]:
    from foxtrot.config.generation import SplitSettings
    from foxtrot.pipeline.dataset_validation import IMAGE_EXTENSIONS
    from foxtrot.pipeline.foxtrot_split import plan_split

    root = initialize_workspace(workspace_root, source_checkout)
    source = Path(source)
    if not source.is_absolute() or not source.resolve().is_dir():
        raise DatasetUnavailable("Dataset folder not found (use an absolute path)")
    source = source.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise DatasetUnavailable("Dataset folder must be outside the Foxtrot workspace")
    if not 0 <= test_share < 1:
        raise DatasetUnavailable("Test share must be at least 0 and below 1")
    DatasetMetadata(dataset_id=dataset_id, class_names=class_names, train_images_ref=images_ref,
                    validation_images_ref=images_ref, content_sha256="0" * 64, role=role)
    report = validate_dataset(source, class_names, {"evaluation": images_ref}, missing_labels="error")
    if not report.valid:
        return [], report
    names = sorted(p.name for p in (source / images_ref).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    groups = _read_groups(source, names)
    if test_share == 0:
        return [_publish_evaluation(root, source, images_ref, dataset_id, role, class_names, names, groups, report)], report
    outcome = plan_split({n: (groups or {}).get(n, n) for n in names},
                         SplitSettings(train_ratio=round(1 - test_share, 6), seed=split_seed,
                                       group_by_background=groups is not None))
    validation = _publish_evaluation(root, source, images_ref, f"{dataset_id}-val", "validation", class_names,
                                     list(outcome.train), groups, report)
    test = _publish_evaluation(root, source, images_ref, f"{dataset_id}-test", "test", class_names,
                               list(outcome.validation), groups, report)
    return [validation, test], report
