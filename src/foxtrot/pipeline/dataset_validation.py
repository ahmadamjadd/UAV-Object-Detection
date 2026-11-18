"""Check a YOLO detection dataset before anyone trains on it.

Reads only. Reports every problem found instead of stopping at the first one,
so a member can fix their dataset in one pass.
"""

from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_ISSUES = 200  # keep reports small; counts stay exact
TOLERANCE = 1e-6  # allow tiny rounding past the image edge


# Base for the report types: unknown fields are rejected and values cannot be changed.
class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# One problem found: which split, file and line it is in, and what is wrong.
class Issue(Contract):
    split: str
    file: str | None = None
    line: int | None = None
    message: str


# Counts for one split (e.g. train): images, how many have labels, objects per class.
class SplitStats(Contract):
    images: int = 0
    labelled_images: int = 0
    missing_labels: int = 0
    objects_per_class: dict[str, int] = Field(default_factory=dict)


# The full result of a check: valid or not, the problems found, and per-split counts.
class DatasetReport(Contract):
    valid: bool
    errors: list[Issue]
    warnings: list[Issue]
    error_count: int
    warning_count: int
    splits: dict[str, SplitStats]


# Find the labels folder for an images folder (YOLO rule: .../images -> .../labels).
def labels_directory(images_directory: Path) -> Path:
    parts = list(images_directory.parts)
    if "images" not in parts:
        raise ValueError("Image folder path must contain an 'images' folder")
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts)


# Gathers errors and warnings while checking; keeps at most MAX_ISSUES examples but counts all.
class _Collector:
    # Start with empty lists and zero counts.
    def __init__(self):
        self.errors, self.warnings = [], []
        self.error_count = self.warning_count = 0

    # Record a problem that makes the dataset invalid.
    def error(self, **issue):
        self.error_count += 1
        if len(self.errors) < MAX_ISSUES:
            self.errors.append(Issue(**issue))

    # Record something worth knowing that does not block training.
    def warning(self, **issue):
        self.warning_count += 1
        if len(self.warnings) < MAX_ISSUES:
            self.warnings.append(Issue(**issue))


# Check every line of one label file and count the valid objects per class.
def _check_label_file(path: Path, split: str, class_count: int, issues: _Collector, counts: Counter):
    name = path.name
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        issues.error(split=split, file=name, message="Label file cannot be read as text")
        return
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            issues.error(split=split, file=name, line=number,
                         message=f"Expected 5 values (class x y width height), found {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError:
            issues.error(split=split, file=name, line=number, message="Values must be numbers; class must be a whole number")
            continue
        if not 0 <= class_id < class_count:
            issues.error(split=split, file=name, line=number,
                         message=f"Class {class_id} is outside 0–{class_count - 1}")
            continue
        if width <= 0 or height <= 0:
            issues.error(split=split, file=name, line=number, message="Box width and height must be positive")
            continue
        if (x - width / 2 < -TOLERANCE or x + width / 2 > 1 + TOLERANCE
                or y - height / 2 < -TOLERANCE or y + height / 2 > 1 + TOLERANCE):
            issues.error(split=split, file=name, line=number, message="Box extends outside the image (values must be 0–1)")
            continue
        counts[class_id] += 1

# Main entry point: check every image and label in each split and return a report.
# `splits` maps a split name to its images folder, e.g. {"train": "train/images"}.
# An image with no label file is a warning ("background") or an error, by policy.
def validate_dataset(
    root: Path, class_names: tuple[str, ...], splits: dict[str, str],
    *, missing_labels: Literal["background", "error"] = "background",
) -> DatasetReport:
    issues = _Collector()
    stats = {}
    seen: dict[str, str] = {}  # image content hash -> split it first appeared in
    for split, reference in splits.items():
        relative = PurePosixPath(reference)
        if relative.is_absolute() or ".." in relative.parts:
            issues.error(split=split, message="Split folder must be a relative path without '..'")
            stats[split] = SplitStats()
            continue
        images_directory = root / reference
        try:
            label_directory = labels_directory(images_directory)
        except ValueError as error:
            issues.error(split=split, message=str(error))
            stats[split] = SplitStats()
            continue
        if not images_directory.is_dir():
            issues.error(split=split, message=f"Image folder not found: {reference}")
            stats[split] = SplitStats()
            continue
        images = sorted(p for p in images_directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        counts, labelled, missing = Counter(), 0, 0
        for image in images:
            try:
                with Image.open(image) as opened:
                    opened.verify()  # detects truncated or corrupt files
            except (UnidentifiedImageError, OSError, SyntaxError):
                issues.error(split=split, file=image.name, message="Image cannot be read")
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen:
                place = "within" if seen[digest] == split else f"in both {seen[digest]} and"
                issues.warning(split=split, file=image.name, message=f"Duplicate image {place} {split}")
            else:
                seen[digest] = split
            label = label_directory / f"{image.stem}.txt"
            if label.is_file():
                labelled += 1
                _check_label_file(label, split, len(class_names), issues, counts)
            else:
                missing += 1
                report = issues.error if missing_labels == "error" else issues.warning
                report(split=split, file=image.name, message="No label file; treated as an image with no objects"
                       if missing_labels == "background" else "No label file")
        if label_directory.is_dir():
            stems = {image.stem for image in images}
            for orphan in sorted(p for p in label_directory.glob("*.txt") if p.stem not in stems):
                issues.warning(split=split, file=orphan.name, message="Label file has no matching image")
        if not images:
            issues.error(split=split, message="Split has no images")
        stats[split] = SplitStats(
            images=len(images), labelled_images=labelled, missing_labels=missing,
            objects_per_class={class_names[k]: counts[k] for k in range(len(class_names))},
        )
    return DatasetReport(
        valid=issues.error_count == 0, errors=issues.errors, warnings=issues.warnings,
        error_count=issues.error_count, warning_count=issues.warning_count, splits=stats,
    )

