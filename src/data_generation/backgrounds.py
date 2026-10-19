"""Validate background inputs before starting destructive dataset generation."""

from pathlib import Path
import warnings

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def read_background(path: str) -> np.ndarray:
    """Decode one color image, raising a useful error if an input changed."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Background image does not exist or is not a file: {source}")
    try:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"Cannot decode background image: {source}") from error
    if image is None or image.size == 0:
        raise ValueError(f"Cannot decode background image: {source}")
    return image


def discover_backgrounds(directory: str) -> list[str]:
    """Return sorted full paths of readable PNG/JPEG images, without caching pixels.

    Unsupported files and subdirectories are ignored. Corrupt supported files
    produce one summary warning. An unusable input directory fails immediately,
    so callers can validate before deleting any previous generated dataset.
    """
    root = Path(directory).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Background directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Background path is not a directory: {root}")
    usable = []
    skipped = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            read_background(str(candidate))
        except (ValueError, FileNotFoundError):
            skipped.append(candidate.name)
            continue
        usable.append(str(candidate))
    if not usable:
        raise ValueError(
            f"No readable background images in {root}. "
            "Provide at least one valid PNG, JPG or JPEG image "
            f"({len(skipped)} supported files could not be decoded)."
        )
    if skipped:
        preview = ", ".join(skipped[:5])
        suffix = ", ..." if len(skipped) > 5 else ""
        warnings.warn(
            f"Skipped {len(skipped)} unreadable background image(s): {preview}{suffix}",
            RuntimeWarning,
            stacklevel=2,
        )
    return usable
