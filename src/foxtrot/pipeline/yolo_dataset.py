"""Write a per-job YOLO dataset config instead of rewriting the root data.yaml.

The legacy ``src/train.py`` writes ``<checkout>/data.yaml`` with the default
Foxtrot classes. Website jobs instead write ``<job>/output/data.yaml`` from the
dataset's recorded class mapping, so two jobs or datasets never share one file.
"""

from pathlib import Path

import yaml

from foxtrot.persistence.files import write_new_text

from .contracts import DatasetInput


def yolo_data_config(dataset: DatasetInput) -> dict:
    metadata = dataset.metadata
    return {
        "path": dataset.root,
        "train": metadata.train_images_ref,
        "val": metadata.validation_images_ref,
        "names": dict(enumerate(metadata.class_names)),
    }


def write_yolo_data_yaml(dataset: DatasetInput, path: Path) -> Path:
    """Create the file exclusively; an existing config is never overwritten."""
    write_new_text(path, yaml.safe_dump(yolo_data_config(dataset), sort_keys=False))
    return path
