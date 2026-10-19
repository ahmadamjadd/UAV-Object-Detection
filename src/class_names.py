"""Shared detection classes; their order defines the dataset's numeric labels.

Keep this module free of generation/training dependencies and resource reads.
Changing this order changes the meaning of existing YOLO label files.
"""

from types import MappingProxyType

SHAPES = ("Hexagon", "Triangle")
CLASS_NAMES = MappingProxyType(dict(enumerate(SHAPES)))
