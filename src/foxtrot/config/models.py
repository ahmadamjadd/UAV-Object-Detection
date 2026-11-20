"""The detection models Foxtrot can train (V3 model registry).

Every entry was checked against the pinned Ultralytics version: its
architecture file builds offline and its official pretrained weights exist in
the same release Ultralytics itself downloads from. Only detection is offered,
because Foxtrot datasets contain bounding boxes.
"""

from dataclasses import dataclass

# The Ultralytics release that holds the official pretrained weights.
WEIGHTS_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

# Model families, in the order the website lists them.
FAMILIES = {"yolov8": "YOLOv8", "yolo11": "YOLO11", "yolo12": "YOLO12", "yolo26": "YOLO26"}

# Sizes within a family: n is the smallest and fastest, x the largest and slowest.
VARIANTS = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "extra large"}


# One trainable model: its ID (e.g. "yolo11s"), family, size, and where its files come from.
@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    family: str
    variant: str
    display_name: str
    runtime: str = "ultralytics"
    task: str = "detect"

    # The architecture file used when training from scratch (random starting weights).
    @property
    def architecture(self) -> str:
        return f"{self.model_id}.yaml"

    # The official download for its pretrained weights.
    @property
    def weights_url(self) -> str:
        return f"{WEIGHTS_RELEASE}/{self.model_id}.pt"


# Build every family x size combination, e.g. "yolov8n" ... "yolo26x".
def _build_registry() -> dict[str, ModelEntry]:
    entries = {}
    for family, family_name in FAMILIES.items():
        for variant, variant_name in VARIANTS.items():
            model_id = f"{family}{variant}"
            entries[model_id] = ModelEntry(model_id=model_id, family=family, variant=variant,
                                           display_name=f"{family_name} {variant_name}")
    return entries


MODELS = _build_registry()
MODEL_IDS = tuple(MODELS)  # "yolov8n" first: the V1 default


# Look up one model, with a clear error for unknown IDs.
def get_model(model_id: str) -> ModelEntry:
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(f"Unknown model: {model_id}") from None
