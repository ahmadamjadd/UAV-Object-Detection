"""Validated training settings, grouped for the website (V1 basics, expanded in V3).

Each setting carries metadata in its JSON schema: its ``group`` (how the form
is organised), a plain ``description``, and the Ultralytics argument it is sent
to (``ultralytics``). The website builds its form from this schema, so there is
no separate frontend list. Defaults equal Ultralytics' own defaults except the
V1 basics (epochs, batch size, image size, seed), which keep their V1 values.

Rules between settings are checked here too, so impossible combinations fail
before a job is created. For example ``optimizer="auto"`` picks its own
learning rate and momentum, so setting those with "auto" is refused.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

OPTIMIZERS = ("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSprop", "Adamax")
# Ultralytics' values when a named optimizer is used and nothing else is given.
OPTIMIZER_DEFAULTS = {"lr0": 0.01, "momentum": 0.937, "warmup_bias_lr": 0.1}
GROUPS = {
    "training": "Training", "early_stopping": "Early stopping", "optimizer": "Optimizer and learning rate", "loss": "Loss weights",
    "augmentation": "Augmentation", "runtime": "Runtime",
}


# A setting with its form group, description and the Ultralytics argument it maps to.
def setting(default, group: str, description: str, ultralytics: str | None = None, **limits):
    extra = {"group": group, "ultralytics": ultralytics}
    return Field(default=default, description=description, json_schema_extra=extra, **limits)


# Every user-editable training setting; values are fixed once a job is created.
class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True, allow_inf_nan=False)

    # Training basics
    epochs: int = setting(3, "training", "Passes over the training images", "epochs", gt=0, le=10_000)
    batch_size: int = setting(16, "training", "Images per training step", "batch", gt=0, le=1024)
    image_size: int = setting(640, "training", "Training/evaluation image size in pixels (multiple of 32)",
                              "imgsz", ge=32, le=4096, multiple_of=32)
    seed: int = setting(42, "training", "Random seed for repeatable training", "seed", ge=0, le=4_294_967_295)
    # Early stopping: after each epoch YOLO scores the validation images (mAP50-95); when that score
    # has not improved for `patience` epochs, training stops and the best checkpoint so far is kept.
    early_stopping: bool = setting(True, "early_stopping",
                                   "Stop when the validation score (mAP50-95) stops improving; the best checkpoint is kept")
    patience: int = setting(100, "early_stopping", "Epochs without improvement before stopping "
                            "(if this is ≥ epochs, it never triggers)", "patience", ge=1, le=10_000)
    fraction: float = setting(1.0, "training", "Share of the training images to use (1 = all)",
                              "fraction", gt=0, le=1)
    freeze: int = setting(0, "training", "Freeze this many first model layers (0 = train all)",
                          "freeze", ge=0, le=50)

    # Optimizer and learning rate
    optimizer: Literal[*OPTIMIZERS] = setting(
        "auto", "optimizer", "Optimizer; 'auto' chooses one and its learning rate itself", "optimizer")
    lr0: float | None = setting(None, "optimizer", "Initial learning rate (not used with 'auto')",
                                "lr0", gt=0, le=1)
    lrf: float = setting(0.01, "optimizer", "Final learning rate as a fraction of the initial one",
                         "lrf", gt=0, le=1)
    momentum: float | None = setting(None, "optimizer", "Momentum / Adam beta1 (not used with 'auto')",
                                     "momentum", ge=0, lt=1)
    weight_decay: float = setting(0.0005, "optimizer", "Weight decay (L2 regularisation)",
                                  "weight_decay", ge=0, le=0.1)
    warmup_epochs: float = setting(3.0, "optimizer", "Epochs of gradual learning-rate warm-up",
                                   "warmup_epochs", ge=0, le=100)
    warmup_momentum: float = setting(0.8, "optimizer", "Momentum at the start of warm-up",
                                     "warmup_momentum", ge=0, lt=1)
    warmup_bias_lr: float | None = setting(None, "optimizer", "Bias learning rate during warm-up (not used with 'auto')",
                                           "warmup_bias_lr", ge=0, le=1)
    cos_lr: bool = setting(False, "optimizer", "Use a cosine learning-rate schedule", "cos_lr")

    # Loss weights
    box: float = setting(7.5, "loss", "Weight of the box-position loss", "box", gt=0, le=100)
    cls: float = setting(0.5, "loss", "Weight of the classification loss", "cls", gt=0, le=100)
    dfl: float = setting(1.5, "loss", "Weight of the distribution focal loss", "dfl", gt=0, le=100)

    # Augmentation (applied to training images only)
    hsv_h: float = setting(0.015, "augmentation", "Random hue change (fraction)", "hsv_h", ge=0, le=1)
    hsv_s: float = setting(0.7, "augmentation", "Random saturation change (fraction)", "hsv_s", ge=0, le=1)
    hsv_v: float = setting(0.4, "augmentation", "Random brightness change (fraction)", "hsv_v", ge=0, le=1)
    degrees: float = setting(0.0, "augmentation", "Random rotation (degrees)", "degrees", ge=0, le=180)
    translate: float = setting(0.1, "augmentation", "Random shift (fraction of image)", "translate", ge=0, le=0.9)
    scale: float = setting(0.5, "augmentation", "Random zoom (fraction)", "scale", ge=0, le=0.9)
    shear: float = setting(0.0, "augmentation", "Random shear (degrees)", "shear", ge=0, le=90)
    perspective: float = setting(0.0, "augmentation", "Random perspective", "perspective", ge=0, le=0.001)
    flipud: float = setting(0.0, "augmentation", "Chance of an upside-down flip", "flipud", ge=0, le=1)
    fliplr: float = setting(0.5, "augmentation", "Chance of a left-right flip", "fliplr", ge=0, le=1)
    bgr: float = setting(0.0, "augmentation", "Chance of swapping colour channels", "bgr", ge=0, le=1)
    mosaic: float = setting(1.0, "augmentation", "Chance of combining 4 images into one", "mosaic", ge=0, le=1)
    close_mosaic: int = setting(10, "augmentation", "Turn mosaic off for the last N epochs",
                                "close_mosaic", ge=0, le=10_000)
    mixup: float = setting(0.0, "augmentation", "Chance of blending two images", "mixup", ge=0, le=1)
    cutmix: float = setting(0.0, "augmentation", "Chance of pasting a patch of another image",
                            "cutmix", ge=0, le=1)

    # Runtime
    device: str = setting("cpu", "runtime", "Where training runs: the CPU or a detected GPU (cuda:0, cuda:1, …)",
                          "device", pattern=r"^(cpu|cuda:[0-9]{1,2})$")
    amp: bool = setting(True, "runtime", "Mixed precision on GPU: faster and uses less memory (ignored on CPU)", "amp")
    workers: int = setting(2, "runtime", "Data-loading worker processes", "workers", ge=0, le=16)
    cache: Literal["none", "ram"] = setting(
        "none", "runtime", "Keep decoded training images in memory ('ram') for speed", "cache")
    deterministic: bool = setting(True, "runtime", "Use repeatable (slightly slower) operations", "deterministic")

    # Fill in a named optimizer's defaults, so the saved settings show the values really used.
    @model_validator(mode="before")
    @classmethod
    def optimizer_defaults(cls, values):
        if isinstance(values, dict) and values.get("optimizer", "auto") != "auto":
            values = {**{k: v for k, v in OPTIMIZER_DEFAULTS.items() if values.get(k) is None},
                      **{k: v for k, v in values.items() if v is not None}}
        return values

    # Refuse combinations the runtime would silently ignore or that cannot work.
    @model_validator(mode="after")
    def check_combinations(self) -> Self:
        if self.optimizer == "auto":
            ignored = [name for name in OPTIMIZER_DEFAULTS if getattr(self, name) is not None]
            if ignored:
                raise ValueError(f"optimizer 'auto' chooses its own {', '.join(ignored)}; "
                                 "pick a named optimizer (e.g. SGD or AdamW) to set them")
        return self


# The Ultralytics arguments for these settings (application-managed ones are added by the trainer).
def ultralytics_arguments(settings: TrainingSettings) -> dict:
    arguments = {}
    for name, field in TrainingSettings.model_fields.items():
        key = (field.json_schema_extra or {}).get("ultralytics")
        value = getattr(settings, name)
        if key is None or value is None:
            continue  # None: let the runtime choose (only possible with optimizer "auto")
        arguments[key] = value
    arguments["cache"] = "ram" if settings.cache == "ram" else False
    arguments["freeze"] = settings.freeze or None
    arguments["patience"] = settings.patience if settings.early_stopping else 0  # 0 = never stop early
    return arguments
