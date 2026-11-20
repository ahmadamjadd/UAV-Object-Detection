"""Validated settings for generating a Foxtrot shapes dataset (V2).

These mirror the ``data_generation`` block of params.yaml, with clearer names.
``images_per_class`` is the old ``n`` and ``canvas_size`` is the old ``size``:
the generated picture size, which is separate from YOLO's training image size.
"""

import hashlib
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


# A strict, read-only base so every recipe field must be given with the right type.
class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, validate_default=True, allow_inf_nan=False,
    )


# A lower/upper pair of whole numbers, such as the blur kernel range.
class IntRange(_Strict):
    low: int
    high: int

    # The lower bound may not be above the upper bound.
    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError("Range low must not exceed high")
        return self


# A lower/upper pair of decimal numbers, such as the shape scale range.
class FloatRange(_Strict):
    low: float
    high: float

    # The lower bound may not be above the upper bound.
    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError("Range low must not exceed high")
        return self


# Every setting the generator uses; the defaults equal today's params.yaml.
class GenerationRecipe(_Strict):
    images_per_class: int = Field(default=100, gt=0, le=100_000)
    canvas_size: int = Field(default=320, ge=64, le=4096)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    shape_scale: FloatRange = FloatRange(low=0.2, high=0.6)
    noise: int = Field(default=50, ge=0, le=255)
    blur: IntRange = IntRange(low=0, high=4)
    brightness_difference: int = Field(default=100, ge=0, le=255)
    contrast: FloatRange = FloatRange(low=0.85, high=1.2)
    brightness: IntRange = IntRange(low=-20, high=20)
    saturation: IntRange = IntRange(low=-10, high=0)
    desaturate: bool = False
    add_letter: bool = True
    letter_probability: float = Field(default=0.5, ge=0, le=1)
    add_light_gradient: bool = False
    light_probability: float = Field(default=0.5, ge=0, le=1)
    light_opacity: float = Field(default=0.2, ge=0, le=1)

    # Check limits that need more than one field or a range's own values.
    @model_validator(mode="after")
    def check_ranges(self) -> Self:
        if not 0 < self.shape_scale.low <= self.shape_scale.high <= 1:
            raise ValueError("Shape scale must be within (0, 1]")
        if self.blur.low < 0 or self.blur.high > 50:
            raise ValueError("Blur must be within 0..50")
        if self.contrast.low <= 0:
            raise ValueError("Contrast must be positive")
        if self.saturation.low < -255 or self.saturation.high > 255:
            raise ValueError("Saturation must be within -255..255")
        return self

    # A stable fingerprint of the recipe: same settings, same hash, any key order.
    def sha256(self) -> str:
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# How generated images are divided into train and validation sets.
# With group_by_background, every image made from one background photo lands in
# the same split, so validation never shows a background the model trained on.
class SplitSettings(_Strict):
    train_ratio: float = Field(default=0.8, ge=0.05, le=0.95)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    group_by_background: bool = True


# Everything one "generate a dataset" request needs: settings, split and backgrounds folder.
class GenerationRequest(_Strict):
    recipe: GenerationRecipe = GenerationRecipe()
    split: SplitSettings = SplitSettings()
    backgrounds: str = Field(min_length=1, max_length=4096)

    # The backgrounds folder must be an absolute path on this computer.
    @model_validator(mode="after")
    def absolute_backgrounds(self) -> Self:
        if "\x00" in self.backgrounds or not self.backgrounds.startswith("/"):
            raise ValueError("Backgrounds folder must be an absolute path")
        return self
