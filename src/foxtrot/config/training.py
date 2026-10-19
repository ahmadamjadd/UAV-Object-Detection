"""Validated settings for the first training workflow."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingSettings(BaseModel):
    """Accept supported training settings and reject invalid inputs."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )

    epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=16, gt=0)
    image_size: int = Field(default=640, gt=0)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    device: Literal["cpu"] = "cpu"