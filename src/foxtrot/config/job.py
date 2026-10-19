"""The first version of a basic training-job specification."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .training import TrainingSettings


class JobSpec(BaseModel):
    """Describe a training request without starting any work."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )

    schema_version: int = Field(default=1, ge=1, le=1)

    dataset_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )

    model_id: Literal["yolov8n"] = "yolov8n"

    tracking_destination_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )

    training: TrainingSettings = Field(
        default_factory=TrainingSettings,
    )