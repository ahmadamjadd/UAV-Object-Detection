"""A training-job specification: dataset, model, how it starts, settings, destination."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import MODEL_IDS
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

    # Any model from the V3 registry (config/models.py); yolov8n stays the default.
    model_id: Literal[*MODEL_IDS] = "yolov8n"

    # How the model's weights start (V3): official pretrained weights, random weights,
    # or the best checkpoint of an earlier finished job in this workspace (same model).
    initialization: Literal["pretrained", "scratch", "checkpoint"] = "pretrained"
    checkpoint_job_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    tracking_destination_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )

    training: TrainingSettings = Field(
        default_factory=TrainingSettings,
    )

    # A checkpoint job is needed exactly when starting from a checkpoint.
    @model_validator(mode="after")
    def checkpoint_matches_initialization(self) -> Self:
        if self.initialization == "checkpoint" and self.checkpoint_job_id is None:
            raise ValueError("Choose the earlier job whose checkpoint to start from")
        if self.initialization != "checkpoint" and self.checkpoint_job_id is not None:
            raise ValueError("checkpoint_job_id is only used when initialization is 'checkpoint'")
        return self
