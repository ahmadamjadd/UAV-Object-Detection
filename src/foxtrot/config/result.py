"""Final execution results and their tracking associations."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .checkpoint import CheckpointReference
from .evaluation import EvaluationResult


class JobResult(BaseModel):
    """Describe how an execution ended and what it produced."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )

    schema_version: int = Field(default=1, ge=1, le=1)

    job_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )
    outcome: Literal["finished", "failed", "cancelled", "interrupted"]

    checkpoint: CheckpointReference | None = None
    evaluation: EvaluationResult | None = None
    error_message: str | None = Field(default=None, pattern=r"\S")

    tracking_destination_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )
    mlflow_run_id: str | None = Field(default=None, pattern=r"^\S+$")

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.outcome == "finished":
            if self.checkpoint is None or self.evaluation is None:
                raise ValueError(
                    "Finished jobs require a checkpoint and evaluation"
                )
            if self.error_message is not None:
                raise ValueError(
                    "Finished jobs must not contain an execution error"
                )

        if self.outcome in {"failed", "interrupted"}:
            if self.error_message is None:
                raise ValueError(
                    "Failed or interrupted jobs require an error message"
                )

        if self.evaluation is not None:
            if self.evaluation.checkpoint != self.checkpoint:
                raise ValueError(
                    "Evaluation must identify the job's selected checkpoint"
                )

        return self