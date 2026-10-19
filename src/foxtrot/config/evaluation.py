"""Evaluation scores tied to an exact checkpoint and dataset."""

from pydantic import BaseModel, ConfigDict, Field

from .checkpoint import CheckpointReference


class EvaluationResult(BaseModel):
    """Describe measured detection performance and its inputs."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    dataset_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    checkpoint: CheckpointReference

    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    map50: float = Field(ge=0, le=1)
    map50_95: float = Field(ge=0, le=1)