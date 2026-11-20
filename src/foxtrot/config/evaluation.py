"""Evaluation scores tied to an exact checkpoint and dataset."""

from pydantic import BaseModel, ConfigDict, Field

from .checkpoint import CheckpointReference

# The fixed evaluation configuration (V4). Every score Foxtrot reports uses exactly these
# settings plus the checkpoint's own training image size, so scores stay comparable and a
# tuning study can reference them by name. Changing them requires a new name.
EVALUATION_CONFIG_NAME = "foxtrot-eval-v1"
EVALUATION_CONFIG = {"split": "val", "batch": 16, "conf": 0.001, "iou": 0.7, "max_det": 300}


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

    # V4: mAP50-95 per class name, when the evaluator provides it.
    per_class_map50_95: dict[str, float] | None = None