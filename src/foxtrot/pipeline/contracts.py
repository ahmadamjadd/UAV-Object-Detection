"""H6 explicit-input pipeline interfaces.

Every step receives its configuration, input locations, and output folder as
arguments and returns a typed result. Nothing here reads params.yaml, writes the
root data.yaml, searches runs/, or imports the generator, YOLO, or MLflow.
"""

import math
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from foxtrot.config.checkpoint import CheckpointReference
from foxtrot.config.dataset import DatasetMetadata
from foxtrot.config.evaluation import EvaluationResult
from foxtrot.config.job import JobSpec


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, validate_default=True, allow_inf_nan=False,
    )


def _absolute(value: str) -> str:
    if not value or "\x00" in value or not Path(value).is_absolute():
        raise ValueError("Pipeline paths must be absolute")
    return value


JOB_ID = r"^[0-9a-f]{32}$"
SHA256 = r"^[0-9a-f]{64}$"


class DatasetInput(Contract):
    """A registered dataset: its recorded metadata and absolute read-only root."""

    metadata: DatasetMetadata
    root: str

    @field_validator("root")
    @classmethod
    def check_root_absolute(cls, value: str) -> str:
        return _absolute(value)


# ---------------------------------------------------------------- generation

class GenerationInput(Contract):
    """Minimum explicit generator inputs; the full recipe is completed in V2.

    ``images_per_class`` is the legacy ``n`` (two classes x 100 = 200 images).
    ``canvas_size`` is the legacy square ``size``; YOLO's training image size is
    a separate training setting and is never inferred from it.
    """

    class_names: tuple[str, ...] = Field(min_length=1)
    images_per_class: int = Field(gt=0)
    canvas_size: int = Field(gt=0)
    seed: int = Field(ge=0, le=4_294_967_295)
    output_directory: str

    @field_validator("output_directory")
    @classmethod
    def check_output_absolute(cls, value: str) -> str:
        return _absolute(value)


class DatasetManifest(Contract):
    metadata: DatasetMetadata
    image_count: int = Field(ge=0)


# ----------------------------------------------------------------- splitting

class SplitPolicy(Contract):
    train_ratio: float = Field(gt=0, lt=1)
    validation_ratio: float = Field(gt=0, lt=1)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> Self:
        if abs(self.train_ratio + self.validation_ratio - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1")
        return self


class SplitInput(Contract):
    dataset: DatasetInput
    policy: SplitPolicy
    output_directory: str

    @field_validator("output_directory")
    @classmethod
    def check_output_absolute(cls, value: str) -> str:
        return _absolute(value)


class SplitManifest(Contract):
    dataset: DatasetMetadata
    policy: SplitPolicy
    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)


# ------------------------------------------------------ training/evaluation

class TrainingInput(Contract):
    """Everything a trainer may use. Outputs go only under ``output_directory``."""

    job_id: str = Field(pattern=JOB_ID)
    resolved: JobSpec
    resolved_sha256: str = Field(pattern=SHA256)
    dataset: DatasetInput
    data_yaml: str
    job_directory: str
    output_directory: str

    @field_validator("data_yaml", "job_directory", "output_directory")
    @classmethod
    def check_paths_absolute(cls, value: str) -> str:
        return _absolute(value)


class TrainingResult(Contract):
    """The exact checkpoint chosen by training, relative to the job directory."""

    checkpoint: CheckpointReference
    epochs_completed: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    fixture: bool = False

    @field_validator("metrics")
    @classmethod
    def finite_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if not all(math.isfinite(number) for number in value.values()):
            raise ValueError("Training metrics must be finite")
        return value


class EvaluationInput(Contract):
    job_id: str = Field(pattern=JOB_ID)
    checkpoint: CheckpointReference
    job_directory: str
    dataset: DatasetInput

    @field_validator("job_directory")
    @classmethod
    def check_directory_absolute(cls, value: str) -> str:
        return _absolute(value)


class PipelineResult(Contract):
    """The child's report; the supervisor re-verifies it before trusting it."""

    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=JOB_ID)
    adapter: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    training: TrainingResult
    evaluation: EvaluationResult

    @model_validator(mode="after")
    def evaluation_scores_selected_checkpoint(self) -> Self:
        if self.evaluation.checkpoint != self.training.checkpoint:
            raise ValueError("Evaluation must score the checkpoint selected by training")
        return self


# ---------------------------------------------------------------- reporting

class PipelineCancelled(Exception):
    """Raised by an adapter after it honours a cancellation request."""


class PipelineFailure(Exception):
    """A safe, displayable adapter failure. Never include credentials."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ProgressReporter(Protocol):
    def progress(self, payload: dict) -> None: ...

    def log(self, payload: dict) -> None: ...

    def wait_cancelled(self, seconds: float = 0.0) -> bool:
        """Wait up to ``seconds``; return True once cancellation was requested."""
        ...


# ----------------------------------------------------------------- adapters

class DatasetGenerator(Protocol):
    def generate(self, request: GenerationInput, reporter: ProgressReporter) -> DatasetManifest: ...


class DatasetSplitter(Protocol):
    def split(self, request: SplitInput, reporter: ProgressReporter) -> SplitManifest: ...


class Trainer(Protocol):
    def train(self, request: TrainingInput, reporter: ProgressReporter) -> TrainingResult: ...


class Evaluator(Protocol):
    def evaluate(self, request: EvaluationInput, reporter: ProgressReporter) -> EvaluationResult: ...
