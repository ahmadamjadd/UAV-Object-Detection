"""Fixed runner messages and explicit inputs; never arbitrary shell commands."""

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator, model_validator

from foxtrot.config.job import JobSpec
from foxtrot.config.lifecycle import ErrorDetails
from foxtrot.pipeline.contracts import DatasetInput
from foxtrot.pipeline.registry import PipelineSettings


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True)


class FixtureSettings(Contract):
    steps: int = Field(default=3, ge=1, le=100)
    step_seconds: float = Field(default=0.05, ge=0.0, le=60.0, allow_inf_nan=False)
    fail_at_step: int | None = Field(default=None, ge=1, le=100)
    spawn_child: bool = False
    ignore_cancel: bool = False
    child_ignores_term: bool = False
    child_new_group: bool = False
    output_lines: int = Field(default=0, ge=0, le=5000)


class ExecutionRequest(Contract):
    schema_version: Literal[1] = 1
    runner: Literal["fixture"] = "fixture"
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    job_directory: str
    output_directory: str
    resolved_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved: JobSpec
    fixture: FixtureSettings

    @field_validator("job_directory", "output_directory")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("Execution paths must be absolute")
        return value


class PipelineExecutionRequest(ExecutionRequest):
    """H6: run a code-registered adapter against a verified registered dataset."""

    runner: Literal["pipeline"] = "pipeline"
    fixture: None = None
    pipeline: PipelineSettings
    dataset: DatasetInput


REQUEST = TypeAdapter(Annotated[ExecutionRequest | PipelineExecutionRequest, Field(discriminator="runner")])


class Ready(Contract):
    kind: Literal["ready"] = "ready"


class Phase(Contract):
    kind: Literal["phase"] = "phase"
    phase: Literal["training", "evaluating"]


class Update(Contract):
    kind: Literal["update"] = "update"
    event_type: Literal["progress", "log", "error"] = "progress"
    payload: dict[str, JsonValue]

    @field_validator("payload")
    @classmethod
    def finite_payload(cls, value):
        json.dumps(value, allow_nan=False)
        return value


class Completed(Contract):
    kind: Literal["completed"] = "completed"
    outcome: Literal["finished", "failed", "cancelled"]
    error: ErrorDetails | None = None

    @model_validator(mode="after")
    def error_only_for_failure(self) -> Self:
        if self.error is not None and self.outcome != "failed":
            raise ValueError("Only a failed completion carries an error")
        return self


MESSAGE = TypeAdapter(Annotated[Ready | Phase | Update | Completed, Field(discriminator="kind")])
MAX_MESSAGE_BYTES = 65536
