"""Durable tracking operations, independent of MLflow and network access."""

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, validate_default=True, allow_inf_nan=False,
    )


class MetricPoint(_Contract):
    """One metric value at one step, as an upload record rather than a live call."""

    key: str = Field(pattern=r"^[A-Za-z0-9_./ -]{1,250}$")
    value: float
    step: int = Field(ge=0)

    @field_validator("value")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Metric values must be finite")
        return value


class ArtifactUpload(_Contract):
    """Upload a registered artifact by name; its hash is rechecked before upload."""

    name: str = Field(min_length=1, max_length=200)
    artifact_path: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")


TerminalStatus = Literal["FINISHED", "FAILED", "KILLED"]


class TrackingOperation(_Contract):
    """One ordered step of a job's pending upload."""

    operation: Literal["create_run", "log_params", "log_metrics", "log_artifact", "set_terminated"]
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def check_payload(self) -> Self:
        payload = self.payload
        if self.operation == "create_run":
            name, tags = payload.get("run_name"), payload.get("tags")
            if set(payload) != {"run_name", "tags"} or not isinstance(name, str) or not name.strip():
                raise ValueError("create_run requires run_name and tags")
            if not isinstance(tags, dict) or not all(isinstance(v, str) for v in tags.values()):
                raise ValueError("Run tags must be strings")
        elif self.operation == "log_params":
            params = payload.get("params")
            if set(payload) != {"params"} or not isinstance(params, dict) or not all(
                isinstance(v, str) for v in params.values()
            ):
                raise ValueError("log_params requires string parameter values")
        elif self.operation == "log_metrics":
            if set(payload) != {"metrics"} or not isinstance(payload["metrics"], list):
                raise ValueError("log_metrics requires a metric list")
            for point in payload["metrics"]:
                MetricPoint.model_validate(point)
        elif self.operation == "log_artifact":
            ArtifactUpload.model_validate(payload)
        elif set(payload) != {"status"} or payload["status"] not in {"FINISHED", "FAILED", "KILLED"}:
            raise ValueError("set_terminated requires FINISHED, FAILED, or KILLED")
        return self
