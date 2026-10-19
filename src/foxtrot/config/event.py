"""Versioned progress events for local jobs."""

import json
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
)


class ProgressEvent(BaseModel):
    """Describe one update without storing or transmitting it."""

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
    sequence: int = Field(ge=1)
    timestamp: AwareDatetime

    event_type: Literal["state_changed", "progress", "log", "error"]
    payload: dict[str, JsonValue]

    @field_validator("payload")
    @classmethod
    def check_payload(
        cls,
        value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        json.dumps(value, allow_nan=False)
        return value