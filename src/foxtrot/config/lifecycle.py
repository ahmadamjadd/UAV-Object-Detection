"""Legal job state changes, independent of storage and training libraries."""

from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExecutionState(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class TrackingState(StrEnum):
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    RETRYING = "retrying"
    FAILED = "failed"


ACTIVE_STATES = frozenset({
    ExecutionState.PREPARING, ExecutionState.TRAINING, ExecutionState.EVALUATING,
})
TERMINAL_STATES = frozenset({
    ExecutionState.FINISHED, ExecutionState.FAILED,
    ExecutionState.CANCELLED, ExecutionState.INTERRUPTED,
})
_STOP = {ExecutionState.FAILED, ExecutionState.CANCELLED, ExecutionState.INTERRUPTED}
EXECUTION_TRANSITIONS = MappingProxyType({
    ExecutionState.DRAFT: frozenset({ExecutionState.VALIDATING, ExecutionState.CANCELLED}),
    ExecutionState.VALIDATING: frozenset({ExecutionState.QUEUED, ExecutionState.FAILED, ExecutionState.CANCELLED}),
    ExecutionState.QUEUED: frozenset({ExecutionState.PREPARING, ExecutionState.FAILED, ExecutionState.CANCELLED}),
    ExecutionState.PREPARING: frozenset({ExecutionState.TRAINING} | _STOP),
    ExecutionState.TRAINING: frozenset({ExecutionState.EVALUATING} | _STOP),
    ExecutionState.EVALUATING: frozenset({ExecutionState.FINISHED} | _STOP),
    **{state: frozenset() for state in TERMINAL_STATES},
})
TRACKING_TRANSITIONS = MappingProxyType({
    TrackingState.PENDING: frozenset({TrackingState.SYNCING}),
    TrackingState.SYNCING: frozenset({TrackingState.SYNCED, TrackingState.RETRYING, TrackingState.FAILED}),
    TrackingState.RETRYING: frozenset({TrackingState.SYNCING, TrackingState.FAILED}),
    TrackingState.FAILED: frozenset({TrackingState.RETRYING}),
    TrackingState.SYNCED: frozenset(),
})


def validate_execution_transition(current: ExecutionState, target: ExecutionState) -> None:
    """Accept legal transitions and idempotent repeats, rejecting plain strings."""
    if not isinstance(current, ExecutionState) or not isinstance(target, ExecutionState):
        raise TypeError("Execution transitions require ExecutionState values")
    if current != target and target not in EXECUTION_TRANSITIONS[current]:
        raise ValueError(f"Invalid execution transition: {current} -> {target}")


def validate_tracking_transition(current: TrackingState, target: TrackingState) -> None:
    if not isinstance(current, TrackingState) or not isinstance(target, TrackingState):
        raise TypeError("Tracking transitions require TrackingState values")
    if current != target and target not in TRACKING_TRANSITIONS[current]:
        raise ValueError(f"Invalid tracking transition: {current} -> {target}")


class ErrorDetails(BaseModel):
    """A durable, safe-to-display reason; callers must exclude credentials."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True)
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    message: str = Field(min_length=1, max_length=4096)

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Error message must not be blank")
        return value
