"""Atomic state changes and durable reasons, separate from process supervision."""

from pathlib import Path

from foxtrot.config.lifecycle import (
    ACTIVE_STATES, TERMINAL_STATES, ErrorDetails, ExecutionState, TrackingState,
    validate_execution_transition, validate_tracking_transition,
)

from .database import connect_database
from .events import _append_event, _require_claim
from .jobs import JobRecord, _record, _required_row
from .processes import Liveness, ProcessIdentity, probe_execution_group


def _check_error(error: ErrorDetails | None) -> None:
    if error is not None and not isinstance(error, ErrorDetails):
        raise TypeError("Error must be an ErrorDetails value")


def transition_execution(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
    target: ExecutionState, *, claim_token: str | None = None,
    error: ErrorDetails | None = None,
    exit_code: int | None = None,
) -> JobRecord:
    """Persist one legal change and event together; never start a process."""
    _check_error(error)
    if not isinstance(target, ExecutionState):
        raise TypeError("Execution transitions require ExecutionState values")
    if exit_code is not None and type(exit_code) is not int:
        raise TypeError("Exit code must be an integer")
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _required_row(connection, job_id)
            current = ExecutionState(row["execution_state"])
            # The cancellation request and final outcome are arbitrated under
            # the same lock, including a request arriving just after child exit.
            if row["cancel_requested"] and target in TERMINAL_STATES:
                target, error = ExecutionState.CANCELLED, None
            validate_execution_transition(current, target)
            if row["claim_token"] is not None or current in ACTIVE_STATES:
                _require_claim(row, claim_token)
            if current == target:
                saved = _record(row)
                if error is not None and error != saved.error_details:
                    raise ValueError("A repeated transition cannot replace its saved error")
                if exit_code is not None and exit_code != saved.exit_code:
                    raise ValueError("A repeated transition cannot replace its exit code")
                return saved
            if target == ExecutionState.PREPARING:
                raise ValueError("Use claim_next_job to enter preparing")
            if target in {ExecutionState.FAILED, ExecutionState.INTERRUPTED} and error is None:
                raise ValueError("Failed and interrupted jobs require error details")
            if target not in {ExecutionState.FAILED, ExecutionState.INTERRUPTED, ExecutionState.CANCELLED} and error is not None:
                raise ValueError("This execution state cannot have an error")
            if target in TERMINAL_STATES:
                if row["process_launch_pending"] or row["recovery_required"]:
                    raise ValueError("Resolve uncertain execution ownership before finalizing")
                if row["process_pid"] is not None:
                    identity = ProcessIdentity(row["process_pid"], row["process_start_id"])
                    if probe_execution_group(identity) != Liveness.DEAD:
                        raise ValueError("Execution children must stop before finalizing")
            elif exit_code is not None:
                raise ValueError("Only final states may record a process exit code")
            connection.execute(
                """UPDATE jobs SET execution_state = ?, error_details = ?, exit_code = ?,
                       finished_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE finished_at END
                   WHERE job_id = ?""",
                (target.value, error.model_dump_json() if error else None,
                 exit_code, target in TERMINAL_STATES, job_id),
            )
            payload = {"domain": "execution", "previous": current.value, "state": target.value}
            if error is not None:
                payload["error"] = error.model_dump(mode="json")
            if exit_code is not None:
                payload["exit_code"] = exit_code
            _append_event(connection, job_id, "state_changed", payload)
            return _record(_required_row(connection, job_id))


def transition_tracking(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
    target: TrackingState, *, error: ErrorDetails | None = None,
) -> JobRecord:
    """Record synchronization independently; an upload error is not training failure."""
    _check_error(error)
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _required_row(connection, job_id)
            current = TrackingState(row["tracking_state"])
            validate_tracking_transition(current, target)
            saved = _record(row)
            if current == target:
                if error is not None and error != saved.tracking_error:
                    raise ValueError("A repeated transition cannot replace its saved error")
                return saved
            if target == TrackingState.FAILED and error is None:
                raise ValueError("Failed synchronization requires error details")
            if target not in {TrackingState.FAILED, TrackingState.RETRYING} and error is not None:
                raise ValueError("This tracking state cannot have an error")
            reason = error or (saved.tracking_error if target == TrackingState.RETRYING else None)
            connection.execute(
                "UPDATE jobs SET tracking_state = ?, tracking_error = ? WHERE job_id = ?",
                (target.value, reason.model_dump_json() if reason else None, job_id),
            )
            payload = {"domain": "tracking", "previous": current.value, "state": target.value}
            if reason is not None:
                payload["error"] = reason.model_dump(mode="json")
            _append_event(connection, job_id, "state_changed", payload)
            return _record(_required_row(connection, job_id))
