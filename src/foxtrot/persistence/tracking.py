"""Local-to-tracking run associations and the durable upload outbox.

Results are saved locally first. Uploading is a separate, retryable step, so a
tracking outage never changes a job's execution outcome.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from foxtrot.config.lifecycle import TERMINAL_STATES, ExecutionState
from foxtrot.config.tracking import TrackingOperation

from .database import connect_database
from .events import _append_event
from .jobs import _required_row


@dataclass(frozen=True)
class OutboxEntry:
    job_id: str
    position: int
    operation: TrackingOperation
    attempts: int
    acknowledged_at: str | None


@dataclass(frozen=True)
class RunAssociation:
    job_id: str
    destination_id: str
    run_id: str
    run_url: str | None


def _entry(row) -> OutboxEntry:
    operation = TrackingOperation(operation=row["operation"], payload=json.loads(row["payload_json"]))
    return OutboxEntry(row["job_id"], row["position"], operation, row["attempts"], row["acknowledged_at"])


def _check_plan(operations: Sequence[TrackingOperation]) -> None:
    kinds = [item.operation for item in operations]
    if not operations or not all(isinstance(item, TrackingOperation) for item in operations):
        raise ValueError("A tracking plan needs TrackingOperation values")
    if kinds[0] != "create_run" or kinds[-1] != "set_terminated":
        raise ValueError("A tracking plan starts with create_run and ends with set_terminated")
    if kinds.count("create_run") != 1 or kinds.count("set_terminated") != 1:
        raise ValueError("A tracking plan creates and terminates exactly one run")


def enqueue_tracking(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
    operations: Sequence[TrackingOperation],
) -> tuple[OutboxEntry, ...]:
    """Save a finished job's upload plan once; an identical retry is a no-op."""
    _check_plan(operations)
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _required_row(connection, job_id)
            if ExecutionState(row["execution_state"]) not in TERMINAL_STATES:
                raise ValueError("Only a job with a final outcome can be queued for upload")
            existing = connection.execute(
                "SELECT * FROM tracking_outbox WHERE job_id = ? ORDER BY position", (job_id,),
            ).fetchall()
            if existing:
                if [_entry(item).operation for item in existing] != list(operations):
                    raise ValueError("A different tracking plan is already queued for this job")
                return tuple(_entry(item) for item in existing)
            for position, operation in enumerate(operations, start=1):
                connection.execute(
                    "INSERT INTO tracking_outbox (job_id, position, operation, payload_json) VALUES (?, ?, ?, ?)",
                    (job_id, position, operation.operation, json.dumps(operation.payload, sort_keys=True)),
                )
            _append_event(connection, job_id, "log", {
                "action": "tracking_queued", "operations": len(operations),
            }, "system:tracking-queued")
            return tuple(_entry(item) for item in connection.execute(
                "SELECT * FROM tracking_outbox WHERE job_id = ? ORDER BY position", (job_id,),
            ).fetchall())


def list_tracking(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str, *, pending_only: bool = False,
) -> tuple[OutboxEntry, ...]:
    with connect_database(workspace_root, source_checkout) as connection:
        _required_row(connection, job_id)
        condition = "AND acknowledged_at IS NULL" if pending_only else ""
        return tuple(_entry(row) for row in connection.execute(
            f"SELECT * FROM tracking_outbox WHERE job_id = ? {condition} ORDER BY position", (job_id,),
        ).fetchall())


def record_tracking_attempt(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str, position: int,
) -> None:
    """Count an attempt BEFORE calling the service, so ambiguity is visible."""
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            changed = connection.execute(
                "UPDATE tracking_outbox SET attempts = attempts + 1 WHERE job_id = ? AND position = ? AND acknowledged_at IS NULL",
                (job_id, position),
            ).rowcount
            if changed != 1:
                raise ValueError("Unknown or already acknowledged tracking operation")


def acknowledge_tracking(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str, position: int,
) -> None:
    """Mark one operation confirmed; operations must be acknowledged in order."""
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            earlier = connection.execute(
                "SELECT 1 FROM tracking_outbox WHERE job_id = ? AND position < ? AND acknowledged_at IS NULL",
                (job_id, position),
            ).fetchone()
            if earlier is not None:
                raise ValueError("Earlier tracking operations are still pending")
            connection.execute(
                "UPDATE tracking_outbox SET acknowledged_at = COALESCE(acknowledged_at, CURRENT_TIMESTAMP) WHERE job_id = ? AND position = ?",
                (job_id, position),
            )


def record_run_association(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str, *,
    destination_id: str, run_id: str, run_url: str | None,
) -> RunAssociation:
    """Save which tracking run belongs to this job; it can never be replaced."""
    association = RunAssociation(job_id, destination_id, run_id, run_url)
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _required_row(connection, job_id)
            row = connection.execute("SELECT * FROM tracking_runs WHERE job_id = ?", (job_id,)).fetchone()
            if row is not None:
                saved = RunAssociation(row["job_id"], row["destination_id"], row["run_id"], row["run_url"])
                if saved != association:
                    raise ValueError("This job is already associated with a different tracking run")
                return saved
            connection.execute(
                "INSERT INTO tracking_runs (job_id, destination_id, run_id, run_url) VALUES (?, ?, ?, ?)",
                (job_id, destination_id, run_id, run_url),
            )
            _append_event(connection, job_id, "log", {
                "action": "tracking_run_associated", "run_id": run_id, "run_url": run_url,
            }, "system:tracking-run")
    return association


def get_run_association(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
) -> RunAssociation | None:
    with connect_database(workspace_root, source_checkout) as connection:
        _required_row(connection, job_id)
        row = connection.execute("SELECT * FROM tracking_runs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return RunAssociation(row["job_id"], row["destination_id"], row["run_id"], row["run_url"])
