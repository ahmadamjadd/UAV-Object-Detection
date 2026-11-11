"""Committed progress history with transactional sequence numbers and replay."""

from datetime import UTC, datetime
import os
from pathlib import Path
import sqlite3
from typing import Literal

from foxtrot.config.event import ProgressEvent

from .database import connect_database
from .processes import capture_process_identity


def _require_claim(row: sqlite3.Row, claim_token: str | None) -> None:
    """Fence writes from another claim or a process that does not own this claim."""
    if not claim_token or claim_token != row["claim_token"]:
        raise ValueError("A matching claim token is required")
    owner = capture_process_identity(os.getpid())
    if (row["owner_pid"], row["owner_start_id"]) != (owner.pid, owner.start_id):
        raise ValueError("Only the recorded owner may update this execution")


def _append_event(
    connection: sqlite3.Connection, job_id: str, event_type: str,
    payload: dict, event_key: str | None = None,
) -> ProgressEvent:
    """Append under the caller's write transaction, or replay an identical key."""
    if not connection.in_transaction:
        raise RuntimeError("Event writes require a transaction")
    row = connection.execute(
        "SELECT last_event_sequence FROM jobs WHERE job_id = ?", (job_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown job: {job_id}")
    if event_key is not None and (
        not isinstance(event_key, str) or not event_key.strip() or len(event_key) > 200
    ):
        raise ValueError("Event key must contain 1–200 characters")
    event = ProgressEvent(
        job_id=job_id, sequence=row[0] + 1, timestamp=datetime.now(UTC),
        event_type=event_type, payload=payload,
    )
    # Serialization takes a snapshot of mutable payloads before committing.
    encoded = event.model_dump_json()
    if event_key is not None:
        existing = connection.execute(
            "SELECT event_json FROM job_events WHERE job_id = ? AND event_key = ?",
            (job_id, event_key),
        ).fetchone()
        if existing is not None:
            saved = ProgressEvent.model_validate_json(existing[0])
            if saved.event_type != event.event_type or saved.payload != event.payload:
                raise ValueError("Event key was already used for a different update")
            return saved
    connection.execute(
        "INSERT INTO job_events (job_id, sequence, event_key, event_json) VALUES (?, ?, ?, ?)",
        (job_id, event.sequence, event_key, encoded),
    )
    connection.execute(
        "UPDATE jobs SET last_event_sequence = ? WHERE job_id = ?",
        (event.sequence, job_id),
    )
    return ProgressEvent.model_validate_json(encoded)


def append_job_event(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
    event_type: Literal["progress", "log", "error"], payload: dict,
    *, claim_token: str, event_key: str | None = None,
) -> ProgressEvent:
    """Save an owned execution update before making it available to clients."""
    if event_type not in {"progress", "log", "error"}:
        raise ValueError("State events must be produced by lifecycle operations")
    if event_key is not None and (not isinstance(event_key, str) or event_key.startswith("system:")):
        raise ValueError("Reserved or invalid event key")
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown job: {job_id}")
            _require_claim(row, claim_token)
            if row["execution_state"] not in {"preparing", "training", "evaluating"}:
                raise ValueError("New execution events require an active job")
            return _append_event(connection, job_id, event_type, payload, event_key)


def read_job_events(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
    *, after_sequence: int = 0, limit: int = 1000,
) -> tuple[ProgressEvent, ...]:
    """Read a bounded page after the client's last seen sequence, in order."""
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError("Event cursor must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Event page size must be between 1 and 1000")
    with connect_database(workspace_root, source_checkout) as connection:
        if connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is None:
            raise KeyError(f"Unknown job: {job_id}")
        rows = connection.execute(
            "SELECT event_json FROM job_events WHERE job_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
            (job_id, after_sequence, limit),
        ).fetchall()
        return tuple(ProgressEvent.model_validate_json(row[0]) for row in rows)
