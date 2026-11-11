"""Durable cancellation requests, consumed by the process owner."""

from pathlib import Path

from foxtrot.config.lifecycle import ACTIVE_STATES, TERMINAL_STATES, ExecutionState

from .database import connect_database
from .events import _append_event
from .jobs import JobRecord, _record, _required_row


def request_cancel(workspace_root: str | Path, source_checkout: str | Path, job_id: str) -> JobRecord:
    """Cancel unstarted work immediately, or persist an active owner's command.

    Requesting cancellation never sends a signal or claims ownership. The worker
    records cancelled only after its execution session has actually stopped.
    """
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _required_row(connection, job_id)
            state = ExecutionState(row["execution_state"])
            if state in TERMINAL_STATES or row["cancel_requested"]:
                return _record(row)
            if state in ACTIVE_STATES:
                connection.execute("UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?", (job_id,))
                _append_event(connection, job_id, "log", {"action": "cancel_requested"})
            else:
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, execution_state = 'cancelled', finished_at = CURRENT_TIMESTAMP WHERE job_id = ?",
                    (job_id,),
                )
                _append_event(connection, job_id, "state_changed", {
                    "domain": "execution", "previous": state.value, "state": "cancelled",
                })
            return _record(_required_row(connection, job_id))
