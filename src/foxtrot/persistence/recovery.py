"""Record execution ownership and reconcile stale jobs without launching work."""

from pathlib import Path

from foxtrot.config.lifecycle import ErrorDetails

from .database import connect_database
from .events import _append_event, _require_claim
from .jobs import JobRecord, _record, _required_row
from .processes import (
    Liveness, ProcessIdentity, capture_execution_process,
    probe_execution_group, probe_process,
)


def begin_process_launch(
    workspace_root: str | Path, source_checkout: str | Path,
    job_id: str, claim_token: str,
) -> JobRecord:
    """Persist launch intent BEFORE H5 starts a child, closing the crash gap.

    If the owner dies before recording the child's identity, recovery cannot
    prove whether a child started, so it keeps the execution slot blocked.
    """
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _required_row(connection, job_id)
            _require_claim(row, claim_token)
            if row["execution_state"] not in {"preparing", "training", "evaluating"}:
                raise ValueError("Process launch requires an active job")
            if row["process_pid"] is not None:
                raise ValueError("An execution process is already recorded")
            if row["process_launch_pending"]:
                return _record(row)
            connection.execute("UPDATE jobs SET process_launch_pending = 1 WHERE job_id = ?", (job_id,))
            _append_event(connection, job_id, "log", {"action": "process_launch_pending"})
            return _record(_required_row(connection, job_id))


def attach_execution_process(
    workspace_root: str | Path, source_checkout: str | Path,
    job_id: str, claim_token: str, pid: int,
) -> JobRecord:
    """Record an owned dedicated-session child; this function does not spawn it."""
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _required_row(connection, job_id)
            _require_claim(row, claim_token)
            if row["execution_state"] not in {"preparing", "training", "evaluating"}:
                raise ValueError("Attaching an execution requires an active job")
            identity = capture_execution_process(pid)
            if row["process_pid"] is not None:
                if (row["process_pid"], row["process_start_id"]) == (pid, identity.start_id):
                    return _record(row)
                raise ValueError("Cannot replace an execution process")
            if not row["process_launch_pending"]:
                raise ValueError("Record launch intent before starting the child")
            connection.execute(
                """UPDATE jobs SET process_pid = ?, process_start_id = ?,
                       process_group_id = ?, process_launch_pending = 0 WHERE job_id = ?""",
                (pid, identity.start_id, pid, job_id),
            )
            _append_event(connection, job_id, "log", {"action": "execution_attached", "pid": pid})
            return _record(_required_row(connection, job_id))


def _evidence(row) -> tuple[Liveness, Liveness]:
    owner = Liveness.UNKNOWN
    if row["owner_pid"] is not None and row["owner_start_id"] is not None:
        owner = probe_process(ProcessIdentity(row["owner_pid"], row["owner_start_id"]))
    execution = Liveness.DEAD
    if row["process_pid"] is not None and row["process_start_id"] is not None:
        execution = probe_execution_group(ProcessIdentity(row["process_pid"], row["process_start_id"]))
    elif row["process_pid"] is not None or row["process_start_id"] is not None:
        execution = Liveness.UNKNOWN
    return owner, execution


def settle_failed_launch(
    workspace_root: str | Path, source_checkout: str | Path,
    job_id: str, claim_token: str, *, stopped_child: ProcessIdentity | None = None,
) -> None:
    """Clear intent only for the still-live owner handling its failed spawn.

    With no child, the caller must have observed Popen raising before returning
    a process. Otherwise it must first stop its known child/session and supply
    that identity. This is not a restart recovery or force-clear operation.
    """
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _required_row(connection, job_id)
            _require_claim(row, claim_token)
            if row["process_pid"] is not None:
                raise ValueError("An attached process is not an uncommitted launch")
            if stopped_child is not None and probe_execution_group(stopped_child) != Liveness.DEAD:
                raise ValueError("Failed-launch children must be proven stopped")
            if row["process_launch_pending"]:
                connection.execute("UPDATE jobs SET process_launch_pending = 0 WHERE job_id = ?", (job_id,))
                _append_event(connection, job_id, "log", {"action": "failed_launch_settled"})


def hold_owned_execution(
    workspace_root: str | Path, source_checkout: str | Path,
    job_id: str, claim_token: str,
) -> None:
    """Record bounded-cleanup uncertainty without pretending execution stopped."""
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _required_row(connection, job_id)
            _require_claim(row, claim_token)
            if row["execution_state"] not in {"preparing", "training", "evaluating"}:
                return
            reason = ErrorDetails(code="cleanup_uncertain", message="Execution cleanup could not be verified; new claims are blocked.")
            connection.execute(
                "UPDATE jobs SET execution_state = 'interrupted', recovery_required = 1, error_details = ? WHERE job_id = ?",
                (reason.model_dump_json(), job_id),
            )
            _append_event(connection, job_id, "state_changed", {
                "domain": "execution", "previous": row["execution_state"], "state": "interrupted",
                "error": reason.model_dump(), "recovery_required": True,
            })


def reconcile_jobs(
    workspace_root: str | Path, source_checkout: str | Path,
) -> tuple[JobRecord, ...]:
    """Interrupt unowned active jobs; never resume, requeue, launch, or signal.

    A matching live owner remains responsible for supervising normal child
    exit. Unknown ownership, an ambiguous launch, or surviving children keep a
    recovery hold. H5 must call this before claiming work after startup.
    """
    changed = []
    with connect_database(workspace_root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT * FROM jobs
                   WHERE execution_state IN ('preparing', 'training', 'evaluating')
                      OR (execution_state = 'interrupted' AND recovery_required = 1)
                   ORDER BY created_at, rowid""",
            ).fetchall()
            for row in rows:
                owner, execution = _evidence(row)
                if owner == Liveness.ALIVE:
                    continue
                hold = (owner != Liveness.DEAD or execution != Liveness.DEAD
                        or bool(row["process_launch_pending"]))
                if row["execution_state"] == "interrupted":
                    if not hold:
                        connection.execute("UPDATE jobs SET recovery_required = 0 WHERE job_id = ?", (row["job_id"],))
                        _append_event(connection, row["job_id"], "log", {
                            "action": "recovery_hold_cleared", "owner": owner.value,
                            "execution_group": execution.value,
                        })
                        changed.append(_record(_required_row(connection, row["job_id"])))
                    continue
                reason = ErrorDetails(
                    code="ownership_uncertain" if hold else "owner_stopped",
                    message=("Execution ownership is uncertain or children may still be running; new claims are blocked."
                             if hold else "The recorded execution owner stopped before saving a final outcome."),
                )
                connection.execute(
                    "UPDATE jobs SET execution_state = 'interrupted', error_details = ?, recovery_required = ? WHERE job_id = ?",
                    (reason.model_dump_json(), int(hold), row["job_id"]),
                )
                _append_event(connection, row["job_id"], "state_changed", {
                    "domain": "execution", "previous": row["execution_state"],
                    "state": "interrupted", "error": reason.model_dump(mode="json"),
                    "owner": owner.value, "execution_group": execution.value,
                    "recovery_required": hold,
                })
                changed.append(_record(_required_row(connection, row["job_id"])))
    return tuple(changed)
