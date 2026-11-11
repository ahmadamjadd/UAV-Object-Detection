"""Signal only verified members of the execution's dedicated Linux session."""

import os
from pathlib import Path
import signal
import time

from foxtrot.persistence.processes import (
    Liveness, ProcessIdentity, _context, _snapshot, capture_process_identity,
    probe_execution_group,
)


class OwnershipUncertain(RuntimeError):
    pass


def session_members(root: ProcessIdentity) -> tuple[ProcessIdentity, ...]:
    """Include loader children whose original leader or process group exited."""
    boot, namespace, start = root.start_id.split("|")
    if (boot, namespace) != _context():
        raise OwnershipUncertain("Execution boot or namespace changed")
    try:
        current = capture_process_identity(root.pid)
    except (FileNotFoundError, ProcessLookupError):
        current = None
    if current is not None and current != root:
        # A new leader with a recycled PID must never receive our signals.
        raise OwnershipUncertain("Execution PID now belongs to a different process")
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            snapshot = _snapshot(int(entry.name))
            if snapshot.session == root.pid and snapshot.state not in {"Z", "X", "x"}:
                identity = capture_process_identity(int(entry.name))
                if int(identity.start_id.split("|")[2]) < int(start):
                    raise OwnershipUncertain("Session member predates the owned execution")
                members.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, ValueError, IndexError) as error:
            raise OwnershipUncertain("Unable to verify execution session") from error
    return tuple(members)


def signal_session(root: ProcessIdentity, number: int) -> None:
    """pidfds pin each target against PID reuse between inspection and signal."""
    for member in session_members(root):
        try:
            descriptor = os.pidfd_open(member.pid)
            try:
                if capture_process_identity(member.pid) != member:
                    raise OwnershipUncertain("Session member changed identity")
                if _snapshot(member.pid).session != root.pid:
                    raise OwnershipUncertain("Execution child left its session")
                signal.pidfd_send_signal(descriptor, number)
            finally:
                os.close(descriptor)
        except (FileNotFoundError, ProcessLookupError):
            continue


def stop_session(root: ProcessIdentity, *, terminate_seconds: float, kill_seconds: float) -> bool:
    """Emergency bounded TERM/KILL cleanup; normal cancellation first uses IPC."""
    for number, timeout in ((signal.SIGTERM, terminate_seconds), (signal.SIGKILL, kill_seconds)):
        deadline = time.monotonic() + timeout
        while probe_execution_group(root) != Liveness.DEAD:
            signal_session(root, number)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        if probe_execution_group(root) == Liveness.DEAD:
            return True
    return False
