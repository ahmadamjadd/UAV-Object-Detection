"""Linux process birth identities and conservative liveness observations.

No process is launched or signalled here. Unsupported or inaccessible evidence
is unknown, never proof that execution stopped.
"""

from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path


class Liveness(StrEnum):
    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_id: str


@dataclass(frozen=True)
class _Snapshot:
    state: str
    parent: int
    group: int
    session: int
    start: str


def _context() -> tuple[str, str]:
    return (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        os.readlink("/proc/self/ns/pid"),
    )


def _snapshot(pid: int) -> _Snapshot:
    # The command name in parentheses can itself contain spaces or ')'.
    value = Path(f"/proc/{pid}/stat").read_text()
    fields = value[value.rfind(")") + 2:].split()
    return _Snapshot(fields[0], int(fields[1]), int(fields[2]), int(fields[3]), fields[19])


def capture_process_identity(pid: int) -> ProcessIdentity:
    """Identify a live process in this boot and PID namespace, including birth time."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("Process ID must be a positive integer")
    boot, namespace = _context()
    if os.readlink(f"/proc/{pid}/ns/pid") != namespace:
        raise ValueError("Process is outside the current PID namespace")
    snapshot = _snapshot(pid)
    if snapshot.state in {"Z", "X", "x"}:
        raise ProcessLookupError("Process has already stopped")
    return ProcessIdentity(pid, "|".join((boot, namespace, snapshot.start)))


def _identity_context(identity: ProcessIdentity) -> Liveness | None:
    boot, namespace, _ = identity.start_id.split("|")
    current_boot, current_namespace = _context()
    if boot != current_boot:
        return Liveness.DEAD
    if namespace != current_namespace:
        return Liveness.UNKNOWN
    return None


def probe_process(identity: ProcessIdentity) -> Liveness:
    """A recycled PID is dead with respect to the recorded process identity."""
    try:
        mismatch = _identity_context(identity)
        if mismatch is not None:
            return mismatch
        try:
            snapshot = _snapshot(identity.pid)
        except FileNotFoundError:
            return Liveness.DEAD
        if snapshot.state in {"Z", "X", "x"}:
            return Liveness.DEAD
        return (Liveness.ALIVE if snapshot.start == identity.start_id.split("|")[2]
                else Liveness.DEAD)
    except (OSError, ValueError, IndexError, AttributeError, TypeError):
        return Liveness.UNKNOWN


def capture_execution_process(pid: int) -> ProcessIdentity:
    """Accept only our direct child in its own process group and session."""
    identity = capture_process_identity(pid)
    snapshot = _snapshot(pid)
    if snapshot.parent != os.getpid() or snapshot.group != pid or snapshot.session != pid:
        raise ValueError("Execution must be an owned child with a dedicated session")
    if probe_process(identity) != Liveness.ALIVE:
        raise ProcessLookupError("Execution exited while recording ownership")
    return identity


def probe_execution_group(identity: ProcessIdentity) -> Liveness:
    """Look for surviving children even after the original group leader exits.

    H5 must keep execution children in this dedicated session. We inspect the
    entire session, including a child's separate process group, without signals.
    """
    try:
        mismatch = _identity_context(identity)
        if mismatch is not None:
            return mismatch
        uncertain = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                snapshot = _snapshot(int(entry.name))
            except FileNotFoundError:
                continue
            except (OSError, ValueError, IndexError):
                uncertain = True
                continue
            if snapshot.session == identity.pid and snapshot.state not in {"Z", "X", "x"}:
                return Liveness.ALIVE
        return Liveness.UNKNOWN if uncertain else Liveness.DEAD
    except (OSError, ValueError, IndexError, AttributeError, TypeError):
        return Liveness.UNKNOWN
