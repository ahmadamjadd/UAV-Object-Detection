"""Kernel-held worker leases; stale files never prove that a worker is alive."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import tempfile


class WorkerAlreadyRunning(RuntimeError):
    pass


def default_runtime_directory() -> Path:
    """One local worker per OS user, including across different workspaces."""
    return Path(tempfile.gettempdir()) / f"foxtrot-worker-{os.getuid()}"


@contextmanager
def worker_lock(path: Path):
    """Hold a non-inherited flock until this context/process exits.

    Keep the inode after release: unlinking a lock file would let two processes
    lock different inodes at the same pathname. No PID-based lock stealing.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Worker lock directory must be private and owned by this user")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077:
            raise ValueError("Unsafe worker lock file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkerAlreadyRunning("A Foxtrot worker already holds this lease") from error
        yield
    finally:
        os.close(descriptor)
