"""Managed job paths and durable, exclusive writes; no training imports."""

import hashlib
import os
from pathlib import Path, PurePosixPath
from uuid import UUID


JOB_MARKER = ".foxtrot-job"
JOB_FORMAT = "foxtrot-job-v1\n"


def job_path(root: Path, job_id: str) -> Path:
    """Resolve an existing job folder without following a job symlink."""
    if not isinstance(job_id, str) or UUID(job_id).hex != job_id:
        raise ValueError("Job ID must use lowercase UUID hex")
    jobs = root / "jobs"
    path = jobs / job_id
    if jobs.is_symlink() or not jobs.is_dir():
        raise ValueError("Workspace jobs folder is missing or unsafe")
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Stored job folder is missing or unsafe")
    return path


def job_file(directory: Path, relative_path: str) -> Path:
    """Require an existing regular file, with no symlinks in its path."""
    if not isinstance(relative_path, str):
        raise ValueError("Artifact path must be a relative string")
    relative = PurePosixPath(relative_path)
    if (
        not relative_path.strip()
        or relative == PurePosixPath(".")
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in relative_path
        or ":" in relative_path
        or "\x00" in relative_path
        or relative.as_posix() != relative_path
    ):
        raise ValueError("Use a canonical relative file path without '..'")
    path = directory
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Job files must not traverse symbolic links")
    if not path.is_file():
        raise ValueError(f"Stored job file is missing or unsafe: {relative_path}")
    return path


def sync_directory(directory: Path) -> None:
    """Flush directory entries on the verified Linux target."""
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_new_text(path: Path, content: str) -> None:
    """Create a file exclusively and flush its contents before returning."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def fingerprint(path: Path) -> tuple[str, int]:
    """Hash one completed file, detecting ordinary concurrent modifications."""
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    current = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if before_identity != after_identity or after_identity != current_identity:
        raise ValueError("Job file changed while its fingerprint was computed")
    return digest, after.st_size
