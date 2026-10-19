"""Exact immutable artifact references within registered job directories."""

from dataclasses import dataclass
from pathlib import Path

from .database import connect_database
from .files import JOB_MARKER, fingerprint, job_file, job_path
from .jobs import _required_row
from .workspace import initialize_workspace


@dataclass(frozen=True)
class ArtifactReference:
    job_id: str
    name: str
    kind: str
    relative_path: str
    sha256: str
    size_bytes: int


def record_job_artifact(
    workspace_root: str | Path,
    source_checkout: str | Path,
    job_id: str,
    name: str,
    relative_path: str,
    *,
    kind: str = "artifact",
) -> ArtifactReference:
    """Register a completed file, never locating it by scanning output trees.

    Writers must finish before registration. A changed registered artifact is
    an error; a different artifact needs its own name and file path.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Artifact name must be a nonblank string")
    if not isinstance(kind, str) or kind not in {"artifact", "checkpoint"}:
        raise ValueError("Unsupported artifact kind")
    root = initialize_workspace(workspace_root, source_checkout)
    with connect_database(root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _required_row(connection, job_id)
            directory = job_path(root, job_id)
            path = job_file(directory, relative_path)
            if relative_path in {"submitted.json", "resolved.json", "provenance.json", JOB_MARKER}:
                raise ValueError("Configuration and ownership files are application-managed")
            digest, size = fingerprint(path)
            reference = ArtifactReference(job_id, name, kind, relative_path, digest, size)
            existing = connection.execute(
                "SELECT * FROM artifacts WHERE job_id = ? AND name = ?", (job_id, name),
            ).fetchone()
            if existing is not None:
                if ArtifactReference(**dict(existing)) != reference:
                    raise ValueError("Registered artifact identity cannot change")
                return reference
            connection.execute(
                """INSERT INTO artifacts (job_id, name, kind, relative_path, sha256, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (job_id, name, kind, relative_path, digest, size),
            )
    return reference


def get_job_artifact(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str, name: str,
) -> ArtifactReference:
    """Retrieve one exact recorded reference and verify its current contents."""
    root = initialize_workspace(workspace_root, source_checkout)
    with connect_database(root, source_checkout) as connection:
        _required_row(connection, job_id)
        row = connection.execute(
            "SELECT * FROM artifacts WHERE job_id = ? AND name = ?", (job_id, name),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown artifact: {name}")
        reference = ArtifactReference(**dict(row))
    path = job_file(job_path(root, job_id), reference.relative_path)
    if fingerprint(path) != (reference.sha256, reference.size_bytes):
        raise ValueError("Registered artifact content hash mismatch")
    return reference
