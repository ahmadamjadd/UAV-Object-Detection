"""Isolated job files, immutable manifests, submissions, and queue claims."""

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4

from foxtrot.config.job import JobSpec
from foxtrot.config.snapshot import JobConfigurations, resolve_job

from .database import connect_database
from .files import (
    JOB_FORMAT, JOB_MARKER, job_file, job_path, sync_directory, write_new_text,
)
from .workspace import initialize_workspace


@dataclass(frozen=True)
class JobRecord:
    """A local database snapshot; tracking and execution remain independent."""

    job_id: str
    submission_key: str
    execution_state: str
    tracking_state: str
    last_event_sequence: int
    created_at: str
    submitted_sha256: str | None
    resolved_sha256: str | None
    claim_token: str | None
    claimed_at: str | None


@dataclass(frozen=True)
class JobClaim:
    """One committed reservation, not evidence of a live training process."""

    job_id: str
    claim_token: str
    directory: Path
    configurations: JobConfigurations


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(**{field.name: row[field.name] for field in fields(JobRecord)})


def _required_row(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM jobs WHERE job_id = ?", (job_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown job: {job_id}")
    return row


def _submitted_matches(submitted: dict, resolved: dict) -> bool:
    """Compare explicit choices while retaining the original saved defaults."""
    for key, value in submitted.items():
        if key not in resolved:
            return False
        if isinstance(value, dict):
            if not isinstance(resolved[key], dict) or not _submitted_matches(
                value, resolved[key],
            ):
                return False
        elif value != resolved[key]:
            return False
    return True


def _read_configurations(
    connection: sqlite3.Connection, root: Path, row: sqlite3.Row,
) -> tuple[Path, JobConfigurations]:
    """Validate files and their saved hashes under a caller's transaction.

    Version-2 jobs have no stored hashes. Their first successful read adopts
    their validated manifests; all later reads require those same bytes.
    """
    directory = job_path(root, row["job_id"])
    contents = []
    hashes = []
    for filename in ("submitted.json", "resolved.json"):
        try:
            content = job_file(directory, filename).read_bytes()
        except ValueError as error:
            raise ValueError(f"Stored job configuration is missing or unsafe: {filename}") from error
        JobSpec.model_validate_json(content)
        contents.append(content.decode("utf-8"))
        hashes.append(hashlib.sha256(content).hexdigest())
    if not _submitted_matches(json.loads(contents[0]), json.loads(contents[1])):
        raise ValueError("Submitted and resolved configurations disagree")
    expected = (row["submitted_sha256"], row["resolved_sha256"])
    if expected == (None, None):
        connection.execute(
            "UPDATE jobs SET submitted_sha256 = ?, resolved_sha256 = ? WHERE job_id = ?",
            (*hashes, row["job_id"]),
        )
    elif expected != tuple(hashes):
        raise ValueError("Stored job configuration hash mismatch")
    return directory, JobConfigurations(contents[0], contents[1])


def create_job_directory(
    workspace_root: str | Path,
    source_checkout: str | Path,
) -> Path:
    """Create a new job directory and return its absolute path."""

    root = initialize_workspace(workspace_root, source_checkout)

    job_id = uuid4().hex
    job_directory = root / "jobs" / job_id

    job_directory.mkdir(exist_ok=False)
    write_new_text(job_directory / JOB_MARKER, JOB_FORMAT)
    sync_directory(job_directory)
    sync_directory(job_directory.parent)

    return job_directory


def create_configured_job(
    workspace_root: str | Path,
    source_checkout: str | Path,
    request: dict[str, object],
) -> Path:
    """Validate a request and save its settings in a new job folder."""

    snapshots = resolve_job(request)

    job_directory = create_job_directory(
        workspace_root,
        source_checkout,
    )

    for filename, content in (
        ("submitted.json", snapshots.submitted_json),
        ("resolved.json", snapshots.resolved_json),
    ):
        write_new_text(job_directory / filename, content)

    # File-level provenance is available before registration or execution.
    # Training/runtime/source provenance is added by the real pipeline later.
    provenance = {
        "schema_version": 1,
        "job_id": job_directory.name,
        "submitted_sha256": hashlib.sha256(snapshots.submitted_json.encode("utf-8")).hexdigest(),
        "resolved_sha256": hashlib.sha256(snapshots.resolved_json.encode("utf-8")).hexdigest(),
    }
    write_new_text(
        job_directory / "provenance.json", json.dumps(provenance, indent=2),
    )
    sync_directory(job_directory)

    return job_directory


def create_recorded_job(
    workspace_root: str | Path,
    source_checkout: str | Path,
    request: dict[str, object],
    submission_key: str,
) -> Path:
    """Create a recorded job or return a matching previous submission."""

    if (
        not isinstance(submission_key, str)
        or not submission_key.strip()
    ):
        raise ValueError(
            "Submission key must be a nonblank string"
        )

    snapshots = resolve_job(request)

    root = initialize_workspace(workspace_root, source_checkout)
    with connect_database(root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")

            existing = connection.execute(
                "SELECT * FROM jobs WHERE submission_key = ?",
                (submission_key,),
            ).fetchone()

            if existing is not None:
                directory, saved = _read_configurations(connection, root, existing)
                if json.loads(saved.submitted_json) != json.loads(snapshots.submitted_json):
                    raise ValueError("Submission key was already used with different settings")
                return directory

            job_directory = create_configured_job(
                root,
                source_checkout,
                request,
            )

            connection.execute(
                """
                INSERT INTO jobs (job_id, submission_key)
                VALUES (?, ?)
                """,
                (job_directory.name, submission_key),
            )
            _, saved = _read_configurations(
                connection, root, _required_row(connection, job_directory.name),
            )
            if json.loads(saved.submitted_json) != json.loads(snapshots.submitted_json):
                raise ValueError("Saved configuration differs from the submitted request")

    return job_directory


def get_job_record(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
) -> JobRecord:
    """Read persisted metadata using a fresh connection."""
    with connect_database(workspace_root, source_checkout) as connection:
        return _record(_required_row(connection, job_id))


def load_job_configurations(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
) -> JobConfigurations:
    """Read validated, fingerprinted configurations without resolving defaults anew."""
    root = initialize_workspace(workspace_root, source_checkout)
    with connect_database(root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _, configurations = _read_configurations(
                connection, root, _required_row(connection, job_id),
            )
            return configurations


def claim_next_job(
    workspace_root: str | Path, source_checkout: str | Path,
) -> JobClaim | None:
    """Reserve the oldest queued job, allowing one active job per workspace.

    A stale active job blocks further claims until H4 reconciles it. No process
    is started here, and closing/reopening the database never resets a claim.
    """
    root = initialize_workspace(workspace_root, source_checkout)
    with connect_database(root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT 1 FROM jobs WHERE execution_state IN ('preparing', 'training', 'evaluating') LIMIT 1",
            ).fetchone()
            if active is not None:
                return None
            row = connection.execute(
                "SELECT * FROM jobs WHERE execution_state = 'queued' ORDER BY created_at, rowid LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            directory, configurations = _read_configurations(connection, root, row)
            token = uuid4().hex
            changed = connection.execute(
                """UPDATE jobs SET execution_state = 'preparing', claim_token = ?,
                          claimed_at = CURRENT_TIMESTAMP
                   WHERE job_id = ? AND execution_state = 'queued'""",
                (token, row["job_id"]),
            ).rowcount
            if changed != 1:
                raise RuntimeError("Queued job changed while being claimed")
            claim = JobClaim(row["job_id"], token, directory, configurations)
    return claim


def cleanup_unregistered_job(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str,
) -> None:
    """Explicitly remove an abandoned, owned folder with no database record.

    Never removes a registered job, including finished or unsynchronized jobs.
    Caller must stop any standalone preparation writer first. Application
    submissions hold the same database lock throughout preparation.
    """
    root = initialize_workspace(workspace_root, source_checkout)
    with connect_database(root, source_checkout) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,),
            ).fetchone() is not None:
                raise ValueError("Registered jobs must not be removed by temporary cleanup")
            directory = job_path(root, job_id)
            marker = job_file(directory, JOB_MARKER)
            if marker.read_text(encoding="utf-8") != JOB_FORMAT:
                raise ValueError("Unrecognized job ownership marker")
            if not shutil.rmtree.avoids_symlink_attacks:
                raise RuntimeError("Safe job cleanup is not supported on this platform")
            shutil.rmtree(directory)
            sync_directory(directory.parent)
