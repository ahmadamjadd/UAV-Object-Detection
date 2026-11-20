"""Initialize and migrate the local database without discarding data."""

from contextlib import closing
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
import sqlite3

from foxtrot.config.event import ProgressEvent

from .workspace import initialize_workspace


DATABASE_SCHEMA_VERSION = 6

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (),
    2: (
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY NOT NULL,

            submission_key TEXT NOT NULL UNIQUE
                CHECK (length(trim(submission_key)) > 0),

            execution_state TEXT NOT NULL DEFAULT 'queued'
                CHECK (
                    execution_state IN (
                        'draft', 'validating', 'queued',
                        'preparing', 'training', 'evaluating',
                        'finished', 'failed', 'cancelled',
                        'interrupted'
                    )
                ),

            tracking_state TEXT NOT NULL DEFAULT 'pending'
                CHECK (
                    tracking_state IN (
                        'pending', 'syncing', 'synced',
                        'retrying', 'failed'
                    )
                ),

            last_event_sequence INTEGER NOT NULL DEFAULT 0
                CHECK (
                    typeof(last_event_sequence) = 'integer'
                    AND last_event_sequence >= 0
                ),

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ),
    3: (
        "ALTER TABLE jobs ADD COLUMN submitted_sha256 TEXT",
        "ALTER TABLE jobs ADD COLUMN resolved_sha256 TEXT",
        "ALTER TABLE jobs ADD COLUMN claim_token TEXT",
        "ALTER TABLE jobs ADD COLUMN claimed_at TEXT",
        """
        CREATE UNIQUE INDEX one_active_job ON jobs ((1))
        WHERE execution_state IN ('preparing', 'training', 'evaluating')
        """,
        """
        CREATE TABLE artifacts (
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            name TEXT NOT NULL CHECK (length(trim(name)) > 0),
            kind TEXT NOT NULL CHECK (kind IN ('artifact', 'checkpoint')),
            relative_path TEXT NOT NULL,
            sha256 TEXT NOT NULL
                CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
            size_bytes INTEGER NOT NULL
                CHECK (typeof(size_bytes) = 'integer' AND size_bytes >= 0),
            PRIMARY KEY (job_id, name),
            UNIQUE (job_id, relative_path)
        )
        """,
    ),
    4: (
        "ALTER TABLE jobs ADD COLUMN error_details TEXT",
        "ALTER TABLE jobs ADD COLUMN tracking_error TEXT",
        "ALTER TABLE jobs ADD COLUMN owner_pid INTEGER",
        "ALTER TABLE jobs ADD COLUMN owner_start_id TEXT",
        "ALTER TABLE jobs ADD COLUMN process_pid INTEGER",
        "ALTER TABLE jobs ADD COLUMN process_start_id TEXT",
        "ALTER TABLE jobs ADD COLUMN process_group_id INTEGER",
        "ALTER TABLE jobs ADD COLUMN process_launch_pending INTEGER NOT NULL DEFAULT 0 CHECK (process_launch_pending IN (0, 1))",
        "ALTER TABLE jobs ADD COLUMN recovery_required INTEGER NOT NULL DEFAULT 0 CHECK (recovery_required IN (0, 1))",
        """
        CREATE TABLE job_events (
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            sequence INTEGER NOT NULL
                CHECK (typeof(sequence) = 'integer' AND sequence >= 1),
            event_key TEXT,
            event_json TEXT NOT NULL,
            PRIMARY KEY (job_id, sequence),
            UNIQUE (job_id, event_key)
        )
        """,
    ),
    5: (
        "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1))",
        "ALTER TABLE jobs ADD COLUMN exit_code INTEGER CHECK (exit_code IS NULL OR typeof(exit_code) = 'integer')",
        "ALTER TABLE jobs ADD COLUMN finished_at TEXT",
    ),
    6: (
        # One tracking run per local job. The browser-facing URL is stored
        # separately from any tracking API address, which stays in settings.
        """
        CREATE TABLE tracking_runs (
            job_id TEXT PRIMARY KEY NOT NULL REFERENCES jobs(job_id),
            destination_id TEXT NOT NULL CHECK (length(trim(destination_id)) > 0),
            run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
            run_url TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        # Ordered pending uploads. Rows are acknowledged, never deleted, so a
        # retry resumes after the last confirmed operation.
        """
        CREATE TABLE tracking_outbox (
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            position INTEGER NOT NULL
                CHECK (typeof(position) = 'integer' AND position >= 1),
            operation TEXT NOT NULL CHECK (operation IN (
                'create_run', 'log_params', 'log_metrics', 'log_artifact', 'set_terminated'
            )),
            payload_json TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0
                CHECK (typeof(attempts) = 'integer' AND attempts >= 0),
            acknowledged_at TEXT,
            PRIMARY KEY (job_id, position)
        )
        """,
    ),
}


def _seed_legacy_events(connection: sqlite3.Connection) -> None:
    """Record the imported state, without inventing pre-H4 progress history."""
    rows = connection.execute(
        "SELECT job_id, execution_state, tracking_state, last_event_sequence FROM jobs"
    ).fetchall()
    for job_id, execution, tracking, sequence in rows:
        event = ProgressEvent(
            job_id=job_id, sequence=sequence + 1, timestamp=datetime.now(UTC),
            event_type="state_changed", payload={
                "source": "migration", "execution_state": execution,
                "tracking_state": tracking, "prior_history_available": False,
            },
        )
        connection.execute(
            "INSERT INTO job_events (job_id, sequence, event_key, event_json) VALUES (?, ?, ?, ?)",
            (job_id, event.sequence, "system:migration-v4", event.model_dump_json()),
        )
        connection.execute(
            "UPDATE jobs SET last_event_sequence = ? WHERE job_id = ?",
            (event.sequence, job_id),
        )


def initialize_database(
    workspace_root: str | Path,
    source_checkout: str | Path,
) -> Path:
    """Create or upgrade a database to the supported schema version."""

    root = initialize_workspace(workspace_root, source_checkout)
    database_path = root / "app.sqlite"

    if database_path.is_symlink():
        raise ValueError(
            "Database path must not be a symbolic link"
        )

    # SQLite can create these adjacent files itself. Reject redirected files
    # before opening the database, as well as a redirected database file.
    for suffix in ("-journal", "-wal", "-shm"):
        if database_path.with_name(database_path.name + suffix).is_symlink():
            raise ValueError("Database sidecar must not be a symbolic link")

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")

            version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]

            if version < 0 or version > DATABASE_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported database schema version: {version}"
                )

            if version == 0:
                existing_object = connection.execute(
                    "SELECT name FROM sqlite_master LIMIT 1"
                ).fetchone()

                if existing_object is not None:
                    raise ValueError(
                        "Refusing a nonempty database without "
                        "a schema version"
                    )

            for next_version in range(
                version + 1,
                DATABASE_SCHEMA_VERSION + 1,
            ):
                for statement in _MIGRATIONS[next_version]:
                    connection.execute(statement)

                if next_version == 4:
                    _seed_legacy_events(connection)

                connection.execute(
                    f"PRAGMA user_version = {next_version}"
                )

    return database_path


@contextmanager
def connect_database(workspace_root: str | Path, source_checkout: str | Path):
    """Open a validated workspace database and always close the connection.

    Callers own their transaction using ``with connection`` and, for claims or
    submissions, ``BEGIN IMMEDIATE``. Foreign keys apply to every app write.
    """
    path = initialize_database(workspace_root, source_checkout)
    with closing(sqlite3.connect(path, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
