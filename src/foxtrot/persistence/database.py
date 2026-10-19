"""Initialize and migrate the local database without discarding data."""

from contextlib import closing
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from .workspace import initialize_workspace


DATABASE_SCHEMA_VERSION = 3

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
}


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
