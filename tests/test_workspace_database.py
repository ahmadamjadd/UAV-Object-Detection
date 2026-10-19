"""Database initialization checks use temporary workspaces."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from foxtrot.persistence.database import (
    DATABASE_SCHEMA_VERSION,
    initialize_database,
)
from foxtrot.persistence.workspace import initialize_workspace


class DatabaseInitializationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(
            prefix="foxtrot-database-"
        )
        self.addCleanup(self.fixture.cleanup)

        self.root = Path(self.fixture.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()

        self.workspace = self.root / "workspace"

    def test_creates_versioned_database(self):
        database = initialize_database(
            self.workspace,
            self.checkout,
        )

        self.assertEqual(database, self.workspace / "app.sqlite")
        self.assertTrue(database.is_file())

        with closing(sqlite3.connect(database)) as connection:
            version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]

        self.assertEqual(version, DATABASE_SCHEMA_VERSION)

    def test_reopening_preserves_existing_data(self):
        database = initialize_database(
            self.workspace,
            self.checkout,
        )

        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE fixture (value TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO fixture (value) VALUES (?)",
                    ("keep this",),
                )

        reopened = initialize_database(
            self.workspace,
            self.checkout,
        )

        self.assertEqual(reopened, database)

        with closing(sqlite3.connect(reopened)) as connection:
            row = connection.execute(
                "SELECT value FROM fixture"
            ).fetchone()

        self.assertEqual(row, ("keep this",))

    def test_rejects_unsupported_version_without_changes(self):
        database = initialize_database(
            self.workspace,
            self.checkout,
        )

        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute("PRAGMA user_version = 99")

        before = database.read_bytes()

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported database schema version",
        ):
            initialize_database(
                self.workspace,
                self.checkout,
            )

        self.assertEqual(database.read_bytes(), before)

    def test_rejects_nonempty_unversioned_database(self):
        initialize_workspace(self.workspace, self.checkout)
        database = self.workspace / "app.sqlite"

        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE fixture (value TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO fixture (value) VALUES (?)",
                    ("preserve this",),
                )

        before = database.read_bytes()

        with self.assertRaisesRegex(
            ValueError,
            "nonempty database",
        ):
            initialize_database(
                self.workspace,
                self.checkout,
            )

        self.assertEqual(database.read_bytes(), before)

    def test_rejects_database_symlink(self):
        initialize_workspace(self.workspace, self.checkout)

        external = self.root / "external.sqlite"
        external.write_bytes(b"preserve this external file")

        database = self.workspace / "app.sqlite"
        database.symlink_to(external)

        with self.assertRaisesRegex(
            ValueError,
            "symbolic link",
        ):
            initialize_database(
                self.workspace,
                self.checkout,
            )

        self.assertEqual(
            external.read_bytes(),
            b"preserve this external file",
        )
    def test_upgrades_version_one_and_preserves_data(self):
        initialize_workspace(self.workspace, self.checkout)
        database = self.workspace / "app.sqlite"

        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE fixture (value TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO fixture (value) VALUES (?)",
                    ("keep this",),
                )
                connection.execute("PRAGMA user_version = 1")

        initialize_database(self.workspace, self.checkout)

        with closing(sqlite3.connect(database)) as connection:
            version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            saved = connection.execute(
                "SELECT value FROM fixture"
            ).fetchone()
            job_count = connection.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]

        self.assertEqual(version, DATABASE_SCHEMA_VERSION)
        self.assertEqual(saved, ("keep this",))
        self.assertEqual(job_count, 0)

    def test_job_record_survives_reopening_with_defaults(self):
        database = initialize_database(
            self.workspace, self.checkout
        )

        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO jobs (job_id, submission_key)
                    VALUES (?, ?)
                    """,
                    ("job-first", "submission-first"),
                )

        initialize_database(self.workspace, self.checkout)

        with closing(sqlite3.connect(database)) as connection:
            row = connection.execute(
                """
                SELECT job_id, submission_key, execution_state,
                       tracking_state, last_event_sequence
                FROM jobs
                WHERE job_id = ?
                """,
                ("job-first",),
            ).fetchone()

        self.assertEqual(
            row,
            (
                "job-first",
                "submission-first",
                "queued",
                "pending",
                0,
            ),
        )

    def test_job_identity_constraints(self):
        database = initialize_database(
            self.workspace, self.checkout
        )

        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO jobs (job_id, submission_key)
                    VALUES (?, ?)
                    """,
                    ("job-first", "submission-first"),
                )

            invalid_pairs = (
                ("job-first", "different-submission"),
                ("different-job", "submission-first"),
                ("different-job", ""),
            )

            for values in invalid_pairs:
                with self.subTest(values=values):
                    with self.assertRaises(sqlite3.IntegrityError):
                        with connection:
                            connection.execute(
                                """
                                INSERT INTO jobs (
                                    job_id, submission_key
                                )
                                VALUES (?, ?)
                                """,
                                values,
                            )

            count = connection.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]

        self.assertEqual(count, 1)

    def test_rejects_invalid_states_and_event_positions(self):
        database = initialize_database(
            self.workspace, self.checkout
        )

        invalid_settings = (
            ("unknown", "pending", 0),
            ("queued", "unknown", 0),
            ("queued", "pending", -1),
            ("queued", "pending", 1.5),
        )

        with closing(sqlite3.connect(database)) as connection:
            for execution, tracking, sequence in invalid_settings:
                with self.subTest(
                    execution=execution,
                    tracking=tracking,
                    sequence=sequence,
                ):
                    with self.assertRaises(sqlite3.IntegrityError):
                        with connection:
                            connection.execute(
                                """
                                INSERT INTO jobs (
                                    job_id,
                                    submission_key,
                                    execution_state,
                                    tracking_state,
                                    last_event_sequence
                                )
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (
                                    "job-invalid",
                                    "submission-invalid",
                                    execution,
                                    tracking,
                                    sequence,
                                ),
                            )

            count = connection.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]

        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()