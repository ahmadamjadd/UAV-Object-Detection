"""Recorded-job checks use temporary workspaces and databases."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from foxtrot.persistence.jobs import create_recorded_job

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from foxtrot.persistence.database import initialize_database


class RecordedJobTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(
            prefix="foxtrot-recorded-job-"
        )
        self.addCleanup(self.fixture.cleanup)

        self.root = Path(self.fixture.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()

        self.workspace = self.root / "workspace"

    def request(self):
        return {
            "dataset_id": "dataset-demo",
            "tracking_destination_id": "local-mlflow",
            "training": {"epochs": 5},
        }

    def test_records_job_with_saved_configurations(self):
        request = self.request()

        job = create_recorded_job(
            self.workspace,
            self.checkout,
            request,
            submission_key="submission-first",
        )

        submitted = json.loads(
            (job / "submitted.json").read_text(encoding="utf-8")
        )
        resolved = json.loads(
            (job / "resolved.json").read_text(encoding="utf-8")
        )

        with closing(
            sqlite3.connect(self.workspace / "app.sqlite")
        ) as connection:
            row = connection.execute(
                """
                SELECT job_id, execution_state, tracking_state,
                       last_event_sequence
                FROM jobs
                WHERE submission_key = ?
                """,
                ("submission-first",),
            ).fetchone()

        self.assertEqual(submitted, request)
        self.assertEqual(resolved["training"]["batch_size"], 16)
        self.assertEqual(
            row,
            (job.name, "queued", "pending", 0),
        )

    def test_records_two_jobs_with_separate_configurations(self):
        first = create_recorded_job(
            self.workspace,
            self.checkout,
            self.request(),
            submission_key="submission-first",
        )

        second_request = self.request()
        second_request["training"]["epochs"] = 9

        second = create_recorded_job(
            self.workspace,
            self.checkout,
            second_request,
            submission_key="submission-second",
        )

        first_config = json.loads(
            (first / "resolved.json").read_text(encoding="utf-8")
        )
        second_config = json.loads(
            (second / "resolved.json").read_text(encoding="utf-8")
        )

        with closing(
            sqlite3.connect(self.workspace / "app.sqlite")
        ) as connection:
            recorded_ids = {
                row[0]
                for row in connection.execute(
                    "SELECT job_id FROM jobs"
                ).fetchall()
            }

        self.assertNotEqual(first, second)
        self.assertEqual(first_config["training"]["epochs"], 5)
        self.assertEqual(second_config["training"]["epochs"], 9)
        self.assertEqual(recorded_ids, {first.name, second.name})

    def test_invalid_inputs_create_no_workspace(self):
        invalid_request = self.request()
        invalid_request["training"]["epochs"] = 0

        cases = (
            (self.request(), "", ValueError),
            (self.request(), "   ", ValueError),
            (self.request(), 123, ValueError),
            (invalid_request, "submission-first", ValidationError),
        )

        for request, key, expected_error in cases:
            with self.subTest(key=key, error=expected_error):
                with self.assertRaises(expected_error):
                    create_recorded_job(
                        self.workspace,
                        self.checkout,
                        request,
                        submission_key=key,
                    )

                self.assertFalse(self.workspace.exists())

    def test_retry_returns_existing_job_without_changes(self):
        first = create_recorded_job(
            self.workspace,
            self.checkout,
            self.request(),
            submission_key="submission-first",
        )

        before = (first / "resolved.json").read_bytes()

        retried = create_recorded_job(
            self.workspace,
            self.checkout,
            self.request(),
            submission_key="submission-first",
        )

        with closing(
            sqlite3.connect(self.workspace / "app.sqlite")
        ) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]

        self.assertEqual(retried, first)
        self.assertEqual(count, 1)
        self.assertEqual(
            set((self.workspace / "jobs").iterdir()),
            {first},
        )
        self.assertEqual(
            (first / "resolved.json").read_bytes(),
            before,
        )

    def test_configuration_failure_commits_no_job_record(self):
        with patch(
            "foxtrot.persistence.jobs.create_configured_job",
            side_effect=OSError("fixture write failure"),
        ):
            with self.assertRaisesRegex(
                OSError,
                "fixture write failure",
            ):
                create_recorded_job(
                    self.workspace,
                    self.checkout,
                    self.request(),
                    submission_key="submission-first",
                )

        with closing(
            sqlite3.connect(self.workspace / "app.sqlite")
        ) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]

        self.assertEqual(count, 0)

    def test_rejects_changed_submission_with_existing_key(self):
        first = create_recorded_job(
            self.workspace,
            self.checkout,
            self.request(),
            submission_key="submission-first",
        )
        before = (first / "resolved.json").read_bytes()

        changed_epochs = self.request()
        changed_epochs["training"]["epochs"] = 9

        explicit_default = self.request()
        explicit_default["training"]["batch_size"] = 16

        for request in (changed_epochs, explicit_default):
            with self.subTest(request=request):
                with self.assertRaisesRegex(
                    ValueError,
                    "different settings",
                ):
                    create_recorded_job(
                        self.workspace,
                        self.checkout,
                        request,
                        submission_key="submission-first",
                    )

        self.assertEqual(
            set((self.workspace / "jobs").iterdir()),
            {first},
        )
        self.assertEqual(
            (first / "resolved.json").read_bytes(),
            before,
        )

    def test_retry_rejects_missing_configuration(self):
        first = create_recorded_job(
            self.workspace,
            self.checkout,
            self.request(),
            submission_key="submission-first",
        )
        missing = first / "resolved.json"
        missing.unlink()

        with self.assertRaisesRegex(ValueError, "configuration"):
            create_recorded_job(
                self.workspace,
                self.checkout,
                self.request(),
                submission_key="submission-first",
            )

        with closing(
            sqlite3.connect(self.workspace / "app.sqlite")
        ) as connection:
            row = connection.execute(
                "SELECT job_id FROM jobs WHERE submission_key = ?",
                ("submission-first",),
            ).fetchone()

        self.assertEqual(row, (first.name,))
        self.assertFalse(missing.exists())
        self.assertEqual(
            set((self.workspace / "jobs").iterdir()),
            {first},
        )

    def test_simultaneous_submissions_create_one_job(self):
        initialize_database(self.workspace, self.checkout)
        ready = Barrier(2)

        def submit():
            ready.wait(timeout=10)
            return create_recorded_job(
                self.workspace,
                self.checkout,
                self.request(),
                submission_key="submission-first",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(submit)
                for _ in range(2)
            ]
            results = [
                future.result(timeout=15)
                for future in futures
            ]

        with closing(
            sqlite3.connect(self.workspace / "app.sqlite")
        ) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]

        self.assertEqual(results[0], results[1])
        self.assertEqual(count, 1)
        self.assertEqual(
            set((self.workspace / "jobs").iterdir()),
            {results[0]},
        )


if __name__ == "__main__":
    unittest.main()