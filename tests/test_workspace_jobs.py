"""Job-directory checks use only owned temporary fixtures."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

from foxtrot.persistence.jobs import create_job_directory
from foxtrot.persistence.workspace import initialize_workspace


class JobDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(
            prefix="foxtrot-job-directory-"
        )
        self.addCleanup(self.fixture.cleanup)

        self.root = Path(self.fixture.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()

        self.workspace = self.root / "workspace"

    def test_jobs_have_distinct_ids_and_separate_outputs(self):
        first = create_job_directory(
            self.workspace, self.checkout
        )
        first_output = first / "output.txt"
        first_output.write_text("first job", encoding="utf-8")

        second = create_job_directory(
            self.workspace, self.checkout
        )
        second_output = second / "output.txt"
        second_output.write_text("second job", encoding="utf-8")

        self.assertNotEqual(first.name, second.name)
        self.assertEqual(first.parent, self.workspace / "jobs")
        self.assertEqual(second.parent, self.workspace / "jobs")

        self.assertEqual(
            first_output.read_text(encoding="utf-8"),
            "first job",
        )
        self.assertEqual(
            second_output.read_text(encoding="utf-8"),
            "second job",
        )

    def test_collision_preserves_existing_job(self):
        identifier = UUID(
            "12345678-1234-4234-8234-123456789abc"
        )

        with patch(
            "foxtrot.persistence.jobs.uuid4",
            return_value=identifier,
        ):
            first = create_job_directory(
                self.workspace, self.checkout
            )
            saved = first / "keep.txt"
            saved.write_text("keep this", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                create_job_directory(
                    self.workspace, self.checkout
                )

        self.assertEqual(
            saved.read_text(encoding="utf-8"),
            "keep this",
        )
        self.assertEqual(
            list((self.workspace / "jobs").iterdir()),
            [first],
        )

    def test_rejects_unrecognized_workspace(self):
        self.workspace.mkdir()
        saved = self.workspace / "keep.txt"
        saved.write_text("unrelated content", encoding="utf-8")

        with self.assertRaises(ValueError):
            create_job_directory(
                self.workspace, self.checkout
            )

        self.assertEqual(
            saved.read_text(encoding="utf-8"),
            "unrelated content",
        )
        self.assertEqual(
            list(self.workspace.iterdir()),
            [saved],
        )

    def test_rejects_jobs_symlink(self):
        initialize_workspace(self.workspace, self.checkout)

        jobs = self.workspace / "jobs"
        jobs.rmdir()

        external = self.root / "external-folder"
        external.mkdir()
        jobs.symlink_to(external, target_is_directory=True)

        with self.assertRaises(ValueError):
            create_job_directory(
                self.workspace, self.checkout
            )

        self.assertEqual(list(external.iterdir()), [])


if __name__ == "__main__":
    unittest.main()