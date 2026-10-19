"""Workspace initialization checks use owned temporary fixtures."""

from pathlib import Path
import tempfile
import unittest

from foxtrot.persistence.workspace import (
    WORKSPACE_FORMAT,
    WORKSPACE_MARKER,
    initialize_workspace,
)


class WorkspaceInitializationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(
            prefix="foxtrot-workspace-init-"
        )
        self.addCleanup(self.fixture.cleanup)

        self.root = Path(self.fixture.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()

    def test_initializes_new_and_empty_directories(self):
        empty = self.root / "empty-workspace"
        empty.mkdir()

        for proposed in (self.root / "new-workspace", empty):
            with self.subTest(path=proposed):
                workspace = initialize_workspace(proposed, self.checkout)

                self.assertEqual(workspace, proposed)
                self.assertTrue((workspace / "jobs").is_dir())
                self.assertEqual(
                    (workspace / WORKSPACE_MARKER).read_text(
                        encoding="utf-8"
                    ),
                    WORKSPACE_FORMAT,
                )

    def test_reopening_preserves_existing_job_files(self):
        workspace = initialize_workspace(
            self.root / "workspace", self.checkout
        )
        job = workspace / "jobs" / "existing-job"
        job.mkdir()
        saved = job / "resolved.json"
        saved.write_text("existing content", encoding="utf-8")

        reopened = initialize_workspace(workspace, self.checkout)

        self.assertEqual(reopened, workspace)
        self.assertEqual(
            saved.read_text(encoding="utf-8"), "existing content"
        )

    def test_rejects_nonempty_unrecognized_directory(self):
        proposed = self.root / "unrelated-folder"
        proposed.mkdir()
        saved = proposed / "keep.txt"
        saved.write_text("keep this", encoding="utf-8")

        with self.assertRaises(ValueError):
            initialize_workspace(proposed, self.checkout)

        self.assertEqual(saved.read_text(encoding="utf-8"), "keep this")
        self.assertFalse((proposed / WORKSPACE_MARKER).exists())
        self.assertFalse((proposed / "jobs").exists())

    def test_rejects_invalid_marker_without_overwriting_it(self):
        proposed = self.root / "workspace"
        proposed.mkdir()
        marker = proposed / WORKSPACE_MARKER
        marker.write_text("unknown-format\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            initialize_workspace(proposed, self.checkout)

        self.assertEqual(
            marker.read_text(encoding="utf-8"), "unknown-format\n"
        )
        self.assertFalse((proposed / "jobs").exists())

    def test_rejects_marker_symlink(self):
        proposed = self.root / "workspace"
        proposed.mkdir()
        external = self.root / "external-marker"
        external.write_text(WORKSPACE_FORMAT, encoding="utf-8")
        (proposed / WORKSPACE_MARKER).symlink_to(external)

        with self.assertRaises(ValueError):
            initialize_workspace(proposed, self.checkout)

        self.assertFalse((proposed / "jobs").exists())

    def test_rejects_jobs_symlink(self):
        workspace = initialize_workspace(
            self.root / "workspace", self.checkout
        )
        jobs = workspace / "jobs"
        jobs.rmdir()

        external = self.root / "external-folder"
        external.mkdir()
        jobs.symlink_to(external, target_is_directory=True)

        with self.assertRaises(ValueError):
            initialize_workspace(workspace, self.checkout)

        self.assertEqual(list(external.iterdir()), [])

    def test_rejects_file_at_jobs_location(self):
        workspace = initialize_workspace(
            self.root / "workspace", self.checkout
        )
        jobs = workspace / "jobs"
        jobs.rmdir()
        jobs.write_text("keep this file", encoding="utf-8")

        with self.assertRaises(NotADirectoryError):
            initialize_workspace(workspace, self.checkout)

        self.assertEqual(
            jobs.read_text(encoding="utf-8"), "keep this file"
        )


if __name__ == "__main__":
    unittest.main()