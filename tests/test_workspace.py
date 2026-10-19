"""Workspace-location checks use only owned temporary fixtures."""

from pathlib import Path
import tempfile
import unittest

from foxtrot.persistence.workspace import resolve_workspace_root


class WorkspacePathTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(
            prefix="foxtrot-workspace-test-"
        )
        self.addCleanup(self.fixture.cleanup)

        self.root = Path(self.fixture.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()

    def test_accepts_new_external_location_without_creating_it(self):
        proposed = self.root / "workspace"

        resolved = resolve_workspace_root(proposed, self.checkout)

        self.assertEqual(resolved, proposed)
        self.assertFalse(proposed.exists())

    def test_accepts_existing_external_directory(self):
        proposed = self.root / "workspace"
        proposed.mkdir()

        self.assertEqual(
            resolve_workspace_root(proposed, self.checkout),
            proposed,
        )

    def test_rejects_overlapping_locations(self):
        for proposed in (
            self.checkout,
            self.checkout / "outputs",
            self.root,
        ):
            with self.subTest(path=proposed):
                with self.assertRaises(ValueError):
                    resolve_workspace_root(proposed, self.checkout)

    def test_rejects_relative_workspace(self):
        with self.assertRaises(ValueError):
            resolve_workspace_root("relative-workspace", self.checkout)

    def test_rejects_existing_file(self):
        proposed = self.root / "existing-file"
        proposed.write_text("fixture", encoding="utf-8")

        with self.assertRaises(NotADirectoryError):
            resolve_workspace_root(proposed, self.checkout)

    def test_resolves_external_directory_symlink(self):
        target = self.root / "external-workspace"
        target.mkdir()

        alias = self.root / "workspace-link"
        alias.symlink_to(target, target_is_directory=True)

        self.assertEqual(
            resolve_workspace_root(alias, self.checkout),
            target,
        )

    def test_rejects_symlink_into_checkout(self):
        alias = self.root / "checkout-link"
        alias.symlink_to(self.checkout, target_is_directory=True)

        with self.assertRaises(ValueError):
            resolve_workspace_root(
                alias / "new-output",
                self.checkout,
            )

    def test_rejects_missing_checkout(self):
        with self.assertRaises(FileNotFoundError):
            resolve_workspace_root(
                self.root / "workspace",
                self.root / "missing-checkout",
            )


if __name__ == "__main__":
    unittest.main()