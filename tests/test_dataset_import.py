"""V1.2 importing an existing dataset: valid ones are copied and registered safely."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from foxtrot.persistence.dataset_import import import_dataset
from foxtrot.persistence.datasets import DatasetUnavailable, list_datasets, load_dataset

CLASSES = ("Hexagon", "Triangle")


# Tests for import_dataset using a small dataset outside the workspace.
class DatasetImportTests(unittest.TestCase):
    # Before each test: make a checkout, a workspace path and a valid source dataset.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-import-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.source = self.root / "my-dataset"
        for split, colour in (("train", (255, 0, 0)), ("val", (0, 255, 0))):
            (self.source / split / "images").mkdir(parents=True)
            (self.source / split / "labels").mkdir(parents=True)
            Image.new("RGB", (32, 32), colour).save(self.source / split / "images" / "a.jpg")
            (self.source / split / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    # Helper: import the source dataset under the given ID and class names.
    def run_import(self, dataset_id="shapes", class_names=CLASSES, **options):
        return import_dataset(self.workspace, self.checkout, source=self.source,
                              dataset_id=dataset_id, class_names=class_names, **options)

    # A valid dataset is copied, marked valid, fingerprinted and loadable.
    def test_valid_dataset_is_copied_registered_and_loadable(self):
        dataset, report = self.run_import()
        self.assertTrue(report.valid)
        self.assertEqual(dataset.metadata.validation_status, "valid")
        self.assertEqual(load_dataset(self.workspace, self.checkout, "shapes"), dataset)
        self.assertEqual([d.dataset_id for d in list_datasets(self.workspace, self.checkout)], ["shapes"])
        copied = Path(dataset.root)
        self.assertTrue((copied / "train" / "labels" / "a.txt").is_file())
        saved = json.loads((copied.parent / "validation.json").read_text())
        self.assertEqual(saved["source"], str(self.source))
        self.assertTrue((self.source / "train" / "images" / "a.jpg").is_file())  # original untouched

    # Only the split folders are copied; YOLO cache files and unrelated files are left behind.
    def test_only_split_folders_are_copied_and_caches_skipped(self):
        (self.source / "notes.txt").write_text("not data")
        (self.source / "train" / "labels.cache").write_bytes(b"cache")
        (self.source / "train" / "labels" / "x.cache").write_bytes(b"cache")
        dataset, _ = self.run_import()
        copied = Path(dataset.root)
        self.assertFalse((copied / "notes.txt").exists())
        self.assertFalse((copied / "train" / "labels" / "x.cache").exists())

    # An invalid dataset is not imported and leaves nothing in the workspace.
    def test_invalid_dataset_is_not_imported(self):
        (self.source / "val" / "labels" / "a.txt").write_text("5 0.5 0.5 0.2 0.2\n")
        dataset, report = self.run_import()
        self.assertIsNone(dataset)
        self.assertFalse(report.valid)
        self.assertEqual(report.errors[0].message, "Class 5 is outside 0–1")
        self.assertEqual(list((self.workspace / "datasets").iterdir()), [])

    # The same ID cannot be imported twice.
    def test_dataset_id_can_only_be_used_once(self):
        self.run_import()
        with self.assertRaisesRegex(DatasetUnavailable, "already used"):
            self.run_import()

    # Relative paths, missing folders, the workspace itself and bad IDs are refused.
    def test_unsafe_sources_and_ids_are_refused(self):
        for options in ({"source": "relative/path"}, {"source": self.root / "missing"},
                        {"source": self.workspace}):
            with self.assertRaises(DatasetUnavailable):
                import_dataset(self.workspace, self.checkout, dataset_id="x", class_names=CLASSES, **options)
        for bad_id in ("../escape", "", "has space"):
            with self.assertRaises(DatasetUnavailable):
                self.run_import(dataset_id=bad_id)
        with self.assertRaises(ValueError):
            self.run_import(class_names=("Hexagon", "Hexagon"))

    # A symbolic link inside the dataset stops the import and leaves no partial copy.
    def test_symbolic_link_aborts_without_partial_dataset(self):
        outside = self.root / "outside.jpg"
        Image.new("RGB", (32, 32), (0, 0, 255)).save(outside)
        (self.source / "train" / "images" / "link.jpg").symlink_to(outside)
        with self.assertRaisesRegex(DatasetUnavailable, "symbolic links"):
            self.run_import()
        self.assertEqual(list((self.workspace / "datasets").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
