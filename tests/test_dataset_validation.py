"""V1.1 dataset checker: valid datasets pass; every kind of problem is reported."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from foxtrot.pipeline.dataset_validation import validate_dataset

CLASSES = ("Hexagon", "Triangle")
SPLITS = {"train": "train/images", "val": "val/images"}


# Tests for validate_dataset, each using a tiny throwaway dataset.
class DatasetValidationTests(unittest.TestCase):
    # Before each test: make an empty temporary dataset with train/val folders.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-validate-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for split in ("train", "val"):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir(parents=True)

    # Helper: save a small solid-colour image into a split.
    def image(self, split, name, colour=(255, 0, 0)):
        Image.new("RGB", (32, 32), colour).save(self.root / split / "images" / f"{name}.jpg")

    # Helper: write a label file with the given text into a split.
    def label(self, split, name, text):
        (self.root / split / "labels" / f"{name}.txt").write_text(text)

    # Helper: fill the dataset with one correct train image and one correct val image.
    def good(self):
        self.image("train", "a", (255, 0, 0))
        self.label("train", "a", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1")
        self.image("val", "b", (0, 255, 0))
        self.label("val", "b", "1 0.5 0.5 0.4 0.4\n")

    # Helper: run the checker on the temporary dataset.
    def check(self, **options):
        return validate_dataset(self.root, CLASSES, SPLITS, **options)

    # Helper: just the error messages from a report.
    def messages(self, report):
        return [issue.message for issue in report.errors]

    # A correct dataset is valid and objects are counted per class and split.
    def test_valid_dataset_passes_with_class_counts(self):
        self.good()
        report = self.check()
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.splits["train"].objects_per_class, {"Hexagon": 1, "Triangle": 1})
        self.assertEqual(report.splits["val"].objects_per_class, {"Hexagon": 0, "Triangle": 1})
        self.assertEqual((report.error_count, report.warning_count), (0, 0))

    # Every bad label line is reported with its line number; good lines still count.
    def test_bad_label_lines_are_all_reported_with_line_numbers(self):
        self.good()
        self.label("train", "a", "\n".join([
            "0 0.5 0.5 0.2",          # 4 values
            "x 0.5 0.5 0.2 0.2",      # not a number
            "2 0.5 0.5 0.2 0.2",      # class out of range
            "0 0.5 0.5 0 0.2",        # zero width
            "0 0.95 0.5 0.2 0.2",     # sticks out on the right
            "1 0.5 0.5 0.2 0.2",      # fine
        ]))
        report = self.check()
        self.assertFalse(report.valid)
        self.assertEqual([issue.line for issue in report.errors], [1, 2, 3, 4, 5])
        self.assertIn("Class 2 is outside 0–1", self.messages(report))
        self.assertEqual(report.splits["train"].objects_per_class["Triangle"], 1)

    # A corrupt image and an empty split both make the dataset invalid.
    def test_corrupt_image_and_empty_split_are_errors(self):
        self.good()
        (self.root / "train" / "images" / "broken.jpg").write_bytes(b"not an image")
        for path in (self.root / "val" / "images").iterdir():
            path.unlink()
        report = self.check()
        self.assertIn("Image cannot be read", self.messages(report))
        self.assertIn("Split has no images", self.messages(report))

    # A missing label is a warning by default, but an error in strict mode.
    def test_missing_label_follows_policy(self):
        self.good()
        self.image("train", "unlabelled", (0, 0, 255))
        background = self.check()
        self.assertTrue(background.valid)
        self.assertEqual(background.splits["train"].missing_labels, 1)
        self.assertEqual(background.warning_count, 1)
        self.assertFalse(self.check(missing_labels="error").valid)

    # The same picture in train and val, and labels with no image, are warnings.
    def test_duplicates_across_splits_and_orphan_labels_are_warnings(self):
        self.good()
        self.image("val", "copy", (255, 0, 0))  # same pixels as train/a
        self.label("val", "copy", "0 0.5 0.5 0.2 0.2")
        self.label("train", "ghost", "0 0.5 0.5 0.2 0.2")
        report = self.check()
        self.assertTrue(report.valid)
        warnings = [issue.message for issue in report.warnings]
        self.assertIn("Duplicate image in both train and val", warnings)
        self.assertIn("Label file has no matching image", warnings)

    # Missing folders, '..' paths and folders not named 'images' are errors.
    def test_missing_or_unsafe_split_folders_are_errors(self):
        report = validate_dataset(self.root, CLASSES, {"train": "nowhere/images", "val": "../val/images",
                                                       "test": "test/pictures"})
        self.assertEqual(report.error_count, 3)

    # 500 errors: the report keeps 200 examples but the count stays exact.
    def test_report_size_is_bounded_but_counts_are_exact(self):
        self.good()
        self.label("train", "a", "\n".join("9 0.5 0.5 0.1 0.1" for _ in range(500)))
        report = self.check()
        self.assertEqual(report.error_count, 500)
        self.assertEqual(len(report.errors), 200)


if __name__ == "__main__":
    unittest.main()
