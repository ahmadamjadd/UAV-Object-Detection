"""Background validation tests use only temporary fixtures."""

from pathlib import Path
import tempfile
import unittest
import warnings

import cv2
import numpy as np

from src.data_generation.backgrounds import discover_backgrounds, read_background


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="foxtrot-backgrounds-")
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)

    def write_image(self, name):
        path = self.root / name
        image = np.full((17, 23, 3), (12, 80, 160), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def test_sorted_mixed_extensions_exclude_corrupt_and_directories(self):
        self.write_image("z.JPG")
        self.write_image("A.PNG")
        self.write_image("middle.JPEG")
        (self.root / "nested.png").mkdir()
        (self.root / "notes.txt").write_text("unsupported")
        (self.root / "bad.png").write_bytes(b"not an image")
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            paths = discover_backgrounds(str(self.root))
        self.assertEqual([Path(path).name for path in paths], ["A.PNG", "middle.JPEG", "z.JPG"])
        self.assertTrue(all(Path(path).is_absolute() for path in paths))
        self.assertEqual(len(emitted), 1)
        self.assertIn("bad.png", str(emitted[0].message))
        self.assertEqual(read_background(paths[0]).shape, (17, 23, 3))

    def test_empty_and_unsupported_only_fail_immediately(self):
        with self.assertRaisesRegex(ValueError, "No readable background images"):
            discover_backgrounds(str(self.root))
        (self.root / "image.webp").write_bytes(b"unsupported")
        with self.assertRaisesRegex(ValueError, "No readable background images"):
            discover_backgrounds(str(self.root))

    def test_corrupt_only_fails_immediately(self):
        (self.root / "bad.jpg").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "1 supported files could not be decoded"):
            discover_backgrounds(str(self.root))

    def test_missing_and_non_directory_inputs(self):
        with self.assertRaises(FileNotFoundError):
            discover_backgrounds(str(self.root / "missing"))
        path = self.write_image("image.png")
        with self.assertRaises(NotADirectoryError):
            discover_backgrounds(str(path))

    def test_image_removed_or_corrupted_after_discovery(self):
        path = self.write_image("image.png")
        discovered = discover_backgrounds(str(self.root))[0]
        path.write_bytes(b"now corrupted")
        with self.assertRaisesRegex(ValueError, "Cannot decode background image"):
            read_background(discovered)
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Background image does not exist"):
            read_background(discovered)

    def test_warning_count_is_bounded_for_many_bad_files(self):
        self.write_image("good.png")
        for index in range(20):
            (self.root / f"bad{index:02}.png").write_bytes(b"corrupt")
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            discover_backgrounds(str(self.root))
        self.assertEqual(len(emitted), 1)
        self.assertIn("20 unreadable", str(emitted[0].message))
        self.assertIn("...", str(emitted[0].message))


if __name__ == "__main__":
    unittest.main()
