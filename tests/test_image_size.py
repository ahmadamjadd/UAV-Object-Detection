"""Output-size regression tests never write to the repository dataset."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from src.data_generation.image_size import resize_background, validate_image_size


ROOT = Path(__file__).resolve().parents[1]


class ImageSizeTests(unittest.TestCase):
    def test_validation_rejects_invalid_values(self):
        for size in (0, -1, 160.0, True, False, "320", None):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "positive integer"):
                validate_image_size(size)

    def test_resize_uses_correct_resolution_and_preserves_matching_canvas(self):
        image = np.full((120, 240, 3), 50, dtype=np.uint8)
        for size in (160, 320, 640):
            self.assertEqual(resize_background(image, size).shape, (size, size, 3))
        square = np.full((320, 320, 3), 50, dtype=np.uint8)
        self.assertIs(resize_background(square, 320), square)

    def run_generator(self, size, letters=False):
        fixture = tempfile.TemporaryDirectory(prefix="foxtrot-size-")
        self.addCleanup(fixture.cleanup)
        root = Path(fixture.name)
        shutil.copytree(ROOT / "src" / "data_generation", root / "src" / "data_generation")
        shutil.copy2(ROOT / "src" / "class_names.py", root / "src" / "class_names.py")
        if letters:
            cache = root / "src" / "data_generation" / "res" / f"letters{size}"
            cache.mkdir(exist_ok=True)
            for cached in cache.glob("*.png"):
                cached.unlink()
            cv2.imwrite(str(cache / "A.png"), np.full((17, 23, 3), 180, dtype=np.uint8))
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        for index, (height, width) in enumerate(((320, 320), (320, 640), (640, 320))):
            cv2.imwrite(str(raw / f"{index}.png"), np.full((height, width, 3), 180, dtype=np.uint8))
        config = yaml.safe_load((ROOT / "params.yaml").read_text())
        config["data_generation"].update({"size": size, "n": 2, "noise": 0})
        config["data_generation"]["letter"].update({"add_letter": int(letters), "letter_prob": 1.0})
        config["data_generation"]["blur"] = {"b0": 0, "b1": 0}
        (root / "params.yaml").write_text(yaml.safe_dump(config))
        sentinel = root / "data" / "generated" / "images" / "keep.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("existing dataset")
        env = os.environ.copy()
        env.update({"MPLCONFIGDIR": str(root / "mpl"), "TERM": "xterm"})
        result = subprocess.run(
            [sys.executable, str(root / "src" / "data_generation" / "DataSetGeneration.py")],
            cwd=root, env=env, capture_output=True, text=True, timeout=60,
        )
        return root, sentinel, result

    def test_entrypoint_resizes_mixed_backgrounds_before_generation(self):
        for size in (160, 320, 640):
            with self.subTest(size=size):
                root, sentinel, result = self.run_generator(size)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                images = list((root / "data" / "generated" / "images").glob("*.jpg"))
                self.assertEqual(len(images), 4, result.stdout + result.stderr)
                for image in images:
                    self.assertEqual(cv2.imread(str(image)).shape[:2], (size, size))
                self.assertFalse(sentinel.exists())

    def test_incomplete_letter_cache_is_regenerated(self):
        root, _, result = self.run_generator(320, letters=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(list((root / "data" / "generated" / "images").glob("*.jpg"))), 4,
                         result.stdout + result.stderr)
        resources = list((root / "src" / "data_generation" / "res" / "letters320").glob("*.png"))
        self.assertEqual(len(resources), 36)
        for resource in resources:
            self.assertEqual(cv2.imread(str(resource)).shape[:2], (320, 320))

    def test_invalid_yaml_size_fails_before_output_cleanup(self):
        for size in (0, -1, 160.0, True):
            with self.subTest(size=size):
                _, sentinel, result = self.run_generator(size)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("data_generation.size must be a positive integer", result.stderr)
                self.assertEqual(sentinel.read_text(), "existing dataset")


if __name__ == "__main__":
    unittest.main()
