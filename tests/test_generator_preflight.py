"""Check actual script behavior without touching the project's datasets."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GeneratorPreflightTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        shutil.copytree(PROJECT_ROOT / "src", root / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "letters*"))
        params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text())
        params["data_generation"]["n"] = 1
        params["data_generation"]["size"] = 128
        params["data_generation"]["letter"]["add_letter"] = 0
        (root / "params.yaml").write_text(yaml.safe_dump(params))
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        sentinel = root / "data" / "generated" / "images" / "sentinel.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("keep existing outputs on invalid input")
        return root, raw, sentinel

    def execute(self, root):
        environment = dict(os.environ, MPLCONFIGDIR=str(root / "mpl"),
                           PYTHONDONTWRITEBYTECODE="1", TERM="dumb")
        return subprocess.run(
            [sys.executable, "-B", str(root / "src/data_generation/DataSetGeneration.py")],
            cwd=root, env=environment, text=True, capture_output=True, timeout=30,
        )

    def test_bad_inputs_fail_before_clearing_previous_outputs(self):
        for case in ("empty", "unsupported", "corrupt", "missing"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="foxtrot-preflight-") as directory:
                root, raw, sentinel = self.fixture(directory)
                if case == "unsupported":
                    (raw / "notes.txt").write_text("not an image")
                elif case == "corrupt":
                    (raw / "broken.JPG").write_bytes(b"not an image")
                elif case == "missing":
                    raw.rmdir()
                result = self.execute(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(sentinel.read_text(), "keep existing outputs on invalid input")
                self.assertIn("background", result.stderr.lower())
                self.assertNotIn("Cannot choose from an empty sequence", result.stderr)
                self.assertNotIn("NoneType", result.stderr)

    def test_actual_script_accepts_uppercase_jpeg_and_skips_corrupt(self):
        with tempfile.TemporaryDirectory(prefix="foxtrot-preflight-good-") as directory:
            root, raw, sentinel = self.fixture(directory)
            image = np.full((96, 192, 3), (70, 140, 190), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(raw / "A.PNG"), image))
            self.assertTrue(cv2.imwrite(str(raw / "b.JPEG"), image))
            (raw / "broken.jpg").write_bytes(b"corrupt")
            (raw / "notes.txt").write_text("unsupported")
            (raw / "folder.png").mkdir()
            result = self.execute(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(sentinel.exists())
            images = sorted((root / "data/generated/images").glob("*.jpg"))
            labels = sorted((root / "data/generated/labels").glob("*.txt"))
            self.assertEqual(len(images), 2)
            self.assertEqual(len(labels), 2)
            self.assertTrue(all(cv2.imread(str(path)).shape[:2] == (128, 128) for path in images))
            self.assertIn("Skipped 1 unreadable", result.stderr)


if __name__ == "__main__":
    unittest.main()
