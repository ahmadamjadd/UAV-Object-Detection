"""Exercise real generation in isolated copies; never replace repository datasets."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def generate_fixture(fixture, reverse, preseed):
    import contextlib
    import io
    import random
    import cv2
    import numpy as np
    import yaml

    source = fixture / "src/data_generation"
    shutil.copytree(ROOT / "src/data_generation", source,
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "src/class_names.py", fixture / "src/class_names.py")
    raw = fixture / "data/raw"
    raw.mkdir(parents=True)
    for index, name in enumerate(("z.png", "a.png", "middle.png")):
        image = np.full((320, 320, 3), (30 + index * 60, 80, 180), np.uint8)
        assert cv2.imwrite(str(raw / name), image)
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    params["data_generation"]["n"] = 3
    # Include both letter and noise paths, using the repository's remaining defaults.
    params["data_generation"]["letter"] = {"add_letter": True, "letter_prob": 1.0}
    (fixture / "params.yaml").write_text(yaml.safe_dump(params))
    original_listdir = os.listdir
    original_iterdir = Path.iterdir
    natural_background_order = [path.name for path in original_iterdir(raw)]
    if reverse:
        os.listdir = lambda path: list(reversed(original_listdir(path)))
        # Perturb the pathlib entry point actually used by discovery, independently
        # of whether this Python version implements it through listdir or scandir.
        def reordered_iterdir(path):
            if path.resolve() == raw.resolve():
                return iter(path / name for name in reversed(natural_background_order))
            entries = list(original_iterdir(path))
            return iter(entries)
        Path.iterdir = reordered_iterdir
    enumerated_background_order = [path.name for path in raw.iterdir()]
    random.seed(preseed)
    sys.path.insert(0, str(source))
    import DataSetGeneration as generator

    generator.os.system = lambda command: 0
    generator.DEBUG = True
    with contextlib.redirect_stdout(io.StringIO()):
        generator.load_params()
        generator.main()
    generated = fixture / "data/generated"
    manifest = {
        str(path.relative_to(generated)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(generated.rglob("*")) if path.is_file()
    }
    # The old import-time saturation default is also exercised directly here.
    sample = np.full((3, 3, 3), (55, 120, 200), dtype=np.uint8)
    random.seed(42)
    implicit_saturation = hashlib.sha256(generator.saturate(sample).tobytes()).hexdigest()
    # Loading another mask directory must discard letters from the previous one.
    replacement = fixture / "replacement_letters"
    replacement.mkdir()
    assert cv2.imwrite(str(replacement / "A.png"), np.zeros((8, 8, 3), np.uint8))
    generator.sh.initialize_letter_masks(str(replacement))
    print(json.dumps({"manifest": manifest, "saturation": implicit_saturation,
                      "default_is_none": generator.saturate.__defaults__ == (None,),
                      "replacement_masks": sorted(generator.sh.Letter_Mask),
                      "natural_background_order": natural_background_order,
                      "enumerated_background_order": enumerated_background_order}))


class DeterminismTests(unittest.TestCase):
    def test_identical_seed_ignores_directory_order_and_preimport_random_state(self):
        with tempfile.TemporaryDirectory(prefix="foxtrot-determinism-") as temporary:
            results = []
            for index, reverse, preseed in ((0, False, 1), (1, False, 2), (2, True, 2)):
                fixture = Path(temporary) / f"case{index}"
                environment = dict(os.environ, MPLCONFIGDIR=str(Path(temporary) / "mpl"),
                                   OPENBLAS_NUM_THREADS="1")
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--fixture",
                     str(fixture), str(int(reverse)), str(preseed)],
                    capture_output=True, text=True, timeout=90, env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                results.append(json.loads(result.stdout))
            for result in results:
                self.assertEqual(sum(name.startswith("images/") for name in result["manifest"]), 6)
                self.assertEqual(sum(name.startswith("labels/") for name in result["manifest"]), 6)
                self.assertTrue(result["default_is_none"])
                self.assertEqual(result["replacement_masks"], ["A"])
            self.assertEqual(results[0]["enumerated_background_order"],
                             results[0]["natural_background_order"])
            self.assertEqual(results[2]["enumerated_background_order"],
                             list(reversed(results[2]["natural_background_order"])))
            self.assertNotEqual(results[0]["enumerated_background_order"],
                                results[2]["enumerated_background_order"])
            for result in results[1:]:
                self.assertEqual(results[0]["manifest"], result["manifest"])
                self.assertEqual(results[0]["saturation"], result["saturation"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fixture":
        generate_fixture(Path(sys.argv[2]), bool(int(sys.argv[3])), int(sys.argv[4]))
    else:
        unittest.main()
