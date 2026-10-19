"""Training must not require generator modules, resources or import-time secrets."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingImportTests(unittest.TestCase):
    def test_package_and_direct_script_without_generator_resources(self):
        with tempfile.TemporaryDirectory(prefix="foxtrot-training-import-") as directory:
            root = Path(directory)
            source = root / "src"
            source.mkdir()
            for name in ("train.py", "class_names.py"):
                shutil.copy2(PROJECT_ROOT / "src" / name, source / name)

            # Load the genuine training dependency first, then reject attempts to
            # load generation dependencies. There are no generator files/resources
            # or .env files anywhere in this isolated project.
            program = r'''
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import ultralytics
import yaml

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

class RejectGeneratorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'DataSetGeneration', 'Shapes', 'generate_letters',
            'resize_backgrounds', 'seaborn', 'colorama'
        }:
            raise AssertionError('Training imported generator dependency: ' + fullname)

sys.meta_path.insert(0, RejectGeneratorImports())
with patch('dotenv.load_dotenv', side_effect=AssertionError('Import read .env')):
    import src.train as training
    assert training.YOLO is ultralytics.YOLO
    assert training.SHAPES == ('Hexagon', 'Triangle')
    training.project_root = str(root)
    yaml_path = training.create_yolo_yaml(str(root / 'existing-dataset'))
    data = yaml.safe_load(Path(yaml_path).read_text())
    assert data == {
        'path': str(root / 'existing-dataset'),
        'train': 'train/images', 'val': 'val/images',
        'names': {0: 'Hexagon', 1: 'Triangle'}
    }, data
    sys.path.insert(0, str(root / 'src'))
    direct = runpy.run_path(str(root / 'src' / 'train.py'), run_name='verify_import')
    assert direct['YOLO'] is ultralytics.YOLO
    assert dict(direct['CLASS_NAMES']) == data['names']
    try:
        direct['CLASS_NAMES'][0] = 'Changed'
    except TypeError:
        pass
    else:
        raise AssertionError('Class mapping must be immutable')
    assert not any(name in sys.modules for name in
                   ['DataSetGeneration', 'Shapes', 'generate_letters', 'resize_backgrounds'])
print(json.dumps({'names': data['names'], 'package_import': True,
                  'direct_script_import': True, 'dotenv_reads': 0}))
'''
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = str(root / "mpl")
            environment["YOLO_CONFIG_DIR"] = str(root / "yolo-settings")
            result = subprocess.run(
                [sys.executable, "-c", program, str(root)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(summary["names"], {"0": "Hexagon", "1": "Triangle"})
            self.assertEqual(summary["dotenv_reads"], 0)


if __name__ == "__main__":
    unittest.main()
