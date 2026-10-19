"""Run H1 checks using temporary caches and verify protected files stay unchanged.

Run with the installed baseline interpreter, from any working directory.
This does not start training, generate production data, or contact MLflow.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "data", "runs", "params.yaml", "data.yaml", "dvc.yaml", "dvc.lock",
    "src/class_names.py", "src/train.py", "src/test.py", "src/split_dataset.py",
    "src/data_generation",
)


def protected_hashes():
    result = {}
    for name in PROTECTED:
        root = ROOT / name
        paths = root.rglob("*") if root.is_dir() else (root,)
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                with path.open("rb") as stream:
                    result[str(path.relative_to(ROOT))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
    return result


def main():
    before = protected_hashes()
    return_code = 0
    with tempfile.TemporaryDirectory(prefix="foxtrot-baseline-") as temporary:
        environment = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            MPLCONFIGDIR=str(Path(temporary) / "mpl"),
            YOLO_CONFIG_DIR=str(Path(temporary) / "yolo"),
            MPLBACKEND="Agg",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            DO_NOT_TRACK="1",
        )
        # Verify installed-package discovery outside the checkout, and keep API
        # imports independent of training/generator imports and credential reads.
        smoke = """
import importlib
import importlib.metadata
import sys
for name in ('foxtrot', 'foxtrot.config', 'foxtrot.persistence',
             'foxtrot.execution', 'foxtrot.pipeline', 'foxtrot.tracking', 'foxtrot.api'):
    importlib.import_module(name)
assert not {'torch', 'ultralytics', 'mlflow', 'dotenv', 'DataSetGeneration'} & sys.modules.keys()
print('Installed application skeleton:', importlib.metadata.version('foxtrot-local'))
"""
        try:
            subprocess.run([sys.executable, "-B", "-c", smoke], cwd=temporary,
                           env=environment, check=True)
            subprocess.run([sys.executable, "-B", "-m", "unittest", "discover",
                            "-s", "tests", "-v"], cwd=ROOT, env=environment, check=True)
        except subprocess.CalledProcessError as error:
            return_code = error.returncode or 1
        finally:
            after = protected_hashes()
            changed = sorted(key for key in before.keys() | after.keys()
                             if before.get(key) != after.get(key))
            if changed:
                print("FAIL: protected files changed:", *changed, sep="\n", file=sys.stderr)
                return_code = 1
            else:
                print(f"Protected files unchanged: {len(before)}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
