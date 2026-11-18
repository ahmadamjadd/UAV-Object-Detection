"""V1.4 real YOLO trainer: a tiny CPU training run, cancellation, and missing weights.

Each training runs in a fresh Python process, like the real job child process.
Uses random starting weights ("scratch"), so no download is needed.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

from foxtrot.persistence.dataset_import import import_dataset
from foxtrot.persistence.datasets import load_dataset

# Runs one training in a child process and prints the outcome as JSON.
RUNNER = r"""
import json, sys, threading
from foxtrot.config.job import JobSpec
from foxtrot.pipeline.contracts import DatasetInput, PipelineCancelled, PipelineFailure, TrainingInput
from foxtrot.pipeline.ultralytics_adapter import UltralyticsTrainer

args = json.loads(sys.argv[1])

class Reporter:
    def __init__(self, cancelled):
        self.events, self.cancelled = [], cancelled
    def progress(self, payload):
        self.events.append(payload)
    def log(self, payload):
        pass
    def wait_cancelled(self, seconds=0.0):
        return self.cancelled

reporter = Reporter(args["cancel"])
request = TrainingInput(
    job_id="a" * 32, resolved=JobSpec.model_validate_json(args["resolved"]), resolved_sha256="b" * 64,
    dataset=DatasetInput.model_validate_json(args["dataset"]), data_yaml=args["data_yaml"],
    job_directory=args["job_directory"], output_directory=args["output_directory"])
try:
    result = UltralyticsTrainer(initialization=args["initialization"], workers=0).train(request, reporter)
    print(json.dumps({"outcome": "finished", "result": result.model_dump(mode="json"), "progress": reporter.events}))
except PipelineCancelled:
    print(json.dumps({"outcome": "cancelled", "progress": reporter.events}))
except PipelineFailure as failure:
    print(json.dumps({"outcome": "failed", "code": failure.code}))
print("ultralytics imported:", "ultralytics" in sys.modules, file=sys.stderr)
"""


# Tests for UltralyticsTrainer using a tiny generated dataset of coloured squares.
class UltralyticsTrainerTests(unittest.TestCase):
    # Before each test: build and import a 12-image dataset, and make a job folder.
    def setUp(self):
        from foxtrot.pipeline.yolo_dataset import write_yolo_data_yaml
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-yolo-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        source = self.root / "squares"
        for split, count in (("train", 8), ("val", 4)):
            (source / split / "images").mkdir(parents=True)
            (source / split / "labels").mkdir(parents=True)
            for index in range(count):
                image = Image.new("RGB", (64, 64), (30, 30, 30))
                ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill=(200, 50 + index * 10, 50))
                image.save(source / split / "images" / f"{index}.jpg")
                (source / split / "labels" / f"{index}.txt").write_text(f"{index % 2} 0.5 0.5 0.5 0.5\n")
        self.dataset, _ = import_dataset(self.workspace, self.checkout, source=source,
                                         dataset_id="squares", class_names=("Red", "Orange"))
        self.job = self.workspace / "jobs" / ("c" * 32)
        (self.job / "output").mkdir(parents=True)
        self.data_yaml = write_yolo_data_yaml(self.dataset, self.job / "output" / "data.yaml")

    # Helper: run the trainer in a fresh process and return its JSON outcome.
    def train(self, *, initialization="scratch", cancel=False, epochs=2):
        resolved = json.dumps({"dataset_id": "squares", "tracking_destination_id": "local",
                               "training": {"epochs": epochs, "batch_size": 4, "image_size": 64, "seed": 1}})
        args = {"resolved": resolved, "dataset": self.dataset.model_dump_json(), "data_yaml": str(self.data_yaml),
                "job_directory": str(self.job), "output_directory": str(self.job / "output"),
                "initialization": initialization, "cancel": cancel}
        result = subprocess.run([sys.executable, "-B", "-c", RUNNER, json.dumps(args)], cwd=self.root,
                                capture_output=True, text=True, timeout=600,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2"))
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        return json.loads(result.stdout.strip().splitlines()[-1]), result.stderr

    # A real 2-epoch run reports both epochs and returns the exact best.pt and its hash.
    def test_tiny_real_training_returns_exact_checkpoint(self):
        outcome, _ = self.train()
        self.assertEqual(outcome["outcome"], "finished")
        self.assertEqual([p["epoch"] for p in outcome["progress"]], [1, 2])
        self.assertIn("metrics/mAP50-95", outcome["progress"][-1])
        result = outcome["result"]
        self.assertEqual(result["checkpoint"]["relative_path"], "output/train/weights/best.pt")
        self.assertFalse(result["fixture"])
        self.assertEqual(result["epochs_completed"], 2)
        from foxtrot.persistence.files import fingerprint
        self.assertEqual(fingerprint(self.job / result["checkpoint"]["relative_path"])[0],
                         result["checkpoint"]["sha256"])
        # Ultralytics' own logging stayed off and everything stayed inside the job folder.
        settings = json.loads((self.job / "output" / ".ultralytics" / "Ultralytics" / "settings.json").read_text())
        self.assertFalse(settings["mlflow"] or settings["sync"] or settings["dvc"])
        self.assertFalse((self.root / "runs").exists())
        # The label cache YOLO wrote into the dataset does not change its identity.
        self.assertEqual(load_dataset(self.workspace, self.checkout, "squares"), self.dataset)

    # A cancellation request stops training at the first batch, without a checkpoint.
    def test_cancellation_stops_training(self):
        outcome, _ = self.train(cancel=True, epochs=50)
        self.assertEqual((outcome["outcome"], outcome["progress"]), ("cancelled", []))
        self.assertFalse((self.job / "output" / "train" / "weights" / "best.pt").exists())

    # Pretrained mode without weights fails clearly, before Ultralytics is even loaded.
    def test_missing_pretrained_weights_fail_clearly(self):
        outcome, stderr = self.train(initialization="pretrained")
        self.assertEqual(outcome, {"outcome": "failed", "code": "weights_missing"})
        self.assertIn("ultralytics imported: False", stderr)

    # Importing the adapter module stays light: no Ultralytics or PyTorch until training starts.
    def test_import_is_lightweight(self):
        code = ("import sys, foxtrot.pipeline.ultralytics_adapter; "
                "assert not {'ultralytics', 'torch'} & sys.modules.keys()")
        subprocess.run([sys.executable, "-B", "-c", code], check=True, cwd=self.root)


if __name__ == "__main__":
    unittest.main()
