"""V1.5 real evaluator: scores the exact trained checkpoint on the validation images.

Training then evaluation run in one fresh process, as in a real job.
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
from foxtrot.pipeline.yolo_dataset import write_yolo_data_yaml

# Trains 1 epoch, optionally tampers with best.pt, evaluates it, and prints the outcome as JSON.
RUNNER = r"""
import json, sys
from pathlib import Path
from foxtrot.config.job import JobSpec
from foxtrot.pipeline.contracts import DatasetInput, EvaluationInput, PipelineFailure, TrainingInput
from foxtrot.pipeline.ultralytics_adapter import UltralyticsEvaluator, UltralyticsTrainer

args = json.loads(sys.argv[1])

class Reporter:
    def __init__(self):
        self.logs = []
    def progress(self, payload):
        pass
    def log(self, payload):
        self.logs.append(payload)
    def wait_cancelled(self, seconds=0.0):
        return False

reporter = Reporter()
dataset = DatasetInput.model_validate_json(args["dataset"])
training = UltralyticsTrainer(initialization="scratch", workers=0).train(TrainingInput(
    job_id="a" * 32, resolved=JobSpec.model_validate_json(args["resolved"]), resolved_sha256="b" * 64,
    dataset=dataset, data_yaml=args["data_yaml"], job_directory=args["job"],
    output_directory=args["job"] + "/output"), reporter)
if args["tamper"]:
    with open(Path(args["job"]) / training.checkpoint.relative_path, "ab") as stream:
        stream.write(b"changed")
try:
    evaluation = UltralyticsEvaluator().evaluate(EvaluationInput(
        job_id="a" * 32, checkpoint=training.checkpoint, job_directory=args["job"], dataset=dataset), reporter)
    print(json.dumps({"outcome": "finished", "training": training.model_dump(mode="json"),
                      "evaluation": evaluation.model_dump(mode="json"), "logs": reporter.logs}))
except PipelineFailure as failure:
    print(json.dumps({"outcome": "failed", "code": failure.code}))
"""


# Tests for UltralyticsEvaluator using a tiny generated dataset of coloured squares.
class UltralyticsEvaluatorTests(unittest.TestCase):
    # Before each test: build and import a 12-image dataset, and make a job folder with data.yaml.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-evaluate-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "checkout").mkdir()
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
        self.dataset, _ = import_dataset(self.workspace, self.root / "checkout", source=source,
                                         dataset_id="squares", class_names=("Red", "Orange"))
        self.job = self.workspace / "jobs" / ("d" * 32)
        (self.job / "output").mkdir(parents=True)
        self.data_yaml = write_yolo_data_yaml(self.dataset, self.job / "output" / "data.yaml")

    # Helper: run training + evaluation in a fresh process and return the JSON outcome.
    def run_job(self, *, tamper=False):
        resolved = json.dumps({"dataset_id": "squares", "tracking_destination_id": "local",
                               "training": {"epochs": 1, "batch_size": 4, "image_size": 64, "seed": 1}})
        args = {"resolved": resolved, "dataset": self.dataset.model_dump_json(),
                "data_yaml": str(self.data_yaml), "job": str(self.job), "tamper": tamper}
        result = subprocess.run([sys.executable, "-B", "-c", RUNNER, json.dumps(args)], cwd=self.root,
                                capture_output=True, text=True, timeout=600,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2"))
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    # The evaluation names the exact checkpoint and dataset and has scores between 0 and 1.
    def test_scores_the_exact_checkpoint_on_the_validation_set(self):
        outcome = self.run_job()
        self.assertEqual(outcome["outcome"], "finished")
        evaluation = outcome["evaluation"]
        self.assertEqual(evaluation["checkpoint"], outcome["training"]["checkpoint"])
        self.assertEqual((evaluation["dataset_id"], evaluation["dataset_sha256"]),
                         ("squares", self.dataset.metadata.content_sha256))
        for key in ("precision", "recall", "map50", "map50_95"):
            self.assertTrue(0.0 <= evaluation[key] <= 1.0, key)
        per_class = [log for log in outcome["logs"] if log.get("action") == "evaluation_complete"][0]
        self.assertEqual(set(per_class["per_class_map50_95"]), {"Red", "Orange"})
        self.assertTrue((self.job / "output" / "evaluation").is_dir())
        self.assertFalse((self.root / "runs").exists())

    # A checkpoint changed after training is refused instead of scored.
    def test_changed_checkpoint_is_not_scored(self):
        self.assertEqual(self.run_job(tamper=True), {"outcome": "failed", "code": "checkpoint_mismatch"})


if __name__ == "__main__":
    unittest.main()
