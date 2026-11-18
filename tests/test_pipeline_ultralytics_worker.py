"""V1.6a: the worker runs a queued job through real YOLO training and evaluation."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from foxtrot.config.result import JobResult
from foxtrot.execution.worker import Worker
from foxtrot.persistence.artifacts import get_job_artifact
from foxtrot.persistence.dataset_import import import_dataset
from foxtrot.persistence.events import read_job_events
from foxtrot.persistence.jobs import create_recorded_job, get_job_record
from foxtrot.persistence.tracking import list_tracking
from foxtrot.pipeline.registry import PipelineSettings, get_adapters


# End-to-end: queue a job, let the real worker train and evaluate it, check every record.
class UltralyticsWorkerTests(unittest.TestCase):
    # Before each test: import a tiny 12-image dataset into a fresh workspace.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-worker-yolo-")
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

    # Helper: queue a job and process it with a real-YOLO worker (random starting weights by default).
    def run_job(self, initialization="scratch", **training):
        job = create_recorded_job(self.workspace, self.checkout, {
            "dataset_id": "squares", "tracking_destination_id": "local",
            "training": {"epochs": 2, "batch_size": 4, "image_size": 64, "seed": 1, **training}}, "job-1").name
        settings = PipelineSettings(adapter="ultralytics", ultralytics_initialization=initialization,
                                    ultralytics_workers=0)
        Worker(self.workspace, self.checkout, pipeline=settings,
               runtime_directory=self.root / "runtime").run(once=True)
        return job, get_job_record(self.workspace, self.checkout, job)

    # A real job finishes with a real (non-fixture) result, epoch progress and a queued upload with metrics.
    def test_real_job_finishes_with_verified_result(self):
        job, record = self.run_job()
        self.assertEqual((record.execution_state, record.exit_code), ("finished", 0), record.error_details)
        directory = self.workspace / "jobs" / job
        result = JobResult.model_validate_json((directory / "result.json").read_text())
        self.assertEqual(result.checkpoint.relative_path, "output/train/weights/best.pt")
        self.assertEqual(result.evaluation.dataset_sha256, self.dataset.metadata.content_sha256)
        checkpoint = get_job_artifact(self.workspace, self.checkout, job, "checkpoint")
        self.assertEqual(checkpoint.sha256, result.checkpoint.sha256)
        self.assertFalse(json.loads((directory / "summary.json").read_text())["fixture"])
        events = read_job_events(self.workspace, self.checkout, job)
        self.assertEqual([e.payload["epoch"] for e in events if e.event_type == "progress"], [1, 2])
        self.assertEqual([e.payload["state"] for e in events if e.event_type == "state_changed"],
                         ["queued", "preparing", "training", "evaluating", "finished"])
        plan = [entry.operation for entry in list_tracking(self.workspace, self.checkout, job)]
        self.assertEqual(plan[0].payload["tags"]["foxtrot.fixture"], "false")
        metrics = next(op for op in plan if op.operation == "log_metrics").payload["metrics"]
        self.assertIn("eval/mAP50-95", {point["key"] for point in metrics})

    # Pretrained mode without installed weights fails the job with that exact, readable reason.
    def test_missing_weights_fail_job_clearly(self):
        job, record = self.run_job(initialization="pretrained")
        self.assertEqual((record.execution_state, record.error_details.code), ("failed", "weights_missing"))
        self.assertIn("yolov8n.pt", record.error_details.message)
        result = JobResult.model_validate_json((self.workspace / "jobs" / job / "result.json").read_text())
        self.assertIsNone(result.checkpoint)

    # The registry builds real adapters only when the trusted settings ask for them.
    def test_registry_selects_real_adapters(self):
        adapters = get_adapters(PipelineSettings(adapter="ultralytics"))
        self.assertFalse(adapters.fixture)
        self.assertEqual(type(adapters.trainer).__name__, "UltralyticsTrainer")
        self.assertTrue(get_adapters(PipelineSettings(adapter="fixture")).fixture)


if __name__ == "__main__":
    unittest.main()
