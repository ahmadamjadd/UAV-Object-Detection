"""V3: model registry, start modes, expanded training settings, and the coverage report."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw
from pydantic import ValidationError
import yaml

from foxtrot.config.job import JobSpec
from foxtrot.config.models import MODEL_IDS, MODELS, get_model
from foxtrot.config.training import GROUPS, TrainingSettings, ultralytics_arguments
from foxtrot.pipeline.coverage import (
    EXPOSED, classify, coverage_report, exposed_keys, ultralytics_default_keys,
)


# Settings, registry and coverage rules that need no training.
class SettingsRulesTests(unittest.TestCase):
    # 4 families x 5 sizes, yolov8n first; every model points at its official weights and architecture.
    def test_model_registry(self):
        self.assertEqual((len(MODEL_IDS), MODEL_IDS[0]), (20, "yolov8n"))
        self.assertEqual(get_model("yolo26x").display_name, "YOLO26 extra large")
        self.assertTrue(all(m.weights_url.endswith(f"/{m.model_id}.pt") and m.task == "detect" for m in MODELS.values()))
        with self.assertRaises(ValueError):
            get_model("yolov5n")

    # 'auto' picks its own learning rate; a named optimizer gets Ultralytics' defaults filled in.
    def test_optimizer_rules(self):
        self.assertIsNone(TrainingSettings().lr0)
        with self.assertRaises(ValidationError) as raised:
            TrainingSettings(optimizer="auto", lr0=0.02)
        self.assertIn("auto", str(raised.exception))
        sgd = TrainingSettings.model_validate({"optimizer": "SGD", "lr0": 0.02})
        self.assertEqual((sgd.lr0, sgd.momentum, sgd.warmup_bias_lr), (0.02, 0.937, 0.1))
        # The saved settings reload to exactly the same values.
        self.assertEqual(TrainingSettings.model_validate_json(sgd.model_dump_json()), sgd)

    # Limits are enforced before a job exists.
    def test_bad_values_are_rejected(self):
        for bad in ({"image_size": 100}, {"optimizer": "Lion"}, {"fliplr": 2}, {"workers": 99},
                    {"cache": "disk"}, {"device": "cuda"}, {"unknown": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                TrainingSettings.model_validate(bad)

    # Settings become Ultralytics arguments; "auto" leaves lr0 to the runtime.
    def test_ultralytics_arguments(self):
        default = ultralytics_arguments(TrainingSettings())
        self.assertEqual((default["imgsz"], default["batch"], default["cache"], default["freeze"]), (640, 16, False, None))
        self.assertNotIn("lr0", default)
        custom = ultralytics_arguments(TrainingSettings(optimizer="AdamW", lr0=0.003, cache="ram", freeze=3))
        self.assertEqual((custom["optimizer"], custom["lr0"], custom["cache"], custom["freeze"]), ("AdamW", 0.003, "ram", 3))

    # A checkpoint start needs exactly one earlier job; other modes must not name one.
    def test_start_mode_rules(self):
        base = {"dataset_id": "d", "tracking_destination_id": "t"}
        self.assertEqual(JobSpec(**base).initialization, "pretrained")
        with self.assertRaises(ValidationError):
            JobSpec(**base, initialization="checkpoint")
        with self.assertRaises(ValidationError):
            JobSpec(**base, initialization="scratch", checkpoint_job_id="a" * 32)
        self.assertEqual(JobSpec(**base, initialization="checkpoint", checkpoint_job_id="a" * 32).checkpoint_job_id, "a" * 32)

    # Every setting has a known group and a description, so the form can show it.
    def test_every_setting_is_described(self):
        for name, field in TrainingSettings.model_fields.items():
            extra = field.json_schema_extra or {}
            self.assertIn(extra.get("group"), GROUPS, name)
            self.assertTrue(field.description, name)

    # Every key of the pinned Ultralytics configuration is classified, and "exposed" keys are real.
    def test_coverage_classifies_every_runtime_key(self):
        keys = ultralytics_default_keys()
        self.assertGreater(len(keys), 50)
        unclassified = [key for key in keys if classify(key)[1].startswith("Not classified")]
        self.assertEqual(unclassified, [])
        self.assertTrue(set(exposed_keys()) <= set(keys), set(exposed_keys()) - set(keys))
        sent = set(ultralytics_arguments(TrainingSettings(optimizer="SGD")))
        self.assertEqual({key for key in keys if classify(key)[0] == EXPOSED} - sent, set())
        report = coverage_report(keys)
        self.assertEqual(len(report["runtime"]), len(keys))
        self.assertTrue(any(row["key"] == "n" and row["status"] == EXPOSED for row in report["generator"]))


# Real YOLO jobs through the worker: chosen model, settings reaching training, checkpoint start.
class TrainingJobTests(unittest.TestCase):
    # Before each test: a tiny 12-image dataset in a fresh workspace.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v3-training-")
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
        from foxtrot.persistence.dataset_import import import_dataset
        import_dataset(self.workspace, self.checkout, source=source, dataset_id="squares",
                       class_names=("Red", "Orange"))

    # Queue one job and let a real-YOLO worker run it, following each job's own start mode.
    def run_job(self, key: str, **request):
        from foxtrot.execution.worker import Worker
        from foxtrot.persistence.jobs import create_recorded_job, get_job_record
        from foxtrot.pipeline.registry import PipelineSettings

        spec = {"dataset_id": "squares", "tracking_destination_id": "local", **request}
        job = create_recorded_job(self.workspace, self.checkout, spec, key).name
        Worker(self.workspace, self.checkout, runtime_directory=self.root / "runtime",
               pipeline=PipelineSettings(adapter="ultralytics", ultralytics_workers=0)).run(once=True)
        return job, get_job_record(self.workspace, self.checkout, job)

    # The MLflow parameters queued for a job.
    def params(self, job: str) -> dict:
        from foxtrot.persistence.tracking import list_tracking
        operations = [entry.operation for entry in list_tracking(self.workspace, self.checkout, job)]
        return next(o for o in operations if o.operation == "log_params").payload["params"]

    # Settings changed in the job reach Ultralytics (its args.yaml) and MLflow; then a checkpoint start.
    def test_settings_reach_training_and_checkpoint_start_is_traceable(self):
        training = {"epochs": 1, "batch_size": 4, "image_size": 64, "seed": 1, "optimizer": "SGD", "lr0": 0.02,
                    "mosaic": 0.0, "close_mosaic": 0, "fliplr": 0.0, "patience": 5}
        first, record = self.run_job("first", model_id="yolo11n", initialization="scratch", training=training)
        self.assertEqual(record.execution_state, "finished", record.error_details)
        used = yaml.safe_load((self.workspace / "jobs" / first / "output" / "train" / "args.yaml").read_text())
        self.assertEqual((used["model"], used["optimizer"], used["lr0"], used["mosaic"], used["fliplr"],
                          used["patience"], used["pretrained"]), ("yolo11n.yaml", "SGD", 0.02, 0.0, 0.0, 5, False))
        params = self.params(first)
        self.assertEqual((params["model_id"], params["initialization"], params["training.optimizer"],
                          params["training.lr0"], params["model.initialization"]),
                         ("yolo11n", "scratch", "SGD", "0.02", "scratch"))
        self.assertNotIn("model.start_weights_sha256", params)  # random weights have no file

        from foxtrot.config.result import JobResult
        earlier = JobResult.model_validate_json((self.workspace / "jobs" / first / "result.json").read_text())
        second, record = self.run_job("second", model_id="yolo11n", initialization="checkpoint",
                                      checkpoint_job_id=first, training={"epochs": 1, "batch_size": 4, "image_size": 64})
        self.assertEqual(record.execution_state, "finished", record.error_details)
        params = self.params(second)
        self.assertEqual(params["model.initialization"], "checkpoint")
        self.assertEqual(params["model.start_weights_sha256"], earlier.checkpoint.sha256)
        self.assertEqual(params["checkpoint_job_id"], first)

        # Submission checks: another model, an unknown job, or an unfinished job are refused up front.
        from foxtrot.api.app import create_app
        from test_api import call, guarded
        app = create_app(self.workspace, self.checkout, port=8765, session_token="test-session-token",
                         static_directory=None)
        for model_id, job_id, message in (("yolov8n", first, "trained yolo11n"),
                                          ("yolo11n", "f" * 32, "Unknown job")):
            body = {"submission_key": f"check-{model_id}-{job_id[:4]}", "request": {
                "dataset_id": "squares", "tracking_destination_id": "local", "model_id": model_id,
                "initialization": "checkpoint", "checkpoint_job_id": job_id}}
            status, content, _ = call(app, "POST", "/api/jobs", body=body, headers=guarded())
            self.assertEqual(status, 422, content)
            self.assertIn(message, json.loads(content)["detail"])
        status, content, _ = call(app, "GET", "/api/checkpoints")
        self.assertEqual({c["job_id"] for c in json.loads(content)}, {first, second})


if __name__ == "__main__":
    unittest.main()
