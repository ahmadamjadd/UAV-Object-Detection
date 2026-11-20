"""H6 explicit-input pipeline contracts, registered datasets, and fixture adapters."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from pydantic import ValidationError
import yaml

from foxtrot.config.checkpoint import CheckpointReference
from foxtrot.config.job import JobSpec
from foxtrot.persistence.datasets import DatasetUnavailable, load_dataset, register_dataset
from foxtrot.pipeline.contracts import (
    EvaluationInput, GenerationInput, PipelineCancelled, PipelineFailure, PipelineResult,
    SplitPolicy, TrainingInput, TrainingResult,
)
from foxtrot.pipeline.fixture import FixtureEvaluator, FixtureTrainer
from foxtrot.pipeline.registry import PipelineSettings, get_adapters
from foxtrot.pipeline.yolo_dataset import write_yolo_data_yaml


class Reporter:
    def __init__(self):
        self.events = []
        self.cancelled = threading.Event()

    def progress(self, payload):
        self.events.append(("progress", payload))

    def log(self, payload):
        self.events.append(("log", payload))

    def wait_cancelled(self, seconds=0.0):
        return self.cancelled.wait(seconds) if seconds else self.cancelled.is_set()


def make_dataset(workspace: Path, checkout: Path, dataset_id="shapes", classes=("Circle", "Square")):
    """Lay out a tiny dataset in the workspace and register it."""
    from foxtrot.persistence.workspace import initialize_workspace
    root = initialize_workspace(workspace, checkout)
    data = root / "datasets" / dataset_id / "data"
    for split in ("train", "val"):
        (data / split / "images").mkdir(parents=True)
        (data / split / "labels").mkdir(parents=True)
        (data / split / "images" / "a.jpg").write_bytes(f"image-{split}".encode())
        (data / split / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return register_dataset(workspace, checkout, dataset_id=dataset_id, class_names=classes,
                            train_images_ref="train/images", validation_images_ref="val/images")


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-h6-pipeline-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"

    def test_generation_keeps_images_per_class_and_canvas_separate_from_training_size(self):
        value = GenerationInput(class_names=("Hexagon", "Triangle"), images_per_class=100,
                                canvas_size=320, seed=1, output_directory=str(self.root))
        self.assertEqual(value.images_per_class * len(value.class_names), 200)
        self.assertNotIn("image_size", GenerationInput.model_fields)
        with self.assertRaises(ValidationError):
            GenerationInput(class_names=("A",), images_per_class=1, canvas_size=1, seed=1, output_directory="relative")

    def test_split_policy_and_results_reject_inconsistent_values(self):
        SplitPolicy(train_ratio=0.8, validation_ratio=0.2)
        with self.assertRaises(ValidationError):
            SplitPolicy(train_ratio=0.8, validation_ratio=0.3)
        checkpoint = CheckpointReference(relative_path="output/weights/best.pt", sha256="a" * 64)
        with self.assertRaises(ValidationError):
            TrainingResult(checkpoint=checkpoint, epochs_completed=1, metrics={"loss": float("nan")})

    def test_dataset_registration_uses_its_own_classes_and_detects_changes(self):
        dataset = make_dataset(self.workspace, self.checkout)
        self.assertEqual(dataset.metadata.class_names, ("Circle", "Square"))
        self.assertEqual(load_dataset(self.workspace, self.checkout, "shapes"), dataset)
        with self.assertRaises(DatasetUnavailable):  # a manifest is written once only
            register_dataset(self.workspace, self.checkout, dataset_id="shapes", class_names=("X",),
                             train_images_ref="train/images", validation_images_ref="val/images")
        self.assertEqual(load_dataset(self.workspace, self.checkout, "shapes").metadata.class_names,
                         ("Circle", "Square"))
        (Path(dataset.root) / "train" / "images" / "a.jpg").write_bytes(b"edited")
        with self.assertRaisesRegex(DatasetUnavailable, "changed"):
            load_dataset(self.workspace, self.checkout, "shapes")

    def test_dataset_ids_paths_and_symlinks_are_refused(self):
        dataset = make_dataset(self.workspace, self.checkout)
        for bad in ("../shapes", "missing", "", "a/b"):
            with self.assertRaises(DatasetUnavailable):
                load_dataset(self.workspace, self.checkout, bad)
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"x")
        (Path(dataset.root) / "train" / "images" / "link.jpg").symlink_to(outside)
        with self.assertRaises(DatasetUnavailable):
            load_dataset(self.workspace, self.checkout, "shapes")

    def test_yolo_config_is_per_job_and_never_overwritten(self):
        dataset = make_dataset(self.workspace, self.checkout)
        path = self.root / "job-output" / "data.yaml"
        path.parent.mkdir()
        write_yolo_data_yaml(dataset, path)
        value = yaml.safe_load(path.read_text())
        self.assertEqual(value, {"path": dataset.root, "train": "train/images", "val": "val/images",
                                 "names": {0: "Circle", 1: "Square"}})
        with self.assertRaises(FileExistsError):
            write_yolo_data_yaml(dataset, path)
        self.assertFalse((self.checkout / "data.yaml").exists())

    def _training_input(self, dataset, epochs=3):
        job = self.root / "job"
        output = job / "output"
        output.mkdir(parents=True)
        data_yaml = write_yolo_data_yaml(dataset, output / "data.yaml")
        resolved = JobSpec(dataset_id="shapes", tracking_destination_id="local", training={"epochs": epochs})
        return TrainingInput(job_id="a" * 32, resolved=resolved, resolved_sha256="b" * 64, dataset=dataset,
                             data_yaml=str(data_yaml), job_directory=str(job), output_directory=str(output))

    def test_fixture_adapters_write_only_assigned_output_and_score_exact_checkpoint(self):
        dataset = make_dataset(self.workspace, self.checkout)
        request = self._training_input(dataset)
        reporter = Reporter()
        training = FixtureTrainer(epoch_seconds=0).train(request, reporter)
        self.assertEqual(training.checkpoint.relative_path, "output/weights/best.pt")
        self.assertEqual([p["epoch"] for kind, p in reporter.events if kind == "progress"], [1, 2, 3])
        self.assertTrue(training.fixture)
        evaluation = FixtureEvaluator().evaluate(EvaluationInput(
            job_id=request.job_id, checkpoint=training.checkpoint,
            job_directory=request.job_directory, dataset=dataset), reporter)
        self.assertEqual(evaluation.dataset_sha256, dataset.metadata.content_sha256)
        PipelineResult(job_id=request.job_id, adapter="fixture", training=training, evaluation=evaluation)
        other = training.checkpoint.model_copy(update={"sha256": "c" * 64})
        with self.assertRaises(ValidationError):
            PipelineResult(job_id=request.job_id, adapter="fixture", training=training,
                           evaluation=evaluation.model_copy(update={"checkpoint": other}))
        # A checkpoint changed after training is never scored.
        (Path(request.job_directory) / training.checkpoint.relative_path).write_text("tampered")
        with self.assertRaises(PipelineFailure):
            FixtureEvaluator().evaluate(EvaluationInput(
                job_id=request.job_id, checkpoint=training.checkpoint,
                job_directory=request.job_directory, dataset=dataset), reporter)

    def test_fixture_trainer_honours_failure_and_cancellation(self):
        dataset = make_dataset(self.workspace, self.checkout)
        request = self._training_input(dataset)
        with self.assertRaises(PipelineFailure) as failure:
            FixtureTrainer(epoch_seconds=0, fail_at_epoch=2).train(request, Reporter())
        self.assertEqual(failure.exception.code, "fixture_training_failed")
        reporter = Reporter()
        reporter.cancelled.set()
        with self.assertRaises(PipelineCancelled):
            FixtureTrainer(epoch_seconds=0.01).train(request, reporter)

    def test_adapter_selection_is_code_owned(self):
        self.assertTrue(get_adapters(PipelineSettings(adapter="fixture")).fixture)
        for bad in ({"adapter": "os.system"}, {"adapter": "fixture", "module": "x"}, {}):
            with self.assertRaises(ValidationError):
                PipelineSettings.model_validate(bad)

    def test_pipeline_and_tracking_imports_stay_lightweight(self):
        program = """
import sys
import foxtrot.pipeline.contracts, foxtrot.pipeline.registry, foxtrot.pipeline.yolo_dataset
import foxtrot.persistence.datasets, foxtrot.persistence.tracking
import foxtrot.tracking.plan, foxtrot.tracking.uploader, foxtrot.tracking.fixture
heavy = {'torch', 'ultralytics', 'mlflow', 'dotenv', 'DataSetGeneration', 'cv2'} & sys.modules.keys()
assert not heavy, heavy
"""
        subprocess.run([sys.executable, "-B", "-c", program], check=True, cwd=self.root,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


if __name__ == "__main__":
    unittest.main()
