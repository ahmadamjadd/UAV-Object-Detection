"""V4: DVC dataset versions, labeled real-image evaluation sets, evaluation jobs, previews.

Sample images (coloured squares) stand in for real drone photos.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw
from pydantic import ValidationError

from foxtrot.config.dataset import DatasetMetadata
from foxtrot.config.job import JobSpec
from foxtrot.config.lifecycle import ExecutionState, validate_execution_transition
from foxtrot.persistence.dataset_import import import_dataset, import_evaluation_dataset
from foxtrot.persistence.datasets import list_datasets, load_dataset

DVC = str(Path(sys.executable).with_name("dvc"))


# Write sample square images with YOLO labels: split layout (train/val) or flat (images/labels).
def write_squares(folder: Path, layout: dict[str, int], *, labels: bool = True, offset: int = 0) -> None:
    for subfolder, count in layout.items():
        images = folder / subfolder / "images" if subfolder else folder / "images"
        images.mkdir(parents=True)
        if labels:
            (images.parent / "labels").mkdir()
        for index in range(count):
            image = Image.new("RGB", (64, 64), (30 + offset, 30, 30))
            ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill=(200, 40 + index * 9 + offset, 50))
            image.save(images / f"{index}.jpg")
            if labels:
                (images.parent / "labels" / f"{index}.txt").write_text(f"{index % 2} 0.5 0.5 0.5 0.5\n")


# Rules that need no files.
class RulesTests(unittest.TestCase):
    # Old manifests default to "training"; test data may never be a tuning objective.
    def test_roles_and_tuning_guard(self):
        base = {"dataset_id": "d", "class_names": ("A",), "train_images_ref": "images",
                "validation_images_ref": "images", "content_sha256": "0" * 64}
        self.assertEqual(DatasetMetadata(**base).role, "training")
        self.assertTrue(DatasetMetadata(**base, role="validation").can_be_tuning_objective())
        self.assertFalse(DatasetMetadata(**base, role="test").can_be_tuning_objective())

    # An evaluation job must name the checkpoint it scores; it goes straight to evaluating.
    def test_evaluate_job_rules(self):
        base = {"dataset_id": "d", "tracking_destination_id": "t", "kind": "evaluate"}
        with self.assertRaises(ValidationError):
            JobSpec(**base)
        spec = JobSpec(**base, initialization="checkpoint", checkpoint_job_id="a" * 32)
        self.assertEqual(spec.kind, "evaluate")
        validate_execution_transition(ExecutionState.PREPARING, ExecutionState.EVALUATING)


# Workspace-based tests share a training dataset, a fixture worker and the run manager.
class WorkspaceCase(unittest.TestCase):
    # Before each test: a workspace with a 12-image training dataset of two classes.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v4-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        write_squares(self.root / "train-source", {"train": 8, "val": 4})
        import_dataset(self.workspace, self.checkout, source=self.root / "train-source", dataset_id="squares",
                       class_names=("Red", "Orange"))

    # Register a flat labeled folder as an evaluation dataset (optionally split by groups).
    def evaluation_set(self, dataset_id="real", count=6, **options):
        source = self.root / f"source-{dataset_id}"
        write_squares(source, {"": count}, offset=40)
        if options.pop("groups", False):
            (source / "groups.json").write_text(json.dumps({f"{i}.jpg": f"flight-{i % 3}" for i in range(count)}))
        return import_evaluation_dataset(self.workspace, self.checkout, source=source, dataset_id=dataset_id,
                                         class_names=options.pop("class_names", ("Red", "Orange")), **options)

    # Queue a job and process it with one worker pass (fixture adapter unless told otherwise).
    def run_job(self, key, adapter="fixture", **request):
        from foxtrot.execution.worker import Worker
        from foxtrot.persistence.jobs import create_recorded_job, get_job_record
        from foxtrot.pipeline.registry import PipelineSettings

        spec = {"dataset_id": "squares", "tracking_destination_id": "local", **request}
        job = create_recorded_job(self.workspace, self.checkout, spec, key).name
        settings = (PipelineSettings(adapter="fixture", fixture_epoch_seconds=0.0) if adapter == "fixture"
                    else PipelineSettings(adapter="ultralytics", ultralytics_workers=0))
        Worker(self.workspace, self.checkout, pipeline=settings, runtime_directory=self.root / "runtime").run(once=True)
        return job, get_job_record(self.workspace, self.checkout, job)

    # The upload operations queued for a job.
    def operations(self, job):
        from foxtrot.persistence.tracking import list_tracking
        return [entry.operation for entry in list_tracking(self.workspace, self.checkout, job)]

    # Start a background run (dvc / predict) and wait for its final status.
    def background(self, kind, request):
        from foxtrot.api.generation import GenerationRuns
        runs = GenerationRuns(self.workspace, self.checkout)
        self.addCleanup(runs.stop)
        status = runs.start(kind, request)
        runs._active[1].wait(timeout=300)
        return runs, runs.status(status["run_id"])


# Labeled real-image sets: roles, flight groups, and how jobs may use them.
class EvaluationDatasetTests(WorkspaceCase):
    # Groups stay whole when a set is divided into validation and test; labels are required.
    def test_group_split_and_required_labels(self):
        found, report = self.evaluation_set(count=9, groups=True, test_share=0.34)
        self.assertTrue(report.valid)
        validation, test = found
        self.assertEqual((validation.metadata.role, test.metadata.role), ("validation", "test"))
        groups = [set(json.loads((Path(d.root) / "groups.json").read_text()).values()) for d in found]
        self.assertFalse(groups[0] & groups[1])
        self.assertEqual(sum(len(list((Path(d.root) / "images").iterdir())) for d in found), 9)
        self.assertEqual(load_dataset(self.workspace, self.checkout, "real-test"), test)
        unlabeled = self.root / "unlabeled"
        write_squares(unlabeled, {"": 3}, labels=False)
        found, report = import_evaluation_dataset(self.workspace, self.checkout, source=unlabeled,
                                                  dataset_id="nolabels", class_names=("Red", "Orange"))
        self.assertEqual((found, report.valid), ([], False))

    # Real evaluation data cannot train; training data cannot be an evaluation set; classes must match.
    def test_role_and_class_rules(self):
        from foxtrot.persistence.start_checkpoints import StartCheckpointError, check_dataset_use

        self.evaluation_set("real")
        self.evaluation_set("other", class_names=("Hexagon", "Triangle"))
        job, record = self.run_job("trained")
        self.assertEqual(record.execution_state, "finished")
        evaluate = {"tracking_destination_id": "t", "kind": "evaluate", "initialization": "checkpoint",
                    "checkpoint_job_id": job}
        for spec, message in ((JobSpec(dataset_id="real", tracking_destination_id="t"), "can only score"),
                              (JobSpec(dataset_id="squares", **evaluate), "training dataset"),
                              (JobSpec(dataset_id="other", **evaluate), "Class mismatch")):
            with self.subTest(message=message), self.assertRaises(StartCheckpointError) as raised:
                check_dataset_use(self.workspace, self.checkout, spec)
            self.assertIn(message, str(raised.exception))
        check_dataset_use(self.workspace, self.checkout, JobSpec(dataset_id="real", **evaluate))
        # The worker refuses the same misuse even when a job bypasses the website.
        _, record = self.run_job("misuse", dataset_id="real")
        self.assertEqual((record.execution_state, record.error_details.code), ("failed", "dataset_role"))


# Evaluation jobs score an exact earlier checkpoint on a labeled set and upload with both identities.
class EvaluationJobTests(WorkspaceCase):
    # Fixture adapter: identities, MLflow plan (kind tag, fixed config, no second weights upload).
    def test_fixture_evaluation_job(self):
        from foxtrot.config.result import JobResult

        (real,), _ = self.evaluation_set("real", role="test")
        trained, _ = self.run_job("trained")
        earlier = JobResult.model_validate_json((self.workspace / "jobs" / trained / "result.json").read_text())
        job, record = self.run_job("evaluate", dataset_id="real", kind="evaluate", initialization="checkpoint",
                                   checkpoint_job_id=trained)
        self.assertEqual(record.execution_state, "finished", record.error_details)
        result = JobResult.model_validate_json((self.workspace / "jobs" / job / "result.json").read_text())
        self.assertEqual(result.checkpoint.sha256, earlier.checkpoint.sha256)
        self.assertEqual((result.evaluation.dataset_id, result.evaluation.dataset_sha256),
                         ("real", real.metadata.content_sha256))
        from foxtrot.persistence.events import read_job_events
        states = [e.payload["state"] for e in read_job_events(self.workspace, self.checkout, job)
                  if e.event_type == "state_changed"]
        self.assertEqual(states, ["queued", "preparing", "evaluating", "finished"])
        operations = self.operations(job)
        self.assertEqual(operations[0].payload["tags"]["foxtrot.kind"], "evaluate")
        params = next(o for o in operations if o.operation == "log_params").payload["params"]
        self.assertEqual((params["evaluation.config"], params["dataset.role"], params["model.start_weights_sha256"]),
                         ("foxtrot-eval-v1", "test", earlier.checkpoint.sha256))
        artifacts = {o.payload["name"] for o in operations if o.operation == "log_artifact"}
        self.assertNotIn("checkpoint", artifacts)

    # Real YOLO: evaluation saves plots and per-class scores; previews show boxes without a score.
    def test_real_evaluation_plots_and_prediction_preview(self):
        self.evaluation_set("real")
        trained, record = self.run_job("trained", adapter="ultralytics", initialization="scratch",
                                       training={"epochs": 1, "batch_size": 4, "image_size": 64})
        self.assertEqual(record.execution_state, "finished", record.error_details)
        job, record = self.run_job("evaluate", adapter="ultralytics", dataset_id="real", kind="evaluate",
                                   initialization="checkpoint", checkpoint_job_id=trained)
        self.assertEqual(record.execution_state, "finished", record.error_details)
        operations = self.operations(job)
        plots = [o.payload["name"] for o in operations
                 if o.operation == "log_artifact" and o.payload["artifact_path"] == "evaluation"]
        self.assertIn("plot-confusion_matrix", plots)
        metrics = {p["key"] for o in operations if o.operation == "log_metrics" for p in o.payload["metrics"]}
        self.assertTrue({"eval/mAP50-95", "eval/class/Red/mAP50-95", "eval/class/Orange/mAP50-95"} <= metrics)

        from foxtrot.pipeline.prediction import PredictionRequest
        photos = self.root / "photos"
        write_squares(photos, {"": 3}, labels=False)
        runs, status = self.background("predict", PredictionRequest(
            checkpoint_job_id=trained, images=str(photos / "images"), confidence=0.01))
        self.assertEqual(status["state"], "finished", status)
        preview = status["result"]
        self.assertIs(preview["measured"], False)
        self.assertEqual(len(preview["images"]), 3)
        self.assertTrue(runs.preview_image(status["run_id"], preview["images"][0]["image"]).is_file())


# DVC: the exact recorded version is downloaded into the workspace and traced to MLflow.
class DvcDatasetTests(WorkspaceCase):
    # A throwaway Git+DVC repository with a local remote and two versions of one dataset.
    def make_repository(self) -> Path:
        repository, remote = self.root / "repo", self.root / "remote"
        remote.mkdir()
        run = lambda *args: subprocess.run(args, cwd=repository, check=True, capture_output=True)
        repository.mkdir()
        run("git", "init", "-q")
        run("git", "config", "user.email", "test@example.com")
        run("git", "config", "user.name", "Test")
        run(DVC, "init", "-q")
        run(DVC, "remote", "add", "-d", "store", str(remote))
        write_squares(repository / "data" / "ds", {"train": 6, "val": 2})
        run(DVC, "add", "data/ds")
        run("git", "add", "-A")
        run("git", "commit", "-qm", "dataset v1")
        run("git", "tag", "v1")
        run(DVC, "push", "-q")
        shutil.rmtree(repository / "data" / "ds")
        write_squares(repository / "data" / "ds", {"train": 6, "val": 2}, offset=60)
        run(DVC, "add", "data/ds")
        run("git", "commit", "-qam", "dataset v2")
        run(DVC, "push", "-q")
        shutil.rmtree(repository / ".dvc" / "cache")  # force downloads from the remote
        return repository

    # Two revisions give two datasets with their own commits; a bad revision publishes nothing.
    def test_revisions_bad_input_and_mlflow_trace(self):
        from foxtrot.persistence.dvc_datasets import DvcImportRequest

        repository = self.make_repository()
        commit_v1 = subprocess.run(["git", "rev-parse", "v1"], cwd=repository, check=True,
                                   capture_output=True, text=True).stdout.strip()
        request = lambda dataset_id, revision: DvcImportRequest(
            repository=str(repository), revision=revision, path="data/ds", dataset_id=dataset_id,
            class_names=("Red", "Orange"))
        _, first = self.background("dvc", request("ds-v1", "v1"))
        self.assertEqual(first["state"], "finished", first)
        self.assertEqual(first["result"]["commit"], commit_v1)
        _, second = self.background("dvc", request("ds-v2", "HEAD"))
        self.assertEqual(second["state"], "finished", second)
        v1, v2 = (load_dataset(self.workspace, self.checkout, name) for name in ("ds-v1", "ds-v2"))
        self.assertNotEqual(v1.metadata.content_sha256, v2.metadata.content_sha256)
        _, bad = self.background("dvc", request("ds-bad", "no-such-tag"))
        self.assertEqual((bad["state"], bad["error"]["code"]), ("failed", "dvc_unavailable"))
        self.assertNotIn("ds-bad", {d.dataset_id for d in list_datasets(self.workspace, self.checkout)})
        self.assertEqual([p.name for p in (self.workspace / "datasets").iterdir() if p.name.startswith(".")], [])
        self.assertEqual(subprocess.run(["git", "status", "--short"], cwd=repository, capture_output=True,
                                        text=True).stdout, "")  # the source repository is unchanged
        job, record = self.run_job("dvc-train", dataset_id="ds-v1")
        self.assertEqual(record.execution_state, "finished", record.error_details)
        params = next(o for o in self.operations(job) if o.operation == "log_params").payload["params"]
        self.assertEqual((params["dataset.dvc.commit"], params["dataset.dvc.revision"]), (commit_v1, "v1"))


if __name__ == "__main__":
    unittest.main()
