"""V1.7 every epoch's metrics go into the upload plan, so MLflow can draw training curves."""

from pathlib import Path
import tempfile
import unittest

from foxtrot.execution.worker import Worker
from foxtrot.persistence.jobs import create_recorded_job
from foxtrot.persistence.tracking import list_tracking
from foxtrot.pipeline.registry import PipelineSettings
from foxtrot.tracking.plan import epoch_metric_points

from test_pipeline_contracts import make_dataset


# Tests for turning saved epoch progress into MLflow metric points.
class EpochMetricPointsTests(unittest.TestCase):
    # Each numeric value becomes a point at step = epoch; the epoch counters themselves are not metrics.
    def test_progress_becomes_points_per_epoch(self):
        points = epoch_metric_points([
            {"epoch": 1, "epochs": 2, "metrics/mAP50-95": 0.4, "train/box_loss": 1.5},
            {"epoch": 2, "epochs": 2, "metrics/mAP50-95": 0.6, "train/box_loss": 1.1},
        ])
        self.assertEqual(points, [
            {"key": "metrics/mAP50-95", "value": 0.4, "step": 1}, {"key": "train/box_loss", "value": 1.5, "step": 1},
            {"key": "metrics/mAP50-95", "value": 0.6, "step": 2}, {"key": "train/box_loss", "value": 1.1, "step": 2},
        ])

    # Text, true/false, infinity, odd names and updates without an epoch number are skipped.
    def test_unusable_values_are_skipped(self):
        points = epoch_metric_points([
            {"epoch": 1, "note": "text", "flag": True, "bad": float("inf"), "weird(name)": 1.0, "ok": 2},
            {"step": 5, "total": 10},          # an H5 fixture update: not an epoch
            {"epoch": 0, "x": 1.0},            # epochs start at 1
        ])
        self.assertEqual(points, [{"key": "ok", "value": 2.0, "step": 1}])


# End-to-end with the fast fixture adapter: the worker queues per-epoch metrics for upload.
class EpochMetricsPlanTests(unittest.TestCase):
    # Before each test: a workspace with the small test dataset.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-epoch-metrics-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        make_dataset(self.workspace, self.checkout)

    # Helper: run one 4-epoch fixture job and return its queued metric points.
    def metrics_for_job(self, **settings):
        job = create_recorded_job(self.workspace, self.checkout, {
            "dataset_id": "shapes", "tracking_destination_id": "local", "training": {"epochs": 4}}, "job-1").name
        Worker(self.workspace, self.checkout, runtime_directory=self.root / "runtime",
               pipeline=PipelineSettings(adapter="fixture", fixture_epoch_seconds=0.0, **settings)).run(once=True)
        operations = [entry.operation for entry in list_tracking(self.workspace, self.checkout, job)]
        return next(op for op in operations if op.operation == "log_metrics").payload["metrics"]

    # A finished job uploads one point per epoch, plus the official evaluation once at the last epoch.
    def test_finished_job_uploads_curves_and_final_evaluation(self):
        metrics = self.metrics_for_job()
        loss = [(p["step"], p["value"]) for p in metrics if p["key"] == "train/loss"]
        self.assertEqual([step for step, _ in loss], [1, 2, 3, 4])
        self.assertEqual(loss[0][1], 1.0)  # the fixture reports loss 1/epoch
        evaluation = [p for p in metrics if p["key"] == "eval/mAP50-95"]
        self.assertEqual([(p["step"], p["value"]) for p in evaluation], [(4, 0.25)])

    # A job that fails at epoch 3 still uploads epochs 1–3, without an evaluation.
    def test_failed_job_uploads_completed_epochs(self):
        metrics = self.metrics_for_job(fixture_fail_at_epoch=3)
        self.assertEqual(sorted({p["step"] for p in metrics}), [1, 2, 3])
        self.assertFalse([p for p in metrics if p["key"].startswith("eval/")])


if __name__ == "__main__":
    unittest.main()
