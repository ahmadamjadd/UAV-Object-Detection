"""H6 acceptance: the real worker/supervisor calls an adapter and records its result."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import yaml

from foxtrot.config.result import JobResult
from foxtrot.execution.outcomes import accept_pipeline_result
from foxtrot.execution.worker import Worker
from foxtrot.persistence.artifacts import get_job_artifact
from foxtrot.persistence.commands import request_cancel
from foxtrot.persistence.events import read_job_events
from foxtrot.persistence.jobs import create_recorded_job, get_job_record
from foxtrot.persistence.tracking import list_tracking
from foxtrot.pipeline.registry import PipelineSettings
from foxtrot.tracking.fixture import RecordingTrackingClient
from foxtrot.tracking.uploader import sync_job_tracking

from test_pipeline_contracts import make_dataset


class PipelineWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-h6-worker-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"

    def submit(self, dataset_id="shapes", epochs=4, key="one"):
        return create_recorded_job(self.workspace, self.checkout, {
            "dataset_id": dataset_id, "tracking_destination_id": "team", "training": {"epochs": epochs},
        }, key).name

    def worker(self, **settings):
        settings = {"adapter": "fixture", "fixture_epoch_seconds": 0.0, **settings}
        return Worker(self.workspace, self.checkout, pipeline=PipelineSettings(**settings),
                      runtime_directory=self.root / "runtime")

    def record(self, job):
        return get_job_record(self.workspace, self.checkout, job)

    def operations(self, job):
        return [entry.operation for entry in list_tracking(self.workspace, self.checkout, job)]

    def test_adapter_receives_intended_configuration_and_returns_identifiable_result(self):
        dataset = make_dataset(self.workspace, self.checkout)
        job = self.submit()
        self.assertEqual(self.worker().run(once=True), 1)
        record = self.record(job)
        self.assertEqual((record.execution_state, record.exit_code, record.tracking_state), ("finished", 0, "pending"))
        directory = self.workspace / "jobs" / job
        # The per-job dataset config uses the dataset's own classes and location.
        config = yaml.safe_load((directory / "output" / "data.yaml").read_text())
        self.assertEqual(config["names"], {0: "Circle", 1: "Square"})
        self.assertEqual(config["path"], dataset.root)
        self.assertFalse((self.checkout / "data.yaml").exists())
        # The H2 result identifies this job, its exact checkpoint, and the dataset.
        result = JobResult.model_validate_json((directory / "result.json").read_text())
        self.assertEqual((result.job_id, result.outcome), (job, "finished"))
        self.assertEqual(result.evaluation.dataset_sha256, dataset.metadata.content_sha256)
        checkpoint = get_job_artifact(self.workspace, self.checkout, job, "checkpoint")
        self.assertEqual((checkpoint.kind, checkpoint.sha256), ("checkpoint", result.checkpoint.sha256))
        content = json.loads((directory / checkpoint.relative_path).read_text())
        self.assertEqual((content["job_id"], content["resolved_sha256"]), (job, record.resolved_sha256))
        events = read_job_events(self.workspace, self.checkout, job)
        self.assertEqual([e.payload["epoch"] for e in events if e.event_type == "progress"], [1, 2, 3, 4])
        self.assertEqual([e.payload["state"] for e in events if e.event_type == "state_changed"],
                         ["queued", "preparing", "training", "evaluating", "finished"])
        kinds = [o.operation for o in self.operations(job)]
        self.assertEqual(kinds[:3], ["create_run", "log_params", "log_metrics"])
        self.assertEqual(kinds[-1], "set_terminated")
        params = self.operations(job)[1].payload["params"]
        self.assertEqual(params["training.epochs"], "4")
        self.assertEqual(params["dataset.content_sha256"], dataset.metadata.content_sha256)
        # Upload is a separate step; afterwards the job links to its exact run.
        client = RecordingTrackingClient()
        self.assertEqual(sync_job_tracking(self.workspace, self.checkout, job, client).tracking_state, "synced")
        (run,) = client.runs.values()
        self.assertEqual(run["status"], "FINISHED")
        self.assertIn(("weights", "best.pt", checkpoint.sha256), run["artifacts"])
        self.assertNotIn("logs", {path for path, *_ in run["artifacts"]})

    def test_missing_or_changed_dataset_fails_without_launching(self):
        job = self.submit(dataset_id="unregistered")
        with patch("foxtrot.execution.supervisor.subprocess.Popen") as spawn:
            self.worker().run(once=True)
            spawn.assert_not_called()
        record = self.record(job)
        self.assertEqual((record.execution_state, record.error_details.code), ("failed", "dataset_unavailable"))
        self.assertEqual(self.operations(job), [])
        dataset = make_dataset(self.workspace, self.checkout)
        (Path(dataset.root) / "val" / "labels" / "a.txt").write_text("1 0.1 0.1 0.1 0.1\n")
        changed = self.submit(key="two")
        with patch("foxtrot.execution.supervisor.subprocess.Popen") as spawn:
            self.worker().run(once=True)
            spawn.assert_not_called()
        self.assertEqual(self.record(changed).error_details.code, "dataset_unavailable")

    def test_adapter_failure_is_recorded_with_its_reason_and_queued_as_failed(self):
        make_dataset(self.workspace, self.checkout)
        job = self.submit()
        self.worker(fixture_fail_at_epoch=2).run(once=True)
        record = self.record(job)
        self.assertEqual((record.execution_state, record.exit_code), ("failed", 1))
        self.assertEqual(record.error_details.code, "fixture_training_failed")
        result = JobResult.model_validate_json((self.workspace / "jobs" / job / "result.json").read_text())
        self.assertIsNone(result.checkpoint)
        self.assertIn("epoch 2", result.error_message)
        operations = self.operations(job)
        self.assertNotIn("log_metrics", [o.operation for o in operations])
        self.assertEqual(operations[-1].payload, {"status": "FAILED"})

    def test_cancellation_reaches_the_adapter_and_is_queued_as_killed(self):
        make_dataset(self.workspace, self.checkout)
        job = self.submit(epochs=200)
        worker = self.worker(fixture_epoch_seconds=0.05)
        thread = threading.Thread(target=worker.run, kwargs={"once": True})
        thread.start()
        deadline = time.monotonic() + 15
        while self.record(job).execution_state != "training":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.02)
        request_cancel(self.workspace, self.checkout, job)
        thread.join(timeout=20)
        self.assertFalse(thread.is_alive())
        record = self.record(job)
        self.assertEqual(record.execution_state, "cancelled")
        self.assertEqual(self.operations(job)[-1].payload, {"status": "KILLED"})
        # The adapter stopped gracefully: no TERM/KILL escalation was needed.
        events = read_job_events(self.workspace, self.checkout, job)
        self.assertFalse([e for e in events if e.payload.get("action") == "termination_signal"])
        self.assertEqual(record.exit_code, 130)

    def test_supervisor_rejects_a_report_that_does_not_match_its_assignment(self):
        make_dataset(self.workspace, self.checkout)
        job = self.submit()
        self.worker().run(once=True)
        directory = self.workspace / "jobs" / job
        report = directory / "output" / "pipeline-result.json"
        value = json.loads(report.read_text())
        claim = type("Claim", (), {"job_id": job, "directory": directory})()
        from foxtrot.execution.protocol import PipelineExecutionRequest
        request = PipelineExecutionRequest.model_validate_json((directory / "execution.json").read_bytes())
        for path, bad in ((("job_id",), "f" * 32), (("evaluation", "dataset_sha256"), "0" * 64)):
            altered = json.loads(json.dumps(value))
            target = altered
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = bad
            report.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                accept_pipeline_result(self.workspace, self.checkout, claim, request)

    def test_worker_requires_exactly_one_trusted_runner(self):
        from foxtrot.execution.protocol import FixtureSettings
        with self.assertRaises(TypeError):
            Worker(self.workspace, self.checkout)
        with self.assertRaises(TypeError):
            Worker(self.workspace, self.checkout, fixture=FixtureSettings(),
                   pipeline=PipelineSettings(adapter="fixture"))


if __name__ == "__main__":
    unittest.main()
