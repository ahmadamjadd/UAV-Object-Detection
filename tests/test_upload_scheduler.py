"""V1.9a automatic uploads: background sending, back-off after failures, Retry, one uploader at a time.

Uses a real local MLflow database (no internet).
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from foxtrot.api.app import create_app
from foxtrot.api.settings import DestinationUpdate, SettingsStore
from foxtrot.api.uploads import UploadScheduler
from foxtrot.execution.locking import worker_lock
from foxtrot.execution.worker import Worker
from foxtrot.persistence.jobs import create_recorded_job, get_job_record
from foxtrot.pipeline.registry import PipelineSettings

from test_api import PORT, TOKEN, call, guarded
from test_pipeline_contracts import make_dataset


# Tests for UploadScheduler, the sync lock and the Retry endpoint.
class UploadSchedulerTests(unittest.TestCase):
    # Before each test: a workspace with a dataset and a working "local-mlflow" destination.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-uploads-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        make_dataset(self.workspace, self.checkout)
        (self.root / "mlflow").mkdir()
        SettingsStore(self.workspace, self.checkout).save_destination(DestinationUpdate(
            destination_id="local-mlflow", kind="local",
            tracking_uri=f"sqlite:///{self.root / 'mlflow' / 'mlflow.db'}"))

    # Helper: run one quick fixture job that reports to the given destination.
    def finished_job(self, destination="local-mlflow"):
        job = create_recorded_job(self.workspace, self.checkout, {
            "dataset_id": "shapes", "tracking_destination_id": destination, "training": {"epochs": 2}}, "j-1").name
        Worker(self.workspace, self.checkout, runtime_directory=self.root / "runtime",
               pipeline=PipelineSettings(adapter="fixture", fixture_epoch_seconds=0.0)).run(once=True)
        return job

    # Helper: wait until a condition is true, failing after a timeout.
    def wait_for(self, condition, timeout=90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.2)
        self.fail("Timed out waiting")

    # Helper: the job's upload state.
    def upload_state(self, job):
        return get_job_record(self.workspace, self.checkout, job).tracking_state

    # Once started, the scheduler uploads finished jobs by itself.
    def test_finished_job_is_uploaded_automatically(self):
        job = self.finished_job()
        scheduler = UploadScheduler(self.workspace, self.checkout, interval=0.2)
        scheduler.start()
        self.addCleanup(scheduler.stop)
        # The database says "synced" a moment before the scheduler has read the upload report.
        self.wait_for(lambda: self.upload_state(job) == "synced" and scheduler.status()["last_results"])
        self.assertEqual(scheduler.status()["last_results"][0]["tracking_state"], "synced")

    # Failed uploads wait longer each time (up to the maximum); Retry goes back to the normal pace.
    def test_failures_back_off_and_retry_resets(self):
        job = self.finished_job(destination="not-configured")
        scheduler = UploadScheduler(self.workspace, self.checkout, interval=0.2, max_interval=1.0)
        scheduler.start()
        self.addCleanup(scheduler.stop)
        self.wait_for(lambda: scheduler.delay == 1.0)
        self.assertEqual(self.upload_state(job), "retrying")  # V7: still queued, now with its reason shown
        scheduler.trigger()
        self.assertEqual(scheduler.delay, 0.2)

    # While one upload runs, a second sync command refuses to start (exit code 3, "busy").
    def test_only_one_uploader_at_a_time(self):
        job = self.finished_job()
        with worker_lock(self.workspace / "settings" / "uploader" / "uploader.lock"):
            result = subprocess.run(
                [sys.executable, "-B", "-m", "foxtrot.tracking.sync", "--workspace", str(self.workspace),
                 "--source-checkout", str(self.checkout), "run"],
                cwd=self.root, capture_output=True, text=True, timeout=120, env=dict(os.environ))
        self.assertEqual((result.returncode, json.loads(result.stdout)), (3, {"busy": True}))
        self.assertEqual(self.upload_state(job), "pending")

    # The Retry endpoint uploads now; with nothing pending it says so; it needs the session.
    def test_retry_endpoint(self):
        job = self.finished_job()
        app = create_app(self.workspace, self.checkout, port=PORT, session_token=TOKEN, static_directory=None)
        path = f"/api/jobs/{job}/retry-sync"
        self.assertEqual(call(app, "POST", path)[0], 403)
        status, content, _ = call(app, "POST", path, headers=guarded())
        self.assertEqual((status, json.loads(content)), (200, {"queued": True}))
        self.wait_for(lambda: self.upload_state(job) == "synced")
        self.assertEqual(call(app, "POST", path, headers=guarded())[0], 409)
        status, content, _ = call(app, "GET", "/api/uploads")
        self.assertFalse(json.loads(content)["running"])  # the launcher starts the background loop


if __name__ == "__main__":
    unittest.main()
