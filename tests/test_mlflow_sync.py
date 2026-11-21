"""V1.8a sending a job's results to a real MLflow (a local MLflow database, no internet).

The sync command and the read-back both run in their own processes, as in real use.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from foxtrot.api.settings import DestinationUpdate, SettingsStore
from foxtrot.execution.worker import Worker
from foxtrot.persistence.jobs import create_recorded_job, get_job_record
from foxtrot.persistence.tracking import get_run_association, jobs_with_pending_tracking
from foxtrot.pipeline.registry import PipelineSettings

from test_pipeline_contracts import make_dataset

# Reads one run back from MLflow and prints what it contains as JSON.
READ_BACK = r"""
import json, os, sys
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri=sys.argv[1])
experiment = client.get_experiment_by_name("Foxtrot Detection")
runs = client.search_runs([experiment.experiment_id])
run = runs[0]
print(json.dumps({
    "count": len(runs), "run_id": run.info.run_id, "status": run.info.status, "name": run.info.run_name,
    "tags": run.data.tags, "params": run.data.params,
    "loss_steps": [m.step for m in client.get_metric_history(run.info.run_id, "train/loss")],
    "eval_map": [(m.step, m.value) for m in client.get_metric_history(run.info.run_id, "eval/mAP50-95")],
    "artifacts": sorted(f.path for d in ("weights", "results") for f in client.list_artifacts(run.info.run_id, d)),
}))
"""


# End-to-end: a finished job's queued upload is sent to MLflow and can be read back.
class MlflowSyncTests(unittest.TestCase):
    # Before each test: a workspace with a dataset and a "local-mlflow" destination (a database file).
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-mlflow-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        make_dataset(self.workspace, self.checkout)
        self.tracking_uri = f"sqlite:///{self.root / 'mlflow' / 'mlflow.db'}"
        (self.root / "mlflow").mkdir()
        SettingsStore(self.workspace, self.checkout).save_destination(DestinationUpdate(
            destination_id="local-mlflow", kind="local", tracking_uri=self.tracking_uri))

    # Helper: run a 3-epoch fixture job that reports to the given destination.
    def finished_job(self, destination="local-mlflow", key="job-1"):
        job = create_recorded_job(self.workspace, self.checkout, {
            "dataset_id": "shapes", "tracking_destination_id": destination, "training": {"epochs": 3}}, key).name
        Worker(self.workspace, self.checkout, runtime_directory=self.root / "runtime",
               pipeline=PipelineSettings(adapter="fixture", fixture_epoch_seconds=0.0)).run(once=True)
        return job

    # Helper: run the sync command in its own process and return its JSON output.
    def sync(self, *args):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "foxtrot.tracking.sync", "--workspace", str(self.workspace),
             "--source-checkout", str(self.checkout), *args],
            cwd=self.root, capture_output=True, text=True, timeout=300,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        return result.returncode, json.loads(result.stdout.strip().splitlines()[-1]), result.stderr

    # Helper: read the run back from MLflow in a separate process.
    def read_back(self):
        result = subprocess.run([sys.executable, "-B", "-c", READ_BACK, self.tracking_uri], cwd=self.root,
                                capture_output=True, text=True, timeout=120, check=True)
        return json.loads(result.stdout.strip().splitlines()[-1])

    # Everything queued for a finished job reaches MLflow: tags, settings, curves, files and status.
    def test_finished_job_is_uploaded_completely(self):
        job = self.finished_job()
        code, results, stderr = self.sync("run")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(results, [{"job_id": job, "tracking_state": "synced", "error": None}])
        run = self.read_back()
        self.assertEqual((run["count"], run["status"], run["name"]), (1, "FINISHED", f"foxtrot-{job[:8]}"))
        self.assertEqual(run["tags"]["foxtrot.job_id"], job)
        self.assertEqual(run["params"]["training.epochs"], "3")
        self.assertEqual(run["loss_steps"], [1, 2, 3])
        self.assertEqual(run["eval_map"], [[3, 0.25]])
        self.assertIn("weights/best.pt", run["artifacts"])
        self.assertIn("results/result.json", run["artifacts"])
        association = get_run_association(self.workspace, self.checkout, job)
        self.assertEqual(association.run_id, run["run_id"])
        self.assertEqual(get_job_record(self.workspace, self.checkout, job).tracking_state, "synced")

    # Syncing again sends nothing and never creates a second run.
    def test_second_sync_changes_nothing(self):
        self.finished_job()
        self.sync("run")
        code, results, _ = self.sync("run")
        self.assertEqual((code, results), (0, []))
        self.assertEqual(self.read_back()["count"], 1)

    # A job whose destination is not configured stays queued, with the reason shown on the job (V7).
    def test_unconfigured_destination_stays_pending(self):
        from foxtrot.persistence.jobs import get_job_record
        job = self.finished_job(destination="dagshub")
        code, results, _ = self.sync("run")
        self.assertEqual(code, 0)
        self.assertEqual(results[0]["tracking_state"], "retrying")
        self.assertIn("not configured", results[0]["skipped"])
        self.assertEqual(get_job_record(self.workspace, self.checkout, job).tracking_error.code, "tracking_not_configured")
        self.assertEqual(jobs_with_pending_tracking(self.workspace, self.checkout), [job])

    # "check" succeeds for a reachable destination and explains failures without leaking tokens.
    def test_check_destination(self):
        code, result, _ = self.sync("check", "local-mlflow")
        self.assertEqual((code, result["ok"]), (0, True))
        SettingsStore(self.workspace, self.checkout).save_destination(DestinationUpdate(
            destination_id="broken", kind="mlflow", tracking_uri="http://127.0.0.1:9",
            username="me", token="secret-token-123"))
        code, result, _ = self.sync("check", "broken")
        self.assertEqual((code, result["ok"]), (1, False))
        self.assertNotIn("secret-token-123", result["message"])
        code, result, _ = self.sync("check", "missing")
        self.assertIn("not configured", result["message"])


if __name__ == "__main__":
    unittest.main()
