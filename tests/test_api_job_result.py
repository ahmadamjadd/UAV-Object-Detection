"""V1.9b the job details include the official result once the job has finished."""

import json
from pathlib import Path
import tempfile
import unittest

from foxtrot.api.app import create_app
from foxtrot.execution.worker import Worker
from foxtrot.persistence.jobs import create_recorded_job
from foxtrot.pipeline.registry import PipelineSettings

from test_api import PORT, TOKEN, call
from test_pipeline_contracts import make_dataset


# Tests for the "result" field of GET /api/jobs/{id}.
class JobResultApiTests(unittest.TestCase):
    # Before each test: an app over a workspace with the small test dataset.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-job-result-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        make_dataset(self.workspace, self.checkout)
        self.app = create_app(self.workspace, self.checkout, port=PORT, session_token=TOKEN, static_directory=None)

    # Helper: GET one job's details.
    def detail(self, job):
        return json.loads(call(self.app, "GET", f"/api/jobs/{job}")[1])

    # Before running there is no result; after finishing the official scores are included.
    def test_result_appears_when_job_finishes(self):
        job = create_recorded_job(self.workspace, self.checkout, {
            "dataset_id": "shapes", "tracking_destination_id": "local", "training": {"epochs": 2}}, "j-1").name
        self.assertIsNone(self.detail(job)["result"])
        Worker(self.workspace, self.checkout, runtime_directory=self.root / "runtime",
               pipeline=PipelineSettings(adapter="fixture", fixture_epoch_seconds=0.0)).run(once=True)
        result = self.detail(job)["result"]
        self.assertEqual(result["outcome"], "finished")
        self.assertEqual(result["evaluation"]["map50_95"], 0.25)


if __name__ == "__main__":
    unittest.main()
