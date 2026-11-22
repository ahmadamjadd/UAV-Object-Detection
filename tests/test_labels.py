"""Member-chosen names for datasets and runs: stored, renamed, shown by the API, used in MLflow."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from foxtrot.api.app import create_app
from foxtrot.persistence.labels import LabelError, read_labels, set_label

from test_api import PORT, TOKEN, call, guarded


# Names live beside the immutable IDs and can change at any time.
class LabelTests(unittest.TestCase):
    # Before each test: an app with one imported dataset.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-labels-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        self.checkout, self.workspace = root / "checkout", root / "workspace"
        self.checkout.mkdir()
        source = root / "data"
        for split in ("train", "val"):
            (source / split / "images").mkdir(parents=True)
            (source / split / "labels").mkdir(parents=True)
            Image.new("RGB", (32, 32), (200, 40, 40)).save(source / split / "images" / "a.jpg")
            (source / split / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        self.app = create_app(self.workspace, self.checkout, port=PORT, session_token=TOKEN, static_directory=None)
        status, content, _ = call(self.app, "POST", "/api/datasets", headers=guarded(), body={
            "source": str(source), "dataset_id": "shapes", "class_names": ["A"], "name": "  Red squares   v1 "})
        self.assertEqual(status, 200, content)

    # Import with a name, rename, remove; bad names refused; unknown items 404.
    def test_dataset_names(self):
        listed = lambda: {d["dataset_id"]: d.get("name") for d in json.loads(call(self.app, "GET", "/api/datasets")[1])}
        self.assertEqual(listed(), {"shapes": "Red squares v1"})  # spaces tidied
        status, _, _ = call(self.app, "PUT", "/api/datasets/shapes/name", headers=guarded(), body={"name": "Airfield"})
        self.assertEqual((status, listed()), (200, {"shapes": "Airfield"}))
        call(self.app, "PUT", "/api/datasets/shapes/name", headers=guarded(), body={"name": "  "})
        self.assertEqual(listed(), {"shapes": None})
        status, _, _ = call(self.app, "PUT", "/api/datasets/nope/name", headers=guarded(), body={"name": "x"})
        self.assertEqual(status, 404)
        status, _, _ = call(self.app, "PUT", "/api/datasets/shapes/name", headers=guarded(), body={"name": "x" * 81})
        self.assertEqual(status, 422)
        with self.assertRaises(LabelError):
            set_label(self.workspace, self.checkout, "datasets", "shapes", "bad\x00name")
        self.assertEqual(call(self.app, "PUT", "/api/datasets/shapes/name", body={"name": "x"})[0], 403)  # needs the session

    # A run named at submission keeps the name in lists and becomes the MLflow run name.
    def test_run_name_reaches_mlflow(self):
        from foxtrot.execution.worker import Worker
        from foxtrot.persistence.tracking import list_tracking
        from foxtrot.pipeline.registry import PipelineSettings
        set_label(self.workspace, self.checkout, "datasets", "shapes", "Red squares")
        status, content, _ = call(self.app, "POST", "/api/jobs", headers=guarded(), body={
            "submission_key": "named-run-1", "name": "Baseline nano",
            "request": {"dataset_id": "shapes", "tracking_destination_id": "local", "training": {"epochs": 1}}})
        self.assertEqual(status, 200, content)
        job = json.loads(content)["job_id"]
        self.assertEqual(json.loads(call(self.app, "GET", "/api/jobs")[1])[0]["name"], "Baseline nano")
        Worker(self.workspace, self.checkout, runtime_directory=self.workspace.parent / "runtime",
               pipeline=PipelineSettings(adapter="fixture", fixture_epoch_seconds=0.0)).run(once=True)
        operations = [entry.operation for entry in list_tracking(self.workspace, self.checkout, job)]
        create, params = operations[0].payload, operations[1].payload["params"]
        self.assertEqual((create["run_name"], create["tags"]["foxtrot.name"], params["dataset.name"]),
                         ("Baseline nano", "Baseline nano", "Red squares"))
        call(self.app, "PUT", f"/api/jobs/{job}/name", headers=guarded(), body={"name": "Renamed"})
        self.assertEqual(json.loads(call(self.app, "GET", f"/api/jobs/{job}")[1])["name"], "Renamed")
        self.assertEqual(read_labels(self.workspace, self.checkout)["jobs"], {job: "Renamed"})


if __name__ == "__main__":
    unittest.main()
