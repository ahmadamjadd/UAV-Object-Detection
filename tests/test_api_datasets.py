"""V1.3 importing a dataset through the API."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from foxtrot.api.app import create_app

from test_api import PORT, TOKEN, call, guarded


# Tests for POST /api/datasets using a small dataset folder outside the workspace.
class DatasetApiTests(unittest.TestCase):
    # Before each test: create the app and a valid dataset folder to import.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-api-datasets-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.source = self.root / "my-dataset"
        for split, colour in (("train", (255, 0, 0)), ("val", (0, 255, 0))):
            (self.source / split / "images").mkdir(parents=True)
            (self.source / split / "labels").mkdir(parents=True)
            Image.new("RGB", (32, 32), colour).save(self.source / split / "images" / "a.jpg")
            (self.source / split / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        self.app = create_app(self.workspace, self.checkout, port=PORT, session_token=TOKEN, static_directory=None)

    # Helper: send an import request (with the session) and return (status, JSON body).
    def post(self, body, headers=None):
        status, content, _ = call(self.app, "POST", "/api/datasets", body=body,
                                  headers=guarded() if headers is None else headers)
        return status, json.loads(content)

    # Helper: a request body for the test dataset, with optional changes.
    def body(self, **changes):
        return {"source": str(self.source), "dataset_id": "shapes", "class_names": ["Hexagon", "Triangle"], **changes}

    # A valid folder is imported and then appears in the dataset list.
    def test_valid_folder_is_imported_and_listed(self):
        status, result = self.post(self.body())
        self.assertEqual((status, result["imported"]), (200, True))
        self.assertEqual(result["dataset"]["validation_status"], "valid")
        self.assertEqual(result["report"]["splits"]["train"]["objects_per_class"], {"Hexagon": 1, "Triangle": 0})
        _, listed, _ = call(self.app, "GET", "/api/datasets")
        self.assertEqual([d["dataset_id"] for d in json.loads(listed)], ["shapes"])

    # An invalid folder is not imported; the report explains why.
    def test_invalid_folder_returns_report_without_importing(self):
        (self.source / "val" / "labels" / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n")
        status, result = self.post(self.body())
        self.assertEqual((status, result["imported"], result["dataset"]), (200, False, None))
        self.assertEqual(result["report"]["errors"][0]["message"], "Class 7 is outside 0–1")

    # Missing folders, reused IDs and duplicate class names give a clear 422 error.
    def test_bad_requests_are_explained(self):
        self.assertEqual(self.post(self.body(source=str(self.root / "missing")))[0], 422)
        self.post(self.body())
        status, error = self.post(self.body())
        self.assertEqual(status, 422)
        self.assertIn("already used", error["detail"])
        self.assertEqual(self.post(self.body(dataset_id="other", class_names=["A", "A"]))[0], 422)

    # Importing changes the workspace, so it needs the local session like other actions.
    def test_import_requires_local_session(self):
        self.assertEqual(self.post(self.body(), headers={})[0], 403)


if __name__ == "__main__":
    unittest.main()
