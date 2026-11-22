"""Early-stopping settings and GPU choice: validated, sent to YOLO correctly, refused when absent."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from foxtrot.config.training import TrainingSettings, ultralytics_arguments


# Settings rules that need no workspace.
class SettingsTests(unittest.TestCase):
    # On: YOLO gets the patience; off: patience 0, which YOLO treats as "never stop early".
    def test_early_stopping_switch(self):
        self.assertEqual(ultralytics_arguments(TrainingSettings(patience=7))["patience"], 7)
        self.assertEqual(ultralytics_arguments(TrainingSettings(early_stopping=False, patience=7))["patience"], 0)
        self.assertEqual(TrainingSettings().patience, 100)

    # Devices are "cpu" or "cuda:N"; mixed precision is a setting sent to YOLO.
    def test_device_values(self):
        self.assertEqual(ultralytics_arguments(TrainingSettings(device="cuda:1"))["device"], "cuda:1")
        self.assertIs(ultralytics_arguments(TrainingSettings(amp=False))["amp"], False)
        for bad in ("cuda", "gpu", "cuda:x", "0"):
            with self.subTest(device=bad), self.assertRaises(ValidationError):
                TrainingSettings(device=bad)


# The website refuses a GPU this computer does not have, with the detected reason.
class GpuSubmissionTests(unittest.TestCase):
    # Before each test: an app with a fake "no GPU" detection and one "cuda:0" detection.
    def setUp(self):
        from foxtrot.api.app import create_app
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-gpu-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        (root / "checkout").mkdir()
        self.app = create_app(root / "workspace", root / "checkout", port=8765, session_token="test-session-token",
                              static_directory=None)

    # Submit one job asking for a device; returns (status, body).
    def submit(self, device: str, report: dict):
        from test_api import call, guarded
        with patch("foxtrot.api.app.detect_gpus", return_value=report):
            status, content, _ = call(self.app, "POST", "/api/jobs", headers=guarded(), body={
                "submission_key": f"gpu-{device}", "request": {"dataset_id": "d", "tracking_destination_id": "t",
                                                               "training": {"device": device}}})
        return status, json.loads(content)

    # No GPU: refused with the reason; a detected GPU is accepted.
    def test_gpu_must_exist(self):
        none = {"available": False, "devices": [], "message": "No GPU available: driver not running."}
        status, body = self.submit("cuda:0", none)
        self.assertEqual(status, 422)
        self.assertIn("driver not running", body["detail"])
        one = {"available": True, "devices": [{"id": "cuda:0", "index": 0, "name": "Test GPU", "memory_gb": 8.0}], "message": "1 GPU"}
        self.assertEqual(self.submit("cuda:0", one)[0], 200)
        self.assertEqual(self.submit("cuda:1", one)[0], 422)  # only cuda:0 exists


if __name__ == "__main__":
    unittest.main()
