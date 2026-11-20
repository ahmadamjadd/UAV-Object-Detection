"""V1.6b pretrained weights: install, fingerprint, verify, and the API/Setup endpoints.

No internet: downloads are replaced by a fake that serves bytes shaped like a
PyTorch weights file.
"""

import io
import json
from pathlib import Path
import tempfile
import unittest

from foxtrot.api.app import create_app
from foxtrot.persistence.models import (
    WeightsError, install_weights, list_weights, read_weights_record, verify_weights,
)
from foxtrot.pipeline.contracts import PipelineFailure
from foxtrot.pipeline.ultralytics_adapter import UltralyticsTrainer

from test_api import PORT, TOKEN, call, guarded

FAKE_WEIGHTS = b"PK" + b"\0" * 200_000  # zip-like start and a realistic minimum size


# Pretends to be the internet: returns the given bytes for any URL and remembers the URL.
class FakeOpener:
    # Store the bytes to serve.
    def __init__(self, content=FAKE_WEIGHTS):
        self.content, self.urls = content, []

    # Act like urllib.request.urlopen.
    def __call__(self, url, timeout):
        self.urls.append(url)
        return io.BytesIO(self.content)


# Tests for installing and checking pretrained weights.
class ModelWeightsTests(unittest.TestCase):
    # Before each test: a fresh checkout path and workspace path.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-weights-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.models = self.workspace / "models"

    # Downloading records the official URL, size and fingerprint; repeating changes nothing.
    def test_download_is_recorded_and_idempotent(self):
        opener = FakeOpener()
        record = install_weights(self.workspace, self.checkout, "yolov8n", opener=opener)
        self.assertEqual(opener.urls, ["https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"])
        self.assertEqual(record.size_bytes, len(FAKE_WEIGHTS))
        self.assertEqual(verify_weights(self.models, "yolov8n"), record.sha256)
        self.assertEqual(read_weights_record(self.models, "yolov8n"), record)
        self.assertEqual(install_weights(self.workspace, self.checkout, "yolov8n", opener=opener), record)
        self.assertEqual(len(opener.urls), 1)  # already installed: no second download
        self.assertEqual(sorted(p.name for p in self.models.iterdir()), ["yolov8n.json", "yolov8n.pt"])

    # A local file can be installed instead of downloading.
    def test_local_file_can_be_installed(self):
        local = self.root / "my-yolov8n.pt"
        local.write_bytes(FAKE_WEIGHTS)
        record = install_weights(self.workspace, self.checkout, "yolov8n", source_file=local)
        self.assertEqual(record.source, f"file://{local}")

    # An error page, a tiny file, or an unknown model is refused and leaves nothing behind.
    def test_bad_downloads_and_unknown_models_are_refused(self):
        for content in (b"<html>Not Found</html>", b"PK" + b"\0" * 10):
            with self.assertRaises(WeightsError):
                install_weights(self.workspace, self.checkout, "yolov8n", opener=FakeOpener(content))
        with self.assertRaises(WeightsError):
            install_weights(self.workspace, self.checkout, "yolov99", opener=FakeOpener())
        self.assertEqual(list(self.models.iterdir()), [])
        self.assertEqual(list_weights(self.workspace, self.checkout)[0]["status"], "missing")

    # Weights changed after installing are reported, blocked for training, and replaced by reinstalling.
    def test_changed_weights_are_detected_and_repaired(self):
        install_weights(self.workspace, self.checkout, "yolov8n", opener=FakeOpener())
        (self.models / "yolov8n.pt").write_bytes(FAKE_WEIGHTS + b"tampered")
        self.assertEqual(list_weights(self.workspace, self.checkout)[0]["status"], "changed")
        with self.assertRaisesRegex(WeightsError, "changed"):
            verify_weights(self.models, "yolov8n")
        install_weights(self.workspace, self.checkout, "yolov8n", opener=FakeOpener())
        self.assertEqual(list_weights(self.workspace, self.checkout)[0]["status"], "installed")

    # The trainer uses verified weights and refuses changed ones, before loading Ultralytics.
    def test_trainer_checks_weights_fingerprint(self):
        from foxtrot.config.job import JobSpec
        request = type("Request", (), {"resolved": JobSpec(dataset_id="d", tracking_destination_id="t"),
                                       "job_directory": str(self.workspace / "jobs" / ("a" * 32))})()
        trainer = UltralyticsTrainer(initialization="pretrained")
        from foxtrot.persistence.workspace import initialize_workspace
        initialize_workspace(self.workspace, self.checkout)  # creates models/ the proper way
        with self.assertRaises(PipelineFailure) as missing:
            trainer._model_source(request)
        self.assertEqual(missing.exception.code, "weights_missing")
        record = install_weights(self.workspace, self.checkout, "yolov8n", opener=FakeOpener())
        self.assertEqual(trainer._model_source(request), (str(self.models / "yolov8n.pt"), record.sha256))
        (self.models / "yolov8n.pt").write_bytes(FAKE_WEIGHTS + b"x")
        with self.assertRaises(PipelineFailure) as changed:
            trainer._model_source(request)
        self.assertEqual(changed.exception.code, "weights_changed")

    # The API lists weights, installs them with the session, and refuses without it.
    def test_api_lists_and_installs_weights(self):
        app = create_app(self.workspace, self.checkout, port=PORT, session_token=TOKEN, static_directory=None)
        app.state.weights_opener = FakeOpener()
        status, content, _ = call(app, "GET", "/api/models")
        self.assertEqual(json.loads(content)[0]["status"], "missing")
        self.assertEqual(call(app, "POST", "/api/models/yolov8n/install")[0], 403)
        status, content, _ = call(app, "POST", "/api/models/yolov8n/install", headers=guarded())
        self.assertEqual((status, json.loads(content)["status"]), (200, "installed"))
        self.assertEqual(call(app, "POST", "/api/models/yolov99/install", headers=guarded())[0], 422)


if __name__ == "__main__":
    unittest.main()
