"""V1.8b "Test connection": the API checks a saved destination in a separate process."""

import json
from pathlib import Path
import tempfile
import unittest

from foxtrot.api.app import create_app
from foxtrot.api.settings import DestinationUpdate

from test_api import PORT, TOKEN, call, guarded


# Tests for POST /api/tracking/destinations/{id}/check.
class TrackingCheckApiTests(unittest.TestCase):
    # Before each test: an app with a working local destination and a broken one.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v1-check-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        (root / "checkout").mkdir()
        (root / "mlflow").mkdir()
        self.app = create_app(root / "workspace", root / "checkout", port=PORT, session_token=TOKEN,
                              static_directory=None)
        settings = self.app.state.settings
        settings.save_destination(DestinationUpdate(destination_id="local-mlflow", kind="local",
                                                    tracking_uri=f"sqlite:///{root / 'mlflow' / 'mlflow.db'}"))
        settings.save_destination(DestinationUpdate(destination_id="broken", kind="mlflow",
                                                    tracking_uri="http://127.0.0.1:9", token="secret-token-123"))

    # Helper: press "Test connection" for one destination and return (status, JSON answer).
    def check(self, destination_id, headers=None):
        status, content, _ = call(self.app, "POST", f"/api/tracking/destinations/{destination_id}/check",
                                  headers=guarded() if headers is None else headers)
        return status, json.loads(content)

    # A reachable destination answers ok; an unreachable one explains why without the token.
    def test_reachable_and_unreachable_destinations(self):
        status, result = self.check("local-mlflow")
        self.assertEqual((status, result["ok"]), (200, True))
        status, result = self.check("broken")
        self.assertEqual((status, result["ok"]), (200, False))
        self.assertNotIn("secret-token-123", result["message"])

    # Unknown destinations are 404, and the check needs the local session like other actions.
    def test_unknown_destination_and_session(self):
        self.assertEqual(self.check("nope")[0], 404)
        self.assertEqual(self.check("local-mlflow", headers={})[0], 403)


if __name__ == "__main__":
    unittest.main()
