"""H7 API contracts, local-request protections, settings redaction, and static serving."""

import asyncio
import json
import logging
from pathlib import Path
import tempfile
import unittest

from foxtrot.api.app import STATIC_DIRECTORY, create_app
from foxtrot.api.settings import REDACTED, RedactingFilter
from foxtrot.persistence.jobs import claim_next_job

from test_pipeline_contracts import make_dataset

PORT = 8765
TOKEN = "test-session-token"
ORIGIN = f"http://127.0.0.1:{PORT}"


def call(app, method, path, *, body=None, headers=None, host="127.0.0.1:8765"):
    """One bounded in-process ASGI request; no socket or TestClient."""
    raw = json.dumps(body).encode() if body is not None else b""
    path, _, query = path.partition("?")
    sent = []
    header_list = [(b"host", host.encode())] + [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if body is not None:
        header_list.append((b"content-type", b"application/json"))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "method": method, "path": path,
             "raw_path": path.encode(), "root_path": "", "query_string": query.encode(),
             "headers": header_list, "scheme": "http", "http_version": "1.1",
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", PORT)}
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await asyncio.sleep(3600)

    async def send(message):
        sent.append(message)

    asyncio.run(asyncio.wait_for(app(scope, receive, send), timeout=10))
    start = next(m for m in sent if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    response_headers = {k.decode(): v.decode() for k, v in start["headers"]}
    return start["status"], content, response_headers


def guarded(extra=None):
    return {"origin": ORIGIN, "x-foxtrot-session": TOKEN, **(extra or {})}


class ApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-h7-api-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.app = create_app(self.workspace, self.checkout, port=PORT, session_token=TOKEN,
                              pipeline_adapter="fixture", static_directory=None)

    def json(self, method, path, **options):
        status, content, _ = call(self.app, method, path, **options)
        return status, json.loads(content)

    def submission(self, key="key-000001", **training):
        return {"submission_key": key, "request": {
            "dataset_id": "shapes", "tracking_destination_id": "local", "training": training}}

    def test_health_capabilities_and_schema(self):
        self.assertEqual(self.json("GET", "/api/health"), (200, {"status": "ok", "database_schema_version": 6}))
        status, capabilities = self.json("GET", "/api/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual((capabilities["models"], capabilities["devices"]), (["yolov8n"], ["cpu"]))
        self.assertEqual(capabilities["active_pipeline_adapter"], "fixture")
        self.assertFalse(capabilities["real_training"])
        status, schemas = self.json("GET", "/api/schemas")
        self.assertEqual(schemas["job"]["$defs"]["TrainingSettings"]["properties"]["epochs"]["default"], 3)

    def test_security_headers_and_unknown_endpoints(self):
        status, _, headers = call(self.app, "GET", "/api/health")
        self.assertIn("frame-ancestors 'none'", headers["content-security-policy"])
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertNotIn("access-control-allow-origin", headers)
        self.assertEqual(self.json("GET", "/api/nope")[0], 404)
        self.assertEqual(self.json("GET", "/api/jobs/" + "0" * 32)[0], 404)

    def test_non_loopback_host_is_refused(self):
        for host in ("evil.example:8765", "192.168.1.5:8765"):
            self.assertEqual(call(self.app, "GET", "/api/session", host=host)[0], 400)

    def test_state_changes_need_session_and_same_origin(self):
        body = self.submission()
        self.assertEqual(self.json("POST", "/api/jobs", body=body)[0], 403)
        self.assertEqual(self.json("POST", "/api/jobs", body=body, headers={"x-foxtrot-session": "wrong"})[0], 403)
        self.assertEqual(self.json("POST", "/api/jobs", body=body,
                                   headers=guarded({"origin": "https://evil.example"}))[0], 403)
        self.assertEqual(self.json("GET", "/api/jobs")[1], [])
        self.assertEqual(self.json("GET", "/api/session")[1], {"token": TOKEN})
        self.assertEqual(self.json("POST", "/api/jobs", body=body, headers=guarded())[0], 200)

    def test_submission_is_idempotent_and_validated_by_field(self):
        status, first = self.json("POST", "/api/jobs", body=self.submission(epochs=5), headers=guarded())
        self.assertEqual((status, first["execution_state"]), (200, "queued"))
        status, again = self.json("POST", "/api/jobs", body=self.submission(epochs=5), headers=guarded())
        self.assertEqual(again["job_id"], first["job_id"])
        self.assertEqual(len(self.json("GET", "/api/jobs")[1]), 1)
        self.assertEqual(self.json("POST", "/api/jobs", body=self.submission(epochs=6), headers=guarded())[0], 409)
        status, error = self.json("POST", "/api/jobs", body=self.submission("key-000002", epochs=0, colour="red"),
                                  headers=guarded())
        self.assertEqual(status, 422)
        self.assertEqual({tuple(item["loc"]) for item in error["detail"]},
                         {("training", "epochs"), ("training", "colour")})

    def test_job_detail_events_cursor_and_cancel(self):
        _, job = self.json("POST", "/api/jobs", body=self.submission(epochs=4), headers=guarded())
        job_id = job["job_id"]
        status, detail = self.json("GET", f"/api/jobs/{job_id}")
        self.assertEqual(detail["resolved"]["training"]["epochs"], 4)
        self.assertIsNone(detail["tracking_run"])
        _, events = self.json("GET", f"/api/jobs/{job_id}/events")
        self.assertEqual([e["payload"]["state"] for e in events], ["queued"])
        self.assertEqual(self.json("GET", f"/api/jobs/{job_id}/events?after=1")[1], [])
        self.assertEqual(self.json("GET", f"/api/jobs/{job_id}/events?after=-1")[0], 422)
        status, cancelled = self.json("POST", f"/api/jobs/{job_id}/cancel", headers=guarded())
        self.assertEqual(cancelled["execution_state"], "cancelled")
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))

    def test_datasets_are_listed_from_manifests(self):
        make_dataset(self.workspace, self.checkout)
        status, datasets = self.json("GET", "/api/datasets")
        self.assertEqual([(d["dataset_id"], d["class_names"]) for d in datasets], [("shapes", ["Circle", "Square"])])

    def test_credentials_are_write_only_private_and_redacted(self):
        secret = "super-secret-token-123"
        body = {"destination_id": "team", "kind": "mlflow", "tracking_uri": "https://mlflow.invalid",
                "browser_url": "https://mlflow.invalid/ui", "username": "me", "token": secret}
        status, saved = self.json("PUT", "/api/tracking/destinations/team", body=body, headers=guarded())
        self.assertEqual(status, 200)
        self.assertTrue(saved["has_credentials"])
        for path in ("/api/tracking/destinations", "/api/capabilities"):
            self.assertNotIn(secret.encode(), call(self.app, "GET", path)[1])
        secrets_file = self.workspace / "settings" / "secrets.json"
        self.assertEqual(secrets_file.stat().st_mode & 0o777, 0o600)
        self.assertNotIn(secret, (self.workspace / "settings" / "tracking.json").read_text())
        self.assertEqual(self.app.state.settings.credentials("team")["token"], secret)
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "connecting with %s", (secret,), None)
        RedactingFilter(self.app.state.settings).filter(record)
        self.assertEqual(record.getMessage(), f"connecting with {REDACTED}")
        bad = {**body, "tracking_uri": "https://user:pass@mlflow.invalid"}
        self.assertEqual(self.json("PUT", "/api/tracking/destinations/team", body=bad, headers=guarded())[0], 422)
        self.assertEqual(self.json("PUT", "/api/tracking/destinations/other", body=body, headers=guarded())[0], 422)

    def test_built_interface_is_served_without_node(self):
        self.assertTrue((STATIC_DIRECTORY / "index.html").is_file(), "run npm run build in frontend/")
        app = create_app(self.workspace, self.checkout, port=PORT, session_token=TOKEN)
        status, content, headers = call(app, "GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b'<div id="root"></div>', content)
        script = content.split(b'src="')[1].split(b'"')[0].decode()
        status, asset, _ = call(app, "GET", script)
        self.assertEqual(status, 200)
        self.assertGreater(len(asset), 1000)
        self.assertEqual(self.json("GET", "/api/health")[0], 200)  # API still wins over static
        status, content, _ = call(self.app, "GET", "/")  # no built interface: helpful page
        self.assertIn(b"npm run build", content)


if __name__ == "__main__":
    unittest.main()
