"""H7 completion check: one launch serves the interface/API and runs the worker."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from foxtrot.execution.locking import worker_lock

from test_pipeline_contracts import make_dataset


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def network_address() -> str | None:
    """A non-loopback local address, if this machine has a route (no traffic is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1; UDP connect only selects a route
            address = probe.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") else address


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-h7-launch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.runtime = self.root / "runtime"
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"

    def launch(self, *extra):
        process = subprocess.Popen(
            [sys.executable, "-B", "-m", "foxtrot.api.launcher", "--workspace", str(self.workspace),
             "--source-checkout", str(self.checkout), "--port", str(self.port), "--no-browser",
             "--runtime-directory", str(self.runtime), *extra],
            cwd=self.root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.addCleanup(self._stop, process)
        return process

    def _stop(self, process):
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()

    def request(self, method, path, body=None, session=None):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if session:
            headers.update({"X-Foxtrot-Session": session, "Origin": self.base})
        data = json.dumps(body).encode() if body is not None else None
        with urllib.request.urlopen(urllib.request.Request(self.base + path, data, headers, method=method),
                                    timeout=5) as response:
            content = response.read()
            return json.loads(content) if response.headers.get_content_type() == "application/json" else content

    def wait_for(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                value = predicate()
                if value:
                    return value
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(0.1)
        self.fail("Timed out waiting for the launched application")

    def test_one_launch_serves_interface_runs_worker_and_stops_cleanly(self):
        make_dataset(self.workspace, self.checkout)
        process = self.launch("--pipeline-adapter", "fixture")
        self.wait_for(lambda: self.request("GET", "/api/health")["status"] == "ok")
        page = self.request("GET", "/")
        self.assertIn(b'<div id="root"></div>', page)  # built interface, no Node server
        self.assertEqual(self.request("GET", "/api/capabilities")["active_pipeline_adapter"], "fixture")
        session = self.request("GET", "/api/session")["token"]
        job = self.request("POST", "/api/jobs", {"submission_key": "launcher-test-1", "request": {
            "dataset_id": "shapes", "tracking_destination_id": "local", "training": {"epochs": 3}}}, session)
        finished = self.wait_for(lambda: (d := self.request("GET", f"/api/jobs/{job['job_id']}"))
                                 ["execution_state"] == "finished" and d)
        self.assertEqual(finished["exit_code"], 0)
        events = self.request("GET", f"/api/jobs/{job['job_id']}/events")
        self.assertEqual([e["payload"]["epoch"] for e in events if e["event_type"] == "progress"], [1, 2, 3])
        # Only loopback is served: the machine's network address must not answer.
        address = network_address()
        if address is not None:
            with self.assertRaises(OSError):
                socket.create_connection((address, self.port), timeout=0.5).close()
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, stderr.decode())
        self.assertIn(b"Foxtrot is running at", stdout)
        # The worker stopped too: its lease is free again.
        with worker_lock(self.runtime / "worker.lock"):
            pass

    def test_port_in_use_is_reported(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", self.port))
            busy.listen()
            process = self.launch("--no-worker")
            _, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 1)
        self.assertIn(b"port already in use", stderr)


if __name__ == "__main__":
    unittest.main()
