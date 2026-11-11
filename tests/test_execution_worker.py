"""H5 acceptance with real isolated worker/child processes and no training."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from foxtrot.execution.locking import WorkerAlreadyRunning, worker_lock
from foxtrot.execution.process_control import stop_session
from foxtrot.execution.protocol import FixtureSettings
from foxtrot.persistence.artifacts import get_job_artifact
from foxtrot.persistence.commands import request_cancel
from foxtrot.persistence.events import read_job_events
from foxtrot.persistence.jobs import create_recorded_job, get_job_record
from foxtrot.persistence.processes import Liveness, ProcessIdentity, probe_execution_group


class WorkerAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-h5-worker-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.runtime = self.root / "runtime"
        self.workers = []
        self.jobs = []
        self.addCleanup(self.cleanup_workers)

    def submit(self, key="first", epochs=7, workspace=None):
        workspace = workspace or self.workspace
        directory = create_recorded_job(workspace, self.checkout, {
            "dataset_id": "fixture-data", "tracking_destination_id": "fixture-tracking", "training": {"epochs": epochs},
        }, key)
        self.jobs.append((workspace, directory.name))
        return directory.name

    def start(self, fixture=None, *, once=False, workspace=None):
        settings = self.root / f"fixture-settings-{len(self.workers)}.json"
        settings.write_text((fixture or FixtureSettings()).model_dump_json())
        command = [sys.executable, "-B", "-m", "foxtrot.execution.worker",
                   "--workspace", str(workspace or self.workspace), "--source-checkout", str(self.checkout),
                   "--fixture", "--fixture-settings", str(settings), "--runtime-directory", str(self.runtime)]
        if once:
            command.append("--once")
        process = subprocess.Popen(command, cwd=self.root, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        self.workers.append(process)
        return process

    def cleanup_workers(self):
        for process in self.workers:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
        for workspace, job in self.jobs:
            record = get_job_record(workspace, self.checkout, job)
            if record.process_pid and record.process_start_id:
                stop_session(ProcessIdentity(record.process_pid, record.process_start_id),
                             terminate_seconds=0.2, kill_seconds=1.0)

    def record(self, job):
        return get_job_record(self.workspace, self.checkout, job)

    def wait_for(self, predicate, *, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        self.fail("Timed out waiting for worker fixture; " + str([(p.pid, p.poll()) for p in self.workers]))

    def wait_state(self, job, states):
        return self.wait_for(lambda: (r if (r := self.record(job)).execution_state in states else None))

    def wait_done(self, process):
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stderr.decode())
        self.assertEqual(stdout, b"")

    def test_success_records_explicit_configuration_progress_logs_exit_and_result(self):
        job = self.submit()
        self.wait_done(self.start(once=True))
        record = self.record(job)
        self.assertEqual((record.execution_state, record.tracking_state, record.exit_code), ("finished", "pending", 0))
        self.assertIsNotNone(record.finished_at)
        self.assertIsNotNone(record.owner_start_id)
        self.assertIsNotNone(record.process_start_id)
        self.assertNotEqual(record.owner_pid, record.process_pid)
        result = get_job_artifact(self.workspace, self.checkout, job, "fixture-result")
        value = json.loads((self.workspace / "jobs" / job / result.relative_path).read_text())
        self.assertEqual(value["resolved"]["training"]["epochs"], 7)
        self.assertEqual(value["job_id"], job)
        summary = get_job_artifact(self.workspace, self.checkout, job, "summary")
        self.assertTrue(summary.size_bytes)
        output = get_job_artifact(self.workspace, self.checkout, job, "stdout-log")
        self.assertIn("fixture step 3", (self.workspace / "jobs" / job / output.relative_path).read_text())
        events = read_job_events(self.workspace, self.checkout, job)
        self.assertEqual([e.sequence for e in events], list(range(1, len(events) + 1)))
        self.assertEqual([e.payload["state"] for e in events if e.event_type == "state_changed"],
                         ["queued", "preparing", "training", "evaluating", "finished"])

    def test_controlled_failure_has_nonzero_exit_and_durable_reason(self):
        job = self.submit()
        self.wait_done(self.start(FixtureSettings(fail_at_step=2), once=True))
        record = self.record(job)
        self.assertEqual((record.execution_state, record.exit_code), ("failed", 17))
        self.assertEqual(record.error_details.code, "execution_failed")
        error_log = get_job_artifact(self.workspace, self.checkout, job, "stderr-log")
        self.assertIn("controlled fixture failure", (self.workspace / "jobs" / job / error_log.relative_path).read_text())
        self.assertEqual(record.tracking_state, "pending")

    def test_graceful_cancel_preserves_history_and_stops_session(self):
        job = self.submit()
        process = self.start(FixtureSettings(steps=100, step_seconds=0.1), once=True)
        self.wait_state(job, {"training"})
        request_cancel(self.workspace, self.checkout, job)
        self.wait_done(process)
        record = self.record(job)
        self.assertEqual((record.execution_state, record.exit_code), ("cancelled", 130))
        self.assertEqual(probe_execution_group(ProcessIdentity(record.process_pid, record.process_start_id)), Liveness.DEAD)
        self.assertFalse(record.recovery_required)

    def test_stubborn_root_and_loader_child_are_killed_within_bound(self):
        job = self.submit()
        process = self.start(FixtureSettings(steps=100, step_seconds=0.1, spawn_child=True,
                                           ignore_cancel=True, child_ignores_term=True, child_new_group=True), once=True)
        self.wait_for(lambda: any("fixture_child_pid" in e.payload for e in read_job_events(self.workspace, self.checkout, job)))
        started = time.monotonic()
        request_cancel(self.workspace, self.checkout, job)
        self.wait_done(process)
        self.assertLess(time.monotonic() - started, 8)
        record = self.record(job)
        self.assertEqual(record.execution_state, "cancelled")
        self.assertEqual(record.exit_code, -signal.SIGKILL)
        self.assertEqual(probe_execution_group(ProcessIdentity(record.process_pid, record.process_start_id)), Liveness.DEAD)
        signals = [e.payload.get("signal") for e in read_job_events(self.workspace, self.checkout, job)]
        self.assertIn(signal.SIGTERM, signals)
        self.assertIn(signal.SIGKILL, signals)

    def test_worker_shutdown_interrupts_active_job_and_leaves_queue(self):
        first = self.submit()
        second = self.submit("second")
        process = self.start(FixtureSettings(steps=100, step_seconds=0.1, spawn_child=True))
        self.wait_state(first, {"training"})
        process.terminate()
        self.wait_done(process)
        record = self.record(first)
        self.assertEqual(record.execution_state, "interrupted")
        self.assertEqual(record.error_details.code, "worker_shutdown")
        self.assertEqual(self.record(second).execution_state, "queued")
        self.assertEqual(probe_execution_group(ProcessIdentity(record.process_pid, record.process_start_id)), Liveness.DEAD)

    def test_competing_worker_is_rejected_across_workspaces(self):
        job = self.submit()
        owner = self.start(FixtureSettings(steps=100, step_seconds=0.1))
        self.wait_state(job, {"training"})
        other_workspace = self.root / "other-workspace"
        other = self.submit("other", workspace=other_workspace)
        competitor = self.start(once=True, workspace=other_workspace)
        _, error = competitor.communicate(timeout=10)
        self.assertEqual(competitor.returncode, 2)
        self.assertIn(b"already running", error)
        self.assertEqual(get_job_record(other_workspace, self.checkout, other).execution_state, "queued")
        self.assertIsNone(owner.poll())

    def test_worker_runs_queued_jobs_sequentially_and_closing_client_does_not_stop_it(self):
        first, second = self.submit(), self.submit("second", epochs=9)
        worker = self.start(FixtureSettings(steps=10, step_seconds=0.05))
        self.wait_state(first, {"training"})
        self.assertEqual(self.record(second).execution_state, "queued")
        # Worker stdin is already /dev/null and is unrelated to any browser.
        self.wait_state(second, {"finished"})
        self.assertEqual(self.record(first).execution_state, "finished")
        self.assertNotEqual(self.record(first).process_pid, self.record(second).process_pid)
        self.assertIsNone(worker.poll())

    def test_large_stdout_and_stderr_are_drained_without_deadlock(self):
        job = self.submit()
        self.wait_done(self.start(FixtureSettings(output_lines=1000), once=True))
        self.assertEqual(self.record(job).execution_state, "finished")
        for channel in ("stdout", "stderr"):
            artifact = get_job_artifact(self.workspace, self.checkout, job, f"{channel}-log")
            self.assertGreater(artifact.size_bytes, 200000)

    def test_http_fixture_remains_responsive_during_execution(self):
        # H7 supplies the product API. This bounded ASGI fixture exercises
        # HTTP dispatch + persisted reads while a separate worker is training.
        job = self.submit()
        worker = self.start(FixtureSettings(steps=100, step_seconds=0.1))
        self.wait_state(job, {"training"})
        fixture = Path(__file__).parent / "fixtures" / "asgi_health.py"
        result = subprocess.run([sys.executable, "-B", str(fixture), str(self.workspace), str(self.checkout), job],
                                cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["requests"], 10)
        self.assertLess(report["seconds"], 3)
        self.assertIsNone(worker.poll())

    def test_execution_crash_cleans_survivors_and_restart_only_runs_next_job(self):
        first = self.submit()
        second = self.submit("second")
        worker = self.start(FixtureSettings(steps=100, step_seconds=0.1, spawn_child=True,
                                           child_ignores_term=True, child_new_group=True), once=True)
        self.wait_for(lambda: any("fixture_child_pid" in e.payload for e in read_job_events(self.workspace, self.checkout, first)))
        before = self.record(first)
        descriptor = os.pidfd_open(before.process_pid)
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
        self.wait_done(worker)
        record = self.record(first)
        self.assertEqual((record.execution_state, record.exit_code), ("failed", -signal.SIGKILL))
        self.assertEqual(probe_execution_group(ProcessIdentity(record.process_pid, record.process_start_id)), Liveness.DEAD)
        cursor = record.last_event_sequence
        self.assertEqual(self.record(second).execution_state, "queued")
        self.wait_done(self.start(once=True))
        self.assertEqual(self.record(first).last_event_sequence, cursor)
        self.assertEqual(self.record(first).claim_token, before.claim_token)
        self.assertEqual(self.record(second).execution_state, "finished")

    def test_idle_worker_shutdown_releases_lease_for_next_worker(self):
        job = self.submit()
        request_cancel(self.workspace, self.checkout, job)
        process = self.start()

        def lease_held():
            if not (self.runtime / "worker.lock").exists():
                return False
            try:
                with worker_lock(self.runtime / "worker.lock"):
                    return False
            except WorkerAlreadyRunning:
                return True

        self.wait_for(lease_held)
        process.terminate()
        self.wait_done(process)
        self.wait_done(self.start(once=True))
