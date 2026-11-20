"""H4 process identity, restart reconciliation, and controlled process loss."""

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from foxtrot.config.lifecycle import ExecutionState as E
from foxtrot.persistence.database import connect_database
from foxtrot.persistence.events import append_job_event, read_job_events
from foxtrot.persistence.jobs import claim_next_job, create_recorded_job, get_job_record
from foxtrot.persistence.lifecycle import transition_execution
from foxtrot.persistence.processes import (
    Liveness, ProcessIdentity, capture_process_identity, probe_execution_group, probe_process,
)
from foxtrot.persistence.recovery import (
    attach_execution_process, begin_process_launch, reconcile_jobs,
)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-h4-recovery-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        request = {"dataset_id": "demo", "tracking_destination_id": "local"}
        self.job = create_recorded_job(self.workspace, self.checkout, request, "first").name
        self.second = create_recorded_job(self.workspace, self.checkout, request, "second").name

    def record(self):
        return get_job_record(self.workspace, self.checkout, self.job)

    def reconcile(self):
        return reconcile_jobs(self.workspace, self.checkout)

    def sql(self, statement, values=()):
        with connect_database(self.workspace, self.checkout) as connection:
            with connection:
                connection.execute(statement, values)

    def start_owner(self, mode="owner"):
        fixture = Path(__file__).parent / "fixtures" / "lifecycle_owner.py"
        process = subprocess.Popen(
            [sys.executable, "-B", str(fixture), str(self.workspace), str(self.checkout), mode],
            cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=False, start_new_session=True,
        )
        self.addCleanup(self.stop_owner, process)
        # Read bytes directly, avoiding buffered readline hiding a second line
        # from select. Timeouts ensure a failed fixture cannot hang the suite.
        data = b""
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b'"ready": true' not in data:
                if not selector.select(timeout=10):
                    self.fail("Timed out waiting for lifecycle fixture")
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    self.fail("Lifecycle fixture exited: " + process.stderr.read().decode())
                data += chunk
        reports = [json.loads(line) for line in data.splitlines()]
        self.assertEqual(reports[-1]["job_id"], self.job)
        return process

    @staticmethod
    def stop_owner(process):
        # The stdin writer belongs to this test. Closing it stops an attached
        # fixture child even when its owner has already been terminated.
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def test_matching_live_owner_is_retained_across_reopen(self):
        owner = self.start_owner()
        before = self.record()
        self.assertEqual(before.owner_pid, owner.pid)
        self.assertEqual(self.reconcile(), ())
        self.assertEqual(self.record(), before)
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))

    def test_fixture_loss_is_interrupted_with_ordered_durable_history(self):
        owner = self.start_owner()
        before = read_job_events(self.workspace, self.checkout, self.job)
        owner.terminate()
        owner.wait(timeout=5)
        restored, = self.reconcile()
        self.assertEqual(restored.execution_state, "interrupted")
        self.assertEqual(restored.error_details.code, "owner_stopped")
        self.assertEqual(restored.tracking_state, "pending")
        self.assertFalse(restored.recovery_required)
        after = read_job_events(self.workspace, self.checkout, self.job)
        self.assertEqual(after[:-1], before)
        self.assertEqual([event.sequence for event in after], list(range(1, len(after) + 1)))
        self.assertEqual([event.payload["epoch"] for event in after if event.event_type == "progress"], [0, 1, 2])
        self.assertEqual(self.reconcile(), ())
        self.assertEqual(get_job_record(self.workspace, self.checkout, self.second).execution_state, "queued")
        # Recovery records the loss without claiming the next job itself.
        self.assertEqual(claim_next_job(self.workspace, self.checkout).job_id, self.second)

    def test_restarted_interpreter_reconciles_a_dead_owner(self):
        owner = self.start_owner()
        owner.terminate()
        owner.wait(timeout=5)
        code = """
import json, sys
from foxtrot.persistence.recovery import reconcile_jobs
from foxtrot.persistence.events import read_job_events
w, s = sys.argv[1:]
record, = reconcile_jobs(w, s)
print(json.dumps({'state': record.execution_state, 'events': len(read_job_events(w, s, record.job_id))}))
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(self.workspace), str(self.checkout)],
                                cwd=self.root, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"state": "interrupted", "events": 7})

    def test_dead_owner_with_live_child_holds_slot_until_child_stops(self):
        owner = self.start_owner("attached")
        original = self.record()
        identity = ProcessIdentity(original.process_pid, original.process_start_id)
        self.assertEqual(probe_execution_group(identity), Liveness.ALIVE)
        owner.terminate()
        owner.wait(timeout=5)
        interrupted, = self.reconcile()
        self.assertTrue(interrupted.recovery_required)
        self.assertEqual(interrupted.execution_state, "interrupted")
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))
        self.assertEqual(self.reconcile(), ())
        # EOF tells only the known fixture child to stop. No PID-based kill.
        owner.stdin.close()
        for _ in range(100):
            if probe_execution_group(identity) == Liveness.DEAD:
                break
            # Bounded wait for this controlled fixture, not production recovery.
            import time
            time.sleep(0.01)
        self.assertEqual(probe_execution_group(identity), Liveness.DEAD)
        released, = self.reconcile()
        self.assertFalse(released.recovery_required)
        self.assertEqual(released.execution_state, "interrupted")
        self.assertEqual(claim_next_job(self.workspace, self.checkout).job_id, self.second)

    def test_ambiguous_launch_is_interrupted_but_never_assumed_safe(self):
        owner = self.start_owner("pending")
        owner.terminate()
        owner.wait(timeout=5)
        record, = self.reconcile()
        self.assertTrue(record.process_launch_pending)
        self.assertTrue(record.recovery_required)
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))
        self.assertEqual(self.reconcile(), ())

    def test_missing_legacy_ownership_is_interrupted_with_recovery_hold(self):
        self.sql("UPDATE jobs SET execution_state = 'training' WHERE job_id = ?", (self.job,))
        record, = self.reconcile()
        self.assertTrue(record.recovery_required)
        self.assertEqual(record.error_details.code, "ownership_uncertain")
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))

    def test_inaccessible_owner_evidence_is_never_treated_as_dead(self):
        claim_next_job(self.workspace, self.checkout)
        with patch("foxtrot.persistence.recovery.probe_process", return_value=Liveness.UNKNOWN):
            record, = self.reconcile()
        self.assertEqual(record.execution_state, "interrupted")
        self.assertTrue(record.recovery_required)
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))

    def test_reconciliation_event_failure_rolls_back_state(self):
        owner = self.start_owner()
        owner.terminate()
        owner.wait(timeout=5)
        before = self.record()
        with patch("foxtrot.persistence.recovery._append_event", side_effect=OSError("fixture disk failure")):
            with self.assertRaises(OSError):
                self.reconcile()
        self.assertEqual(self.record(), before)

    def test_launch_and_attachment_require_claim_and_owned_dedicated_child(self):
        claim = claim_next_job(self.workspace, self.checkout)
        with self.assertRaises(ValueError):
            begin_process_launch(self.workspace, self.checkout, self.job, "wrong")
        with self.assertRaises(ValueError):
            attach_execution_process(self.workspace, self.checkout, self.job, claim.claim_token, os.getpid())
        first = begin_process_launch(self.workspace, self.checkout, self.job, claim.claim_token)
        self.assertEqual(begin_process_launch(self.workspace, self.checkout, self.job, claim.claim_token), first)
        with self.assertRaises(ValueError):
            transition_execution(self.workspace, self.checkout, self.job, E.CANCELLED, claim_token=claim.claim_token)

    def test_owned_execution_is_recorded_once_and_must_stop_before_finalizing(self):
        claim = claim_next_job(self.workspace, self.checkout)
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", "import sys; sys.stdin.buffer.read()"],
            stdin=subprocess.PIPE, start_new_session=True,
        )
        try:
            with self.assertRaisesRegex(ValueError, "launch intent"):
                attach_execution_process(self.workspace, self.checkout, self.job, claim.claim_token, child.pid)
            begin_process_launch(self.workspace, self.checkout, self.job, claim.claim_token)
            attached = attach_execution_process(self.workspace, self.checkout, self.job, claim.claim_token, child.pid)
            self.assertEqual(attached.process_pid, child.pid)
            self.assertEqual(attached.process_group_id, child.pid)
            self.assertFalse(attached.process_launch_pending)
            self.assertEqual(attach_execution_process(self.workspace, self.checkout, self.job, claim.claim_token, child.pid), attached)
            with self.assertRaisesRegex(ValueError, "already recorded"):
                begin_process_launch(self.workspace, self.checkout, self.job, claim.claim_token)
            with self.assertRaisesRegex(ValueError, "children must stop"):
                transition_execution(self.workspace, self.checkout, self.job, E.CANCELLED, claim_token=claim.claim_token)
            # A live supervisor remains responsible even when its child exits;
            # startup reconciliation must not race its normal outcome recording.
            child.stdin.close()
            child.wait(timeout=5)
            self.assertEqual(self.reconcile(), ())
            record = transition_execution(self.workspace, self.checkout, self.job, E.CANCELLED, claim_token=claim.claim_token)
            self.assertEqual(record.execution_state, "cancelled")
            self.assertEqual(claim_next_job(self.workspace, self.checkout).job_id, self.second)
        finally:
            if not child.stdin.closed:
                child.stdin.close()
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_shared_session_child_cannot_be_attached_as_execution(self):
        claim = claim_next_job(self.workspace, self.checkout)
        begin_process_launch(self.workspace, self.checkout, self.job, claim.claim_token)
        child = subprocess.Popen([sys.executable, "-B", "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE)
        try:
            with self.assertRaisesRegex(ValueError, "dedicated session"):
                attach_execution_process(self.workspace, self.checkout, self.job, claim.claim_token, child.pid)
        finally:
            child.stdin.close()
            child.wait(timeout=5)

    def test_a_different_process_cannot_write_using_the_owners_token(self):
        owner = self.start_owner()
        token = self.record().claim_token
        with self.assertRaisesRegex(ValueError, "recorded owner"):
            append_job_event(self.workspace, self.checkout, self.job, "progress", {}, claim_token=token)
        with self.assertRaisesRegex(ValueError, "recorded owner"):
            transition_execution(self.workspace, self.checkout, self.job, E.EVALUATING, claim_token=token)
        self.assertIsNone(owner.poll())


class ProcessEvidenceTests(unittest.TestCase):
    def test_current_process_birth_identity_is_live(self):
        identity = capture_process_identity(os.getpid())
        self.assertEqual(probe_process(identity), Liveness.ALIVE)

    def test_reused_pid_is_not_the_original_owner(self):
        identity = capture_process_identity(os.getpid())
        parts = identity.start_id.split("|")
        parts[2] = str(int(parts[2]) + 1)
        self.assertEqual(probe_process(ProcessIdentity(identity.pid, "|".join(parts))), Liveness.DEAD)

    def test_boot_change_is_dead_and_namespace_change_is_unknown(self):
        identity = capture_process_identity(os.getpid())
        boot, namespace, start = identity.start_id.split("|")
        self.assertEqual(probe_process(ProcessIdentity(identity.pid, f"different|{namespace}|{start}")), Liveness.DEAD)
        self.assertEqual(probe_process(ProcessIdentity(identity.pid, f"{boot}|different|{start}")), Liveness.UNKNOWN)

    def test_unreadable_or_malformed_evidence_is_unknown(self):
        identity = capture_process_identity(os.getpid())
        with patch("foxtrot.persistence.processes._snapshot", side_effect=PermissionError):
            self.assertEqual(probe_process(identity), Liveness.UNKNOWN)
            self.assertEqual(probe_execution_group(identity), Liveness.UNKNOWN)
        self.assertEqual(probe_process(ProcessIdentity(identity.pid, "broken")), Liveness.UNKNOWN)

    def test_dead_process_remains_dead_after_exit(self):
        child = subprocess.Popen([sys.executable, "-B", "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE)
        try:
            identity = capture_process_identity(child.pid)
            child.stdin.close()
            child.wait(timeout=5)
            self.assertEqual(probe_process(identity), Liveness.DEAD)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_surviving_session_child_is_detected_after_leader_exits(self):
        # Parent and grandchild share a read pipe. EOF is a bounded, PID-free
        # cleanup mechanism even after the session leader has exited.
        read_end, write_end = os.pipe()
        script = """
import subprocess, sys
child = subprocess.Popen([sys.executable, '-B', '-c', 'import sys; sys.stdin.buffer.read()'],
                         stdin=int(sys.argv[1]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('ready', flush=True)
sys.stdin.buffer.read()
"""
        leader = subprocess.Popen([sys.executable, "-B", "-c", script, str(read_end)],
                                  pass_fds=(read_end,), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  start_new_session=True)
        try:
            identity = capture_process_identity(leader.pid)
            with selectors.DefaultSelector() as selector:
                selector.register(leader.stdout, selectors.EVENT_READ)
                self.assertTrue(selector.select(timeout=10))
            self.assertEqual(leader.stdout.readline(), b"ready\n")
            leader.stdin.close()
            leader.wait(timeout=5)
            self.assertEqual(probe_process(identity), Liveness.DEAD)
            self.assertEqual(probe_execution_group(identity), Liveness.ALIVE)
        finally:
            os.close(read_end)
            os.close(write_end)
            if leader.poll() is None:
                leader.kill()
                leader.wait(timeout=5)
            leader.stdout.close()
            if not leader.stdin.closed:
                leader.stdin.close()
