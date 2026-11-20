"""H5 input/protocol/lease guards and durable cancellation commands."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from foxtrot.config.lifecycle import ErrorDetails, ExecutionState as E
from foxtrot.execution.locking import WorkerAlreadyRunning, worker_lock
from foxtrot.execution.protocol import FixtureSettings, MESSAGE
from foxtrot.execution.supervisor import Supervisor, SupervisorLimits
from foxtrot.execution.worker import Worker
from foxtrot.persistence.commands import request_cancel
from foxtrot.persistence.database import DATABASE_SCHEMA_VERSION, _MIGRATIONS, initialize_database
from foxtrot.persistence.events import read_job_events
from foxtrot.persistence.jobs import claim_next_job, create_recorded_job, get_job_record
from foxtrot.persistence.lifecycle import transition_execution
from foxtrot.persistence.workspace import initialize_workspace


class ExecutionContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-h5-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.request = {"dataset_id": "fixture", "tracking_destination_id": "local"}

    def submit(self):
        return create_recorded_job(self.workspace, self.checkout, self.request, "one").name

    def test_cancel_queued_job_is_immediate_idempotent_and_never_claimed(self):
        job = self.submit()
        cancelled = request_cancel(self.workspace, self.checkout, job)
        self.assertEqual(cancelled.execution_state, "cancelled")
        self.assertTrue(cancelled.cancel_requested)
        self.assertIsNone(cancelled.exit_code)
        self.assertIsNotNone(cancelled.finished_at)
        self.assertEqual(request_cancel(self.workspace, self.checkout, job), cancelled)
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))

    def test_active_cancel_is_a_command_until_owner_stops(self):
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        before = get_job_record(self.workspace, self.checkout, job)
        requested = request_cancel(self.workspace, self.checkout, job)
        self.assertEqual(requested.execution_state, "preparing")
        self.assertTrue(requested.cancel_requested)
        self.assertEqual(requested.last_event_sequence, before.last_event_sequence + 1)
        self.assertEqual(request_cancel(self.workspace, self.checkout, job), requested)
        final = transition_execution(self.workspace, self.checkout, job, E.FAILED,
                                     claim_token=claim.claim_token, error=ErrorDetails(code="failure", message="Failure"), exit_code=-15)
        self.assertEqual(final.execution_state, "cancelled")
        self.assertIsNone(final.error_details)
        self.assertEqual(final.exit_code, -15)

    def test_cancel_wins_atomic_finalization_and_cannot_restart_terminal_job(self):
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        for state in (E.TRAINING, E.EVALUATING):
            transition_execution(self.workspace, self.checkout, job, state, claim_token=claim.claim_token)
        request_cancel(self.workspace, self.checkout, job)
        record = transition_execution(self.workspace, self.checkout, job, E.FINISHED,
                                       claim_token=claim.claim_token, exit_code=0)
        self.assertEqual(record.execution_state, "cancelled")
        self.assertEqual(request_cancel(self.workspace, self.checkout, job), record)
        with self.assertRaises(TypeError):
            transition_execution(self.workspace, self.checkout, job, "finished", claim_token=claim.claim_token)

    def test_cancel_unknown_job_fails_without_creating_one(self):
        self.submit()
        with self.assertRaises(KeyError):
            request_cancel(self.workspace, self.checkout, "missing")

    def test_fixture_and_protocol_reject_arbitrary_commands_and_invalid_numbers(self):
        for values in ({"command": "echo unsafe"}, {"step_seconds": float("nan")}, {"steps": True}, {"steps": 0}):
            with self.assertRaises(ValidationError):
                FixtureSettings(**values)
        for message in ({"kind": "execute", "command": "ignored"},
                        {"kind": "phase", "phase": "finished"},
                        {"kind": "update", "payload": {"loss": float("inf")}}):
            with self.assertRaises(ValidationError):
                MESSAGE.validate_json(json.dumps(message))
        with self.assertRaises(ValidationError):
            from foxtrot.config.job import JobSpec
            JobSpec(**self.request, command="ignored")

    def test_limits_reject_unbounded_or_negative_timeouts(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                SupervisorLimits(grace_seconds=value)

    def test_locks_reject_competition_and_release_without_unlinking(self):
        path = self.root / "private" / "worker.lock"
        with worker_lock(path):
            inode = path.stat().st_ino
            with self.assertRaises(WorkerAlreadyRunning):
                with worker_lock(path):
                    self.fail("Second lock was acquired")
        self.assertEqual(path.stat().st_ino, inode)
        with worker_lock(path):
            self.assertTrue(path.is_file())

    def test_lock_rejects_symlink_and_nonprivate_parent(self):
        directory = self.root / "private"
        directory.mkdir(mode=0o700)
        target = self.root / "keep"
        target.write_text("keep")
        link = directory / "worker.lock"
        link.symlink_to(target)
        with self.assertRaises(OSError):
            with worker_lock(link):
                self.fail("Followed lock symlink")
        self.assertEqual(target.read_text(), "keep")
        link.unlink()
        directory.chmod(0o755)
        with self.assertRaises(ValueError):
            with worker_lock(link):
                self.fail("Accepted public lock directory")

    def test_spawn_failure_clears_only_live_owners_unstarted_intent(self):
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        supervisor = Supervisor(self.workspace, self.checkout, FixtureSettings(), threading.Event(), SupervisorLimits())
        with patch("foxtrot.execution.supervisor.subprocess.Popen", side_effect=OSError("fixture spawn failed")):
            record = supervisor.run(claim)
        self.assertEqual(record.execution_state, "failed")
        self.assertFalse(record.process_launch_pending)
        self.assertIsNone(record.process_pid)
        self.assertIsNone(record.exit_code)
        self.assertEqual(record.error_details.code, "supervision_failed")
        self.assertTrue(any(e.payload.get("action") == "failed_launch_settled"
                            for e in read_job_events(self.workspace, self.checkout, job)))

    def test_attachment_failure_stops_unstarted_child_before_releasing_intent(self):
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        supervisor = Supervisor(self.workspace, self.checkout, FixtureSettings(), threading.Event(), SupervisorLimits())
        with patch("foxtrot.execution.supervisor.attach_execution_process", side_effect=sqlite3.OperationalError("fixture failed attach")):
            record = supervisor.run(claim)
        self.assertEqual(record.execution_state, "failed")
        self.assertFalse(record.process_launch_pending)
        self.assertIsNone(record.process_pid)
        self.assertLess(record.exit_code, 0)

    def test_shutdown_before_launch_interrupts_claim_without_spawning(self):
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        stop = threading.Event()
        stop.set()
        supervisor = Supervisor(self.workspace, self.checkout, FixtureSettings(), stop, SupervisorLimits())
        with patch("foxtrot.execution.supervisor.subprocess.Popen") as spawn:
            record = supervisor.run(claim)
            spawn.assert_not_called()
        self.assertEqual(record.execution_state, "interrupted")
        self.assertIsNone(record.process_pid)
        self.assertEqual(record.error_details.code, "worker_shutdown")

    def test_uncertain_cleanup_is_interrupted_and_blocks_another_claim(self):
        from foxtrot.execution.process_control import stop_session
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        supervisor = Supervisor(self.workspace, self.checkout, FixtureSettings(), threading.Event(), SupervisorLimits())

        def uncertain(identity, **options):
            # Stop the real test process, then simulate an uncertain observation.
            self.assertTrue(stop_session(identity, **options))
            return False

        with patch("foxtrot.execution.supervisor.MESSAGE.validate_json", side_effect=ValueError("fixture failure")), \
                patch("foxtrot.execution.supervisor.stop_session", side_effect=uncertain):
            with self.assertRaisesRegex(RuntimeError, "claim remains blocked"):
                supervisor.run(claim)
        record = get_job_record(self.workspace, self.checkout, job)
        self.assertEqual(record.execution_state, "interrupted")
        self.assertTrue(record.recovery_required)
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))

    def test_cancel_request_event_failure_rolls_back_the_command(self):
        job = self.submit()
        claim_next_job(self.workspace, self.checkout)
        before = get_job_record(self.workspace, self.checkout, job)
        with patch("foxtrot.persistence.commands._append_event", side_effect=sqlite3.OperationalError("fixture disk failure")):
            with self.assertRaises(sqlite3.OperationalError):
                request_cancel(self.workspace, self.checkout, job)
        self.assertEqual(get_job_record(self.workspace, self.checkout, job), before)

    def test_malformed_protocol_fails_and_stops_the_owned_session(self):
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        supervisor = Supervisor(self.workspace, self.checkout, FixtureSettings(), threading.Event(), SupervisorLimits())
        with patch("foxtrot.execution.supervisor.MESSAGE.validate_json", side_effect=ValueError("fixture invalid message")):
            record = supervisor.run(claim)
        self.assertEqual(record.execution_state, "failed")
        self.assertEqual(record.error_details.code, "supervision_failed")
        from foxtrot.persistence.processes import Liveness, ProcessIdentity, probe_execution_group
        self.assertEqual(probe_execution_group(ProcessIdentity(record.process_pid, record.process_start_id)), Liveness.DEAD)

    def test_preexisting_output_symlink_cannot_redirect_execution_writes(self):
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        external = self.root / "external"
        external.mkdir()
        (claim.directory / "output").symlink_to(external, target_is_directory=True)
        supervisor = Supervisor(self.workspace, self.checkout, FixtureSettings(), threading.Event(), SupervisorLimits())
        with patch("foxtrot.execution.supervisor.subprocess.Popen") as spawn:
            record = supervisor.run(claim)
            spawn.assert_not_called()
        self.assertEqual(record.execution_state, "failed")
        self.assertEqual(list(external.iterdir()), [])

    def test_request_validation_refuses_redirected_managed_paths(self):
        from foxtrot.execution.child import load_request
        job = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        supervisor = Supervisor(self.workspace, self.checkout, FixtureSettings(), threading.Event(), SupervisorLimits())
        with patch("foxtrot.execution.supervisor.subprocess.Popen", side_effect=OSError("fixture")):
            supervisor.run(claim)
        request = claim.directory / "execution.json"
        self.assertEqual(load_request(request).job_id, job)
        value = json.loads(request.read_text())
        value["output_directory"] = str(self.root / "external")
        request.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            load_request(request)
        self.assertFalse((self.root / "external").exists())

    def test_changed_configuration_is_failed_without_spawning(self):
        job = self.submit()
        path = self.workspace / "jobs" / job / "resolved.json"
        value = json.loads(path.read_text())
        value["training"]["epochs"] = 99
        path.write_text(json.dumps(value))
        worker = Worker(self.workspace, self.checkout, fixture=FixtureSettings(), runtime_directory=self.root / "runtime")
        with patch("foxtrot.execution.supervisor.subprocess.Popen") as spawn:
            self.assertEqual(worker.run(once=True), 1)
            spawn.assert_not_called()
        record = get_job_record(self.workspace, self.checkout, job)
        self.assertEqual(record.execution_state, "failed")
        self.assertEqual(record.error_details.code, "invalid_configuration")

    def test_migrate_four_to_five_preserves_history_and_defaults(self):
        root = initialize_workspace(self.workspace, self.checkout)
        database = root / "app.sqlite"
        with closing(sqlite3.connect(database)) as connection:
            with connection:
                for version in range(1, 5):
                    for statement in _MIGRATIONS[version]:
                        connection.execute(statement)
                connection.execute("INSERT INTO jobs (job_id, submission_key, last_event_sequence) VALUES ('legacy', 'key', 7)")
                connection.execute("PRAGMA user_version = 4")
        initialize_database(root, self.checkout)
        record = get_job_record(root, self.checkout, "legacy")
        self.assertEqual(record.last_event_sequence, 7)
        self.assertFalse(record.cancel_requested)
        self.assertIsNone(record.exit_code)
        with closing(sqlite3.connect(database)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], DATABASE_SCHEMA_VERSION)
