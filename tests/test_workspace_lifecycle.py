"""H4 events, state updates, rollback, and pre-H4 database migration."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from foxtrot.config.lifecycle import ErrorDetails, ExecutionState as E, TrackingState as T
from foxtrot.persistence.database import DATABASE_SCHEMA_VERSION, _MIGRATIONS, connect_database, initialize_database
from foxtrot.persistence.events import append_job_event, read_job_events
from foxtrot.persistence.jobs import claim_next_job, create_recorded_job, get_job_record
from foxtrot.persistence.lifecycle import transition_execution, transition_tracking
from foxtrot.persistence.workspace import initialize_workspace


class LifecyclePersistenceTests(unittest.TestCase):
    def setUp(self):
        fixture = tempfile.TemporaryDirectory(prefix="foxtrot-h4-events-")
        self.addCleanup(fixture.cleanup)
        self.root = Path(fixture.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.request = {"dataset_id": "demo", "tracking_destination_id": "local"}
        self.job = create_recorded_job(self.workspace, self.checkout, self.request, "first").name
        self.claim = claim_next_job(self.workspace, self.checkout)
        self.reason = ErrorDetails(code="fixture_failed", message="Fixture operation failed")

    def record(self):
        return get_job_record(self.workspace, self.checkout, self.job)

    def events(self, **options):
        return read_job_events(self.workspace, self.checkout, self.job, **options)

    def move(self, target, **options):
        return transition_execution(self.workspace, self.checkout, self.job, target,
                                    claim_token=self.claim.claim_token, **options)

    def track(self, target, **options):
        return transition_tracking(self.workspace, self.checkout, self.job, target, **options)

    def append(self, payload=None, **options):
        return append_job_event(self.workspace, self.checkout, self.job, "progress",
                                {"epoch": 1} if payload is None else payload,
                                claim_token=self.claim.claim_token, **options)

    def test_submission_and_claim_commit_their_own_ordered_events(self):
        events = self.events()
        self.assertEqual([event.sequence for event in events], [1, 2])
        self.assertEqual([event.payload["state"] for event in events], ["queued", "preparing"])
        self.assertEqual(self.record().owner_pid, os.getpid())
        self.assertIsNotNone(self.record().owner_start_id)
        self.assertEqual(create_recorded_job(self.workspace, self.checkout, self.request, "first").name, self.job)
        self.assertEqual(self.record().last_event_sequence, 2)

    def test_finished_execution_can_have_failed_tracking_and_retry(self):
        for state in (E.TRAINING, E.EVALUATING, E.FINISHED):
            self.move(state)
        self.track(T.SYNCING)
        failed = self.track(T.FAILED, error=self.reason)
        self.assertEqual(failed.execution_state, "finished")
        self.assertIsNone(failed.error_details)
        self.assertEqual(failed.tracking_error, self.reason)
        self.assertEqual(self.track(T.RETRYING).tracking_error, self.reason)
        self.assertIsNone(self.track(T.SYNCING).tracking_error)
        complete = self.track(T.SYNCED)
        self.assertEqual((complete.execution_state, complete.tracking_state), ("finished", "synced"))
        history = self.events()
        self.assertTrue(any(event.payload.get("error") == self.reason.model_dump() for event in history))
        with self.assertRaises(ValueError):
            self.move(E.TRAINING)

    def test_execution_failures_require_reasons_and_preserve_tracking(self):
        before = self.record()
        for state in (E.FAILED, E.INTERRUPTED):
            with self.assertRaises(ValueError):
                self.move(state)
        with self.assertRaises(ValueError):
            self.move(E.TRAINING, error=self.reason)
        self.assertEqual(self.record(), before)
        failed = self.move(E.FAILED, error=self.reason)
        self.assertEqual(failed.error_details, self.reason)
        self.assertEqual(failed.tracking_state, "pending")
        self.assertEqual(self.events()[-1].payload["error"]["code"], "fixture_failed")

    def test_cancelled_is_terminal_but_does_not_fake_upload_completion(self):
        record = self.move(E.CANCELLED)
        self.assertEqual((record.execution_state, record.tracking_state), ("cancelled", "pending"))
        with self.assertRaises(ValueError):
            self.move(E.PREPARING)
        with self.assertRaises(ValueError):
            self.append()

    def test_repeated_transition_is_noop_and_cannot_replace_reason(self):
        initial = self.record()
        self.assertEqual(self.move(E.PREPARING), initial)
        self.move(E.FAILED, error=self.reason)
        before = self.record()
        self.assertEqual(self.move(E.FAILED, error=self.reason), before)
        with self.assertRaises(ValueError):
            self.move(E.FAILED, error=ErrorDetails(code="different", message="Changed"))
        self.assertEqual(self.record(), before)
        self.track(T.SYNCING)
        self.track(T.FAILED, error=self.reason)
        before = self.record()
        self.assertEqual(self.track(T.FAILED), before)
        with self.assertRaises(ValueError):
            self.track(T.FAILED, error=ErrorDetails(code="different", message="Changed"))

    def test_invalid_tracking_changes_leave_state_and_events_unchanged(self):
        before = self.record()
        with self.assertRaises(ValueError):
            self.track(T.SYNCED)
        with self.assertRaises(ValueError):
            self.track(T.SYNCING, error=self.reason)
        self.assertEqual(self.record(), before)
        self.track(T.SYNCING)
        before = self.record()
        with self.assertRaises(ValueError):
            self.track(T.FAILED)
        self.assertEqual(self.record(), before)

    def test_claim_token_and_owner_both_required(self):
        for token in (None, "wrong-token"):
            with self.assertRaises(ValueError):
                transition_execution(self.workspace, self.checkout, self.job, E.TRAINING, claim_token=token)
        with patch("foxtrot.persistence.events.capture_process_identity") as capture:
            capture.return_value.pid = os.getpid() + 1
            capture.return_value.start_id = self.record().owner_start_id
            with self.assertRaises(ValueError):
                self.move(E.TRAINING)
            with self.assertRaises(ValueError):
                self.append()
        self.assertEqual(self.record().last_event_sequence, 2)

    def test_queued_job_cannot_skip_claim_or_fake_progress(self):
        second = create_recorded_job(self.workspace, self.checkout, self.request, "second").name
        with self.assertRaisesRegex(ValueError, "claim_next_job"):
            transition_execution(self.workspace, self.checkout, second, E.PREPARING)
        with self.assertRaises(ValueError):
            append_job_event(self.workspace, self.checkout, second, "progress", {}, claim_token="wrong")
        cancelled = transition_execution(self.workspace, self.checkout, second, E.CANCELLED)
        self.assertEqual(cancelled.execution_state, "cancelled")

    def test_event_pages_recover_exactly_missed_updates(self):
        for epoch in range(3):
            self.append({"epoch": epoch})
        first = self.events(limit=2)
        rest = self.events(after_sequence=first[-1].sequence)
        self.assertEqual([event.sequence for event in rest], [3, 4, 5])
        self.assertEqual([event.payload["epoch"] for event in rest], [0, 1, 2])
        self.assertEqual(self.events(after_sequence=5), ())
        self.assertEqual(self.events(after_sequence=100), ())

    def test_concurrent_events_receive_contiguous_sequences(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            events = list(pool.map(lambda i: self.append({"epoch": i}), range(12)))
        self.assertEqual(sorted(event.sequence for event in events), list(range(3, 15)))
        self.assertEqual([event.sequence for event in self.events()], list(range(1, 15)))
        self.assertEqual(self.record().last_event_sequence, 14)

    def test_event_key_replay_does_not_duplicate_or_change_saved_update(self):
        saved = self.append(event_key="epoch:1")
        self.assertEqual(self.append(event_key="epoch:1"), saved)
        with self.assertRaises(ValueError):
            self.append({"epoch": 2}, event_key="epoch:1")
        self.assertEqual(self.record().last_event_sequence, 3)
        for key in ("system:claimed", " ", 42):
            with self.assertRaises(ValueError):
                self.append(event_key=key)

    def test_payload_is_a_saved_snapshot(self):
        payload = {"nested": {"loss": 0.5}}
        event = self.append(payload)
        payload["nested"]["loss"] = 9
        event.payload["nested"]["loss"] = 8
        self.assertEqual(self.events()[-1].payload["nested"]["loss"], 0.5)

    def test_invalid_payload_or_type_never_advances_sequence(self):
        for payload in ({"loss": float("nan")}, {"nested": [float("inf")]}, {"bad": object()}):
            with self.assertRaises(ValidationError):
                self.append(payload)
        with self.assertRaises(ValueError):
            append_job_event(self.workspace, self.checkout, self.job, "state_changed", {}, claim_token=self.claim.claim_token)
        self.assertEqual(self.record().last_event_sequence, 2)

    def test_cursor_limits_and_unknown_jobs_fail_clearly(self):
        for value in (-1, True, 1.2, "1"):
            with self.assertRaises(ValueError):
                self.events(after_sequence=value)
        for value in (0, 1001, True, "5"):
            with self.assertRaises(ValueError):
                self.events(limit=value)
        with self.assertRaises(KeyError):
            read_job_events(self.workspace, self.checkout, "missing")

    def test_state_and_event_write_roll_back_together(self):
        before = self.record()
        with patch("foxtrot.persistence.lifecycle._append_event", side_effect=sqlite3.OperationalError("fixture full disk")):
            with self.assertRaises(sqlite3.OperationalError):
                self.move(E.TRAINING)
            with self.assertRaises(sqlite3.OperationalError):
                self.track(T.SYNCING)
        self.assertEqual(self.record(), before)
        self.assertEqual(len(self.events()), 2)

    def test_submission_and_claim_roll_back_if_event_cannot_be_saved(self):
        with patch("foxtrot.persistence.jobs._append_event", side_effect=sqlite3.OperationalError("fixture write error")):
            with self.assertRaises(sqlite3.OperationalError):
                create_recorded_job(self.workspace, self.checkout, self.request, "failed-submit")
        self.move(E.CANCELLED)
        second = create_recorded_job(self.workspace, self.checkout, self.request, "second").name
        with patch("foxtrot.persistence.jobs._append_event", side_effect=sqlite3.OperationalError("fixture write error")):
            with self.assertRaises(sqlite3.OperationalError):
                claim_next_job(self.workspace, self.checkout)
        record = get_job_record(self.workspace, self.checkout, second)
        self.assertEqual(record.execution_state, "queued")
        self.assertIsNone(record.claim_token)
        self.assertEqual(record.last_event_sequence, 1)

    def test_state_and_history_survive_fresh_python_process(self):
        self.move(E.TRAINING)
        self.append({"epoch": 2, "loss": 0.25})
        script = """
import json, sys
from foxtrot.persistence.events import read_job_events
from foxtrot.persistence.jobs import get_job_record
w, s, j = sys.argv[1:]
record = get_job_record(w, s, j)
events = read_job_events(w, s, j, after_sequence=2)
print(json.dumps({'state': record.execution_state, 'cursor': record.last_event_sequence,
                  'events': [e.model_dump(mode='json') for e in events]}))
"""
        completed = subprocess.run([sys.executable, "-B", "-c", script, str(self.workspace), str(self.checkout), self.job],
                                   cwd=self.root, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        saved = json.loads(completed.stdout)
        self.assertEqual((saved["state"], saved["cursor"]), ("training", 4))
        self.assertEqual([event["sequence"] for event in saved["events"]], [3, 4])
        self.assertEqual(saved["events"][-1]["payload"], {"epoch": 2, "loss": 0.25})


class EventMigrationTests(unittest.TestCase):
    def test_failed_version_four_migration_preserves_version_three_schema(self):
        with tempfile.TemporaryDirectory(prefix="foxtrot-h4-rollback-") as temporary:
            root = Path(temporary).resolve()
            checkout = root / "checkout"
            checkout.mkdir()
            workspace = initialize_workspace(root / "workspace", checkout)
            path = workspace / "app.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    for version in range(1, 4):
                        for statement in _MIGRATIONS[version]:
                            connection.execute(statement)
                    connection.execute("PRAGMA user_version = 3")
                    connection.execute("INSERT INTO jobs (job_id, submission_key) VALUES ('legacy', 'legacy-key')")
            with patch("foxtrot.persistence.database._seed_legacy_events", side_effect=sqlite3.DatabaseError("fixture migration failure")):
                with self.assertRaises(sqlite3.DatabaseError):
                    initialize_database(workspace, checkout)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
                self.assertEqual(connection.execute("SELECT job_id FROM jobs").fetchone()[0], "legacy")
                columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
                self.assertNotIn("owner_pid", columns)
                self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name = 'job_events'").fetchone())

    def test_version_three_import_preserves_cursor_and_truthfully_marks_missing_history(self):
        with tempfile.TemporaryDirectory(prefix="foxtrot-h4-migrate-") as temporary:
            root = Path(temporary).resolve()
            checkout = root / "checkout"
            checkout.mkdir()
            workspace = initialize_workspace(root / "workspace", checkout)
            path = workspace / "app.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    for version in range(1, 4):
                        for statement in _MIGRATIONS[version]:
                            connection.execute(statement)
                    connection.execute("PRAGMA user_version = 3")
                    connection.execute("INSERT INTO jobs (job_id, submission_key, execution_state, last_event_sequence) VALUES ('legacy', 'legacy-key', 'training', 7)")
            initialize_database(workspace, checkout)
            record = get_job_record(workspace, checkout, "legacy")
            self.assertEqual(record.execution_state, "training")
            self.assertEqual(record.last_event_sequence, 8)
            self.assertIsNone(record.owner_pid)
            event, = read_job_events(workspace, checkout, "legacy")
            self.assertEqual(event.sequence, 8)
            self.assertEqual(event.payload["prior_history_available"], False)
            initialize_database(workspace, checkout)
            self.assertEqual(len(read_job_events(workspace, checkout, "legacy")), 1)
            with connect_database(workspace, checkout) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], DATABASE_SCHEMA_VERSION)
