"""H6 run associations, durable upload outbox, and resumable synchronization."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from foxtrot.config.lifecycle import ExecutionState as E
from foxtrot.config.tracking import TrackingOperation
from foxtrot.persistence.artifacts import record_job_artifact
from foxtrot.persistence.database import DATABASE_SCHEMA_VERSION, _MIGRATIONS, initialize_database
from foxtrot.persistence.jobs import claim_next_job, create_recorded_job, get_job_record
from foxtrot.persistence.lifecycle import transition_execution
from foxtrot.persistence.tracking import (
    enqueue_tracking, get_run_association, list_tracking, record_run_association,
)
from foxtrot.persistence.workspace import initialize_workspace
from foxtrot.tracking.fixture import RecordingTrackingClient
from foxtrot.tracking.plan import JOB_TAG
from foxtrot.tracking.uploader import sync_job_tracking


def op(kind, **payload):
    return TrackingOperation(operation=kind, payload=payload)


class TrackingOutboxTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-h6-tracking-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"

    def finished_job(self):
        job = create_recorded_job(self.workspace, self.checkout,
                                  {"dataset_id": "shapes", "tracking_destination_id": "team"}, "key").name
        claim = claim_next_job(self.workspace, self.checkout)
        for state in (E.TRAINING, E.EVALUATING, E.FINISHED):
            transition_execution(self.workspace, self.checkout, job, state, claim_token=claim.claim_token)
        (claim.directory / "result.json").write_text('{"fixture": true}')
        record_job_artifact(self.workspace, self.checkout, job, "job-result", "result.json")
        return job, claim.directory

    def plan(self, job):
        return [
            op("create_run", run_name="foxtrot-run", tags={JOB_TAG: job}),
            op("log_params", params={"training.epochs": "3"}),
            op("log_metrics", metrics=[{"key": "eval/mAP50-95", "value": 0.25, "step": 3}]),
            op("log_artifact", name="job-result", artifact_path="results"),
            op("set_terminated", status="FINISHED"),
        ]

    def test_migration_to_six_preserves_jobs(self):
        root = initialize_workspace(self.workspace, self.checkout)
        with closing(sqlite3.connect(root / "app.sqlite")) as connection:
            with connection:
                for version in range(1, 6):
                    for statement in _MIGRATIONS[version]:
                        connection.execute(statement)
                connection.execute("INSERT INTO jobs (job_id, submission_key, execution_state) VALUES ('legacy', 'k', 'finished')")
                connection.execute("PRAGMA user_version = 5")
        initialize_database(root, self.checkout)
        self.assertEqual(get_job_record(root, self.checkout, "legacy").execution_state, "finished")
        self.assertIsNone(get_run_association(root, self.checkout, "legacy"))
        self.assertEqual(DATABASE_SCHEMA_VERSION, 6)

    def test_operations_validate_their_payloads(self):
        for kind, payload in (("create_run", {"run_name": ""}), ("log_params", {"params": {"a": 1}}),
                              ("log_metrics", {"metrics": [{"key": "x", "value": float("inf"), "step": 0}]}),
                              ("log_artifact", {"name": "a", "artifact_path": "../x"}),
                              ("set_terminated", {"status": "DONE"}), ("delete_run", {})):
            with self.assertRaises(ValueError):
                TrackingOperation(operation=kind, payload=payload)

    def test_enqueue_requires_final_job_valid_plan_and_is_idempotent(self):
        job = create_recorded_job(self.workspace, self.checkout,
                                  {"dataset_id": "shapes", "tracking_destination_id": "team"}, "key").name
        with self.assertRaisesRegex(ValueError, "final outcome"):
            enqueue_tracking(self.workspace, self.checkout, job, self.plan(job))
        job, _ = self.finished_job()  # same idempotency key: the queued job is claimed and finished
        with self.assertRaises(ValueError):
            enqueue_tracking(self.workspace, self.checkout, job, self.plan(job)[1:])
        first = enqueue_tracking(self.workspace, self.checkout, job, self.plan(job))
        self.assertEqual(enqueue_tracking(self.workspace, self.checkout, job, self.plan(job)), first)
        changed = self.plan(job)
        changed[-1] = op("set_terminated", status="FAILED")
        with self.assertRaisesRegex(ValueError, "different tracking plan"):
            enqueue_tracking(self.workspace, self.checkout, job, changed)

    def test_sync_uploads_in_order_and_records_association(self):
        job, directory = self.finished_job()
        enqueue_tracking(self.workspace, self.checkout, job, self.plan(job))
        self.assertEqual(get_job_record(self.workspace, self.checkout, job).tracking_state, "pending")
        client = RecordingTrackingClient()
        record = sync_job_tracking(self.workspace, self.checkout, job, client)
        self.assertEqual(record.tracking_state, "synced")
        association = get_run_association(self.workspace, self.checkout, job)
        self.assertEqual(association.destination_id, "team")
        run = client.runs[association.run_id]
        self.assertTrue(association.run_url.endswith(association.run_id))
        self.assertEqual(run["params"], {"training.epochs": "3"})
        self.assertEqual(run["metrics"], [("eval/mAP50-95", 0.25, 3)])
        self.assertEqual(run["artifacts"][0][:2], ("results", "result.json"))
        self.assertEqual(run["status"], "FINISHED")
        self.assertFalse(list_tracking(self.workspace, self.checkout, job, pending_only=True))
        # Synced is final: another sync makes no service calls.
        calls = list(client.calls)
        sync_job_tracking(self.workspace, self.checkout, job, client)
        self.assertEqual(client.calls, calls)

    def test_outage_keeps_local_result_and_retry_resumes_without_duplicates(self):
        job, _ = self.finished_job()
        enqueue_tracking(self.workspace, self.checkout, job, self.plan(job))
        client = RecordingTrackingClient(fail_once={"log_metrics"})
        record = sync_job_tracking(self.workspace, self.checkout, job, client)
        self.assertEqual((record.execution_state, record.tracking_state), ("finished", "retrying"))
        self.assertEqual(record.tracking_error.code, "tracking_upload_failed")
        self.assertNotIn("outage", record.tracking_error.message)  # exception text is not saved
        pending = list_tracking(self.workspace, self.checkout, job, pending_only=True)
        self.assertEqual([e.operation.operation for e in pending][0], "log_metrics")
        self.assertEqual(pending[0].attempts, 1)
        record = sync_job_tracking(self.workspace, self.checkout, job, client)
        self.assertEqual(record.tracking_state, "synced")
        self.assertEqual(client.calls.count("create_run"), 1)
        self.assertEqual(client.calls.count("log_params"), 1)
        self.assertEqual(len(client.runs), 1)

    def test_ambiguous_run_creation_reuses_the_tagged_run(self):
        job, _ = self.finished_job()
        enqueue_tracking(self.workspace, self.checkout, job, self.plan(job))
        client = RecordingTrackingClient()
        existing = client.create_run(job, "earlier-attempt", {JOB_TAG: job})  # created, but never recorded
        sync_job_tracking(self.workspace, self.checkout, job, client)
        self.assertEqual(get_run_association(self.workspace, self.checkout, job).run_id, existing.run_id)
        self.assertEqual(len(client.runs), 1)

    def test_changed_artifact_is_never_uploaded(self):
        job, directory = self.finished_job()
        enqueue_tracking(self.workspace, self.checkout, job, self.plan(job))
        (directory / "result.json").write_text('{"changed": true}')
        client = RecordingTrackingClient()
        record = sync_job_tracking(self.workspace, self.checkout, job, client)
        self.assertEqual(record.tracking_state, "retrying")
        self.assertNotIn("log_artifact", client.calls)

    def test_run_association_cannot_be_replaced(self):
        job, _ = self.finished_job()
        record_run_association(self.workspace, self.checkout, job, destination_id="team", run_id="one", run_url=None)
        record_run_association(self.workspace, self.checkout, job, destination_id="team", run_id="one", run_url=None)
        with self.assertRaises(ValueError):
            record_run_association(self.workspace, self.checkout, job, destination_id="team", run_id="two", run_url=None)


if __name__ == "__main__":
    unittest.main()
