"""H3 acceptance: durable records, reservations, file identities, safe cleanup."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch
from uuid import uuid4

from foxtrot.persistence.artifacts import get_job_artifact, record_job_artifact
from foxtrot.persistence.database import (
    DATABASE_SCHEMA_VERSION, connect_database, initialize_database,
)
from foxtrot.persistence.files import JOB_FORMAT, JOB_MARKER, write_new_text
from foxtrot.persistence.jobs import (
    claim_next_job, cleanup_unregistered_job, create_configured_job,
    create_recorded_job, get_job_record, load_job_configurations,
)
from foxtrot.persistence.workspace import initialize_workspace


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="foxtrot-h3-")
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"

    def request(self, epochs=5):
        return {
            "dataset_id": "dataset-demo",
            "tracking_destination_id": "local-mlflow",
            "training": {"epochs": epochs},
        }

    def submit(self, key="first", epochs=5):
        return create_recorded_job(
            self.workspace, self.checkout, self.request(epochs), key,
        )

    def execute(self, statement, parameters=()):
        with connect_database(self.workspace, self.checkout) as connection:
            with connection:
                return connection.execute(statement, parameters).fetchall()

    def test_empty_queue_returns_none(self):
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))

    def test_claim_persists_token_and_leaves_tracking_independent(self):
        directory = self.submit()
        claim = claim_next_job(self.workspace, self.checkout)
        record = get_job_record(self.workspace, self.checkout, directory.name)
        self.assertEqual(claim.directory, directory)
        self.assertEqual(claim.job_id, directory.name)
        self.assertEqual(record.execution_state, "preparing")
        self.assertEqual(record.tracking_state, "pending")
        self.assertEqual(record.claim_token, claim.claim_token)
        self.assertIsNotNone(record.claimed_at)
        self.assertEqual(record.last_event_sequence, 2)
        self.assertEqual(json.loads(claim.configurations.resolved_json)["training"]["epochs"], 5)
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))

    def test_only_one_job_is_active_even_with_two_queued_jobs(self):
        first = self.submit("first")
        second = self.submit("second", epochs=9)
        # Make the ordering explicit; UUIDs are not a submission-order clock.
        self.execute("UPDATE jobs SET created_at = '2026-01-01' WHERE job_id = ?", (first.name,))
        self.execute("UPDATE jobs SET created_at = '2026-01-02' WHERE job_id = ?", (second.name,))
        self.assertEqual(claim_next_job(self.workspace, self.checkout).job_id, first.name)
        self.assertIsNone(claim_next_job(self.workspace, self.checkout))
        self.assertEqual(get_job_record(self.workspace, self.checkout, second.name).execution_state, "queued")
        # Fixture-only completion. Actual lifecycle updates belong to H4.
        self.execute("UPDATE jobs SET execution_state = 'finished' WHERE job_id = ?", (first.name,))
        self.assertEqual(claim_next_job(self.workspace, self.checkout).job_id, second.name)
        self.assertEqual(get_job_record(self.workspace, self.checkout, first.name).tracking_state, "pending")

    def test_concurrent_claims_reserve_one_job(self):
        first = self.submit("first")
        self.submit("second")
        ready = Barrier(2)

        def claim():
            ready.wait(timeout=10)
            return claim_next_job(self.workspace, self.checkout)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(claim) for _ in range(2)]
            claims = [future.result(timeout=20) for future in futures]
        successful = [item for item in claims if item is not None]
        self.assertEqual(len(successful), 1)
        self.assertEqual(successful[0].job_id, first.name)
        self.assertEqual(len(self.execute("SELECT * FROM jobs WHERE execution_state = 'preparing'")), 1)

    def test_database_constraint_prevents_a_second_active_job(self):
        self.submit("first")
        second = self.submit("second")
        claim_next_job(self.workspace, self.checkout)
        with self.assertRaises(sqlite3.IntegrityError):
            self.execute("UPDATE jobs SET execution_state = 'training' WHERE job_id = ?", (second.name,))
        self.assertEqual(get_job_record(self.workspace, self.checkout, second.name).execution_state, "queued")

    def test_retry_after_claim_preserves_execution_and_manifests(self):
        directory = self.submit()
        before = (directory / "resolved.json").read_bytes()
        claim = claim_next_job(self.workspace, self.checkout)
        self.assertEqual(self.submit(), directory)
        record = get_job_record(self.workspace, self.checkout, directory.name)
        self.assertEqual(record.execution_state, "preparing")
        self.assertEqual(record.claim_token, claim.claim_token)
        self.assertEqual((directory / "resolved.json").read_bytes(), before)

    def test_same_settings_with_a_new_submission_key_create_a_new_job(self):
        first = self.submit("first")
        second = self.submit("second")
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.execute("SELECT * FROM jobs")), 2)

    def test_valid_json_tampering_is_detected_before_claim_or_retry(self):
        directory = self.submit()
        for filename in ("submitted.json", "resolved.json"):
            path = directory / filename
            value = json.loads(path.read_text(encoding="utf-8"))
            value["training"]["epochs"] = 9
            path.write_text(json.dumps(value), encoding="utf-8")
        for operation in (
            lambda: self.submit(),
            lambda: claim_next_job(self.workspace, self.checkout),
            lambda: load_job_configurations(self.workspace, self.checkout, directory.name),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    operation()
        self.assertEqual(get_job_record(self.workspace, self.checkout, directory.name).execution_state, "queued")

    def test_missing_or_symlinked_manifests_cannot_be_claimed(self):
        directory = self.submit()
        path = directory / "resolved.json"
        saved = path.read_bytes()
        path.unlink()
        with self.assertRaises(ValueError):
            claim_next_job(self.workspace, self.checkout)
        external = self.root / "external.json"
        external.write_bytes(saved)
        path.symlink_to(external)
        with self.assertRaises(ValueError):
            claim_next_job(self.workspace, self.checkout)
        self.assertEqual(external.read_bytes(), saved)
        self.assertEqual(get_job_record(self.workspace, self.checkout, directory.name).execution_state, "queued")

    def test_partial_preparation_is_not_queued_and_can_be_cleaned(self):
        def interrupted_write(path, content):
            if path.name == "resolved.json":
                raise OSError("fixture disk failure")
            write_new_text(path, content)

        with patch("foxtrot.persistence.jobs.write_new_text", side_effect=interrupted_write):
            with self.assertRaisesRegex(OSError, "fixture disk failure"):
                self.submit()
        self.assertEqual(len(self.execute("SELECT * FROM jobs")), 0)
        abandoned, = (self.workspace / "jobs").iterdir()
        self.assertEqual((abandoned / JOB_MARKER).read_text(), JOB_FORMAT)
        cleanup_unregistered_job(self.workspace, self.checkout, abandoned.name)
        self.assertFalse(abandoned.exists())
        self.assertTrue(self.submit().is_dir())

    def test_cleanup_refuses_registered_jobs_even_when_finished(self):
        directory = self.submit()
        for state in ("queued", "preparing", "finished", "failed", "interrupted"):
            with self.subTest(state=state):
                self.execute("UPDATE jobs SET execution_state = ? WHERE job_id = ?", (state, directory.name))
                with self.assertRaisesRegex(ValueError, "Registered jobs"):
                    cleanup_unregistered_job(self.workspace, self.checkout, directory.name)
                self.assertTrue((directory / "resolved.json").is_file())

    def test_cleanup_refuses_unowned_folders_and_unsafe_ids(self):
        initialize_database(self.workspace, self.checkout)
        unowned = self.workspace / "jobs" / uuid4().hex
        unowned.mkdir()
        (unowned / "keep.txt").write_text("keep")
        with self.assertRaises(ValueError):
            cleanup_unregistered_job(self.workspace, self.checkout, unowned.name)
        for job_id in ("..", "../checkout", str(self.root), "not-an-id"):
            with self.subTest(job_id=job_id):
                with self.assertRaises(ValueError):
                    cleanup_unregistered_job(self.workspace, self.checkout, job_id)
        self.assertEqual((unowned / "keep.txt").read_text(), "keep")

    def test_cleanup_does_not_follow_job_or_child_symlinks(self):
        initialize_database(self.workspace, self.checkout)
        external = self.root / "external"
        external.mkdir()
        sentinel = external / "keep.txt"
        sentinel.write_text("keep")
        link = self.workspace / "jobs" / uuid4().hex
        link.symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError):
            cleanup_unregistered_job(self.workspace, self.checkout, link.name)
        owned = create_configured_job(self.workspace, self.checkout, self.request())
        (owned / "external-link").symlink_to(external, target_is_directory=True)
        cleanup_unregistered_job(self.workspace, self.checkout, owned.name)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_artifact_exact_path_hash_and_bytes_survive_reopening(self):
        directory = self.submit()
        weights = directory / "weights"
        weights.mkdir()
        checkpoint = weights / "best.pt"
        checkpoint.write_bytes(b"fixture checkpoint, not a model")
        reference = record_job_artifact(
            self.workspace, self.checkout, directory.name, "best", "weights/best.pt", kind="checkpoint",
        )
        self.assertEqual(reference.sha256, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertEqual(reference.size_bytes, checkpoint.stat().st_size)
        self.assertEqual(reference.relative_path, "weights/best.pt")
        self.assertEqual(get_job_artifact(self.workspace, self.checkout, directory.name, "best"), reference)
        self.assertEqual(record_job_artifact(
            self.workspace, self.checkout, directory.name, "best", "weights/best.pt", kind="checkpoint",
        ), reference)

    def test_changed_or_missing_artifact_is_not_silently_replaced(self):
        directory = self.submit()
        path = directory / "result.txt"
        path.write_text("first")
        record_job_artifact(self.workspace, self.checkout, directory.name, "result", "result.txt")
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            get_job_artifact(self.workspace, self.checkout, directory.name, "result")
        with self.assertRaisesRegex(ValueError, "identity cannot change"):
            record_job_artifact(self.workspace, self.checkout, directory.name, "result", "result.txt")
        path.unlink()
        with self.assertRaises(ValueError):
            get_job_artifact(self.workspace, self.checkout, directory.name, "result")

    def test_artifact_paths_cannot_escape_or_reference_managed_configurations(self):
        directory = self.submit()
        external = self.root / "outside.txt"
        external.write_text("outside")
        (directory / "link.txt").symlink_to(external)
        for path in (
            "../outside.txt", str(external), "./resolved.json", "weights//best.pt",
            "resolved.json", "submitted.json", "provenance.json", JOB_MARKER,
            "link.txt", "missing.pt", "C:\\outside.pt", "bad\x00path", "", None, [],
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    record_job_artifact(self.workspace, self.checkout, directory.name, "bad", path)
        self.assertEqual(external.read_text(), "outside")
        self.assertEqual(len(self.execute("SELECT * FROM artifacts")), 0)

    def test_artifact_replaced_during_hashing_is_rejected(self):
        directory = self.submit()
        path = directory / "result.txt"
        path.write_bytes(b"original")
        replacement = directory / "replacement.txt"
        replacement.write_bytes(b"replacement")
        real_digest = hashlib.file_digest

        def replace_during_hash(stream, algorithm):
            digest = real_digest(stream, algorithm)
            replacement.replace(path)
            return digest

        with patch("foxtrot.persistence.files.hashlib.file_digest", side_effect=replace_during_hash):
            with self.assertRaisesRegex(ValueError, "changed while"):
                record_job_artifact(self.workspace, self.checkout, directory.name, "result", "result.txt")
        self.assertEqual(len(self.execute("SELECT * FROM artifacts")), 0)

    def test_artifacts_cannot_traverse_a_directory_symlink(self):
        directory = self.submit()
        external = self.root / "external-weights"
        external.mkdir()
        (external / "best.pt").write_bytes(b"external")
        (directory / "weights").symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError):
            record_job_artifact(self.workspace, self.checkout, directory.name, "best", "weights/best.pt")

    def test_artifact_registration_requires_a_job_and_unique_path(self):
        with self.assertRaises(KeyError):
            record_job_artifact(self.workspace, self.checkout, uuid4().hex, "result", "result.txt")
        directory = self.submit()
        (directory / "result.txt").write_text("result")
        record_job_artifact(self.workspace, self.checkout, directory.name, "result", "result.txt")
        with self.assertRaises(sqlite3.IntegrityError):
            record_job_artifact(self.workspace, self.checkout, directory.name, "alias", "result.txt")
        with self.assertRaises(KeyError):
            get_job_artifact(self.workspace, self.checkout, directory.name, "missing")

    def test_version_two_job_backfills_hashes_without_changing_saved_defaults(self):
        directory = create_configured_job(self.workspace, self.checkout, self.request())
        # These files did not exist before this H3 completion step.
        (directory / JOB_MARKER).unlink()
        (directory / "provenance.json").unlink()
        resolved_path = directory / "resolved.json"
        old_resolved = json.loads(resolved_path.read_text())
        old_resolved["training"]["batch_size"] = 8
        resolved_path.write_text(json.dumps(old_resolved))
        database = self.workspace / "app.sqlite"
        # Reproduce the previous released foundation layout exactly.
        from foxtrot.persistence.database import _MIGRATIONS
        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute(_MIGRATIONS[2][0])
                connection.execute("PRAGMA user_version = 2")
                connection.execute(
                    "INSERT INTO jobs (job_id, submission_key) VALUES (?, ?)", (directory.name, "legacy"),
                )
        saved = load_job_configurations(self.workspace, self.checkout, directory.name)
        self.assertEqual(json.loads(saved.resolved_json)["training"]["batch_size"], 8)
        record = get_job_record(self.workspace, self.checkout, directory.name)
        self.assertIsNotNone(record.submitted_sha256)
        self.assertIsNotNone(record.resolved_sha256)
        self.assertEqual(create_recorded_job(self.workspace, self.checkout, self.request(), "legacy"), directory)

    def test_failed_migration_rolls_back_schema_and_preserves_records(self):
        initialize_workspace(self.workspace, self.checkout)
        database = self.workspace / "app.sqlite"
        from foxtrot.persistence.database import _MIGRATIONS
        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute(_MIGRATIONS[2][0])
                connection.execute("PRAGMA user_version = 2")
                for index in range(2):
                    connection.execute(
                        "INSERT INTO jobs (job_id, submission_key, execution_state) VALUES (?, ?, 'training')",
                        (f"legacy-{index}", f"key-{index}"),
                    )
        with self.assertRaises(sqlite3.IntegrityError):
            initialize_database(self.workspace, self.checkout)
        with closing(sqlite3.connect(database)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 2)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            self.assertNotIn("claim_token", columns)

    def test_database_sidecar_symlinks_are_rejected(self):
        database = initialize_database(self.workspace, self.checkout)
        target = self.root / "keep"
        target.write_bytes(b"keep")
        for suffix in ("-journal", "-wal", "-shm"):
            link = database.with_name(database.name + suffix)
            with self.subTest(suffix=suffix):
                link.symlink_to(target)
                try:
                    with self.assertRaises(ValueError):
                        initialize_database(self.workspace, self.checkout)
                finally:
                    link.unlink()
        self.assertEqual(target.read_bytes(), b"keep")

    def test_app_connections_enforce_artifact_foreign_keys(self):
        with connect_database(self.workspace, self.checkout) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                with connection:
                    connection.execute(
                        """INSERT INTO artifacts (job_id, name, kind, relative_path, sha256, size_bytes)
                           VALUES (?, 'orphan', 'artifact', 'orphan.txt', ?, 1)""",
                        (uuid4().hex, "0" * 64),
                    )

    def test_process_exit_during_a_transaction_does_not_publish_a_claim(self):
        directory = self.submit()
        script = """
import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1])
connection.execute('BEGIN IMMEDIATE')
connection.execute("UPDATE jobs SET execution_state = 'preparing' WHERE job_id = ?", (sys.argv[2],))
os._exit(23)
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script, str(self.workspace / "app.sqlite"), directory.name],
            cwd=self.root, capture_output=True, timeout=15,
        )
        self.assertEqual(completed.returncode, 23)
        record = get_job_record(self.workspace, self.checkout, directory.name)
        self.assertEqual(record.execution_state, "queued")
        self.assertIsNone(record.claim_token)
        self.assertEqual(claim_next_job(self.workspace, self.checkout).job_id, directory.name)

    def test_managed_workspace_directories_reject_symlinks(self):
        initialize_workspace(self.workspace, self.checkout)
        for name in ("settings", "models", "datasets", "tracking-outbox"):
            path = self.workspace / name
            path.rmdir()
            path.symlink_to(self.checkout, target_is_directory=True)
            with self.subTest(name=name):
                try:
                    with self.assertRaises(ValueError):
                        initialize_workspace(self.workspace, self.checkout)
                finally:
                    path.unlink()
                    path.mkdir()

    def test_unknown_job_is_not_created_by_read(self):
        with self.assertRaises(KeyError):
            get_job_record(self.workspace, self.checkout, uuid4().hex)
        self.assertEqual(len(self.execute("SELECT * FROM jobs")), 0)

    def test_two_jobs_artifacts_and_claim_survive_a_fresh_python_process(self):
        first = self.submit("first")
        second = self.submit("second", epochs=9)
        for directory, value in ((first, b"first checkpoint"), (second, b"second checkpoint")):
            (directory / "best.pt").write_bytes(value)
            record_job_artifact(
                self.workspace, self.checkout, directory.name, "best", "best.pt", kind="checkpoint",
            )
        claim = claim_next_job(self.workspace, self.checkout)
        script = """
import json, sys
from foxtrot.persistence.jobs import (create_recorded_job, get_job_record,
    load_job_configurations, claim_next_job)
from foxtrot.persistence.artifacts import get_job_artifact
w, s, first, second = sys.argv[1:]
request = {'dataset_id': 'dataset-demo', 'tracking_destination_id': 'local-mlflow',
           'training': {'epochs': 5}}
assert create_recorded_job(w, s, request, 'first').name == first
assert json.loads(load_job_configurations(w, s, first).resolved_json)['training']['epochs'] == 5
assert json.loads(load_job_configurations(w, s, second).resolved_json)['training']['epochs'] == 9
assert get_job_artifact(w, s, first, 'best').sha256 != get_job_artifact(w, s, second, 'best').sha256
assert claim_next_job(w, s) is None
record = get_job_record(w, s, first)
print(json.dumps({'job_id': record.job_id, 'claim_token': record.claim_token,
                  'execution_state': record.execution_state}))
"""
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script, str(self.workspace), str(self.checkout), first.name, second.name],
            cwd=self.root, env=environment, text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        restored = json.loads(completed.stdout)
        self.assertEqual(restored["claim_token"], claim.claim_token)
        self.assertEqual(restored["execution_state"], "preparing")


if __name__ == "__main__":
    unittest.main()
