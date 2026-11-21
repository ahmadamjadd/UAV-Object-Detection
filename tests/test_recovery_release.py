"""V7 recovery (resume, upload errors, crash boundaries, recovery list) and V8 release
features (safe cleanup, member attribution, setup check)."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from PIL import Image, ImageDraw

from foxtrot.config.job import JobSpec


# Hash every file under a folder (to prove an old job folder is untouched).
def tree_hash(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(folder).as_posix().encode() + path.read_bytes())
    return digest.hexdigest()


# A workspace with a tiny 2-class training dataset and helpers to run jobs.
class RecoveryCase(unittest.TestCase):
    # Before each test: workspace + 12-image dataset.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v7-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        source = self.root / "squares"
        for split, count in (("train", 8), ("val", 4)):
            (source / split / "images").mkdir(parents=True)
            (source / split / "labels").mkdir(parents=True)
            for index in range(count):
                image = Image.new("RGB", (64, 64), (30, 30, 30))
                ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill=(200, 50 + index * 10, 50))
                image.save(source / split / "images" / f"{index}.jpg")
                (source / split / "labels" / f"{index}.txt").write_text(f"{index % 2} 0.5 0.5 0.5 0.5\n")
        from foxtrot.persistence.dataset_import import import_dataset
        import_dataset(self.workspace, self.checkout, source=source, dataset_id="squares", class_names=("R", "O"))

    # A worker for the given adapter (real YOLO uses 0 data-loading workers in tests).
    def worker(self, adapter="fixture", **settings):
        from foxtrot.execution.worker import Worker
        from foxtrot.pipeline.registry import PipelineSettings
        options = ({"adapter": "fixture", "fixture_epoch_seconds": 0.0} if adapter == "fixture"
                   else {"adapter": "ultralytics", "ultralytics_workers": 0})
        return Worker(self.workspace, self.checkout, runtime_directory=self.root / "runtime",
                      pipeline=PipelineSettings(**{**options, **settings}))

    # Queue one job (validated like the website does) and return its ID.
    def submit(self, key, **request):
        from foxtrot.persistence.jobs import create_recorded_job
        from foxtrot.persistence.start_checkpoints import check_start_checkpoint
        spec = {"dataset_id": "squares", "tracking_destination_id": "local", **request}
        check_start_checkpoint(self.workspace, self.checkout, JobSpec.model_validate(spec))
        return create_recorded_job(self.workspace, self.checkout, spec, key).name

    # The job's record.
    def record(self, job):
        from foxtrot.persistence.jobs import get_job_record
        return get_job_record(self.workspace, self.checkout, job)


# V7: resuming interrupted training from its last.pt.
class ResumeTests(RecoveryCase):
    # Cancel after epoch 1, resume: continues at epoch 2 in a new job; the old job is untouched.
    def test_cancelled_training_resumes_with_provenance(self):
        from foxtrot.persistence.commands import request_cancel
        from foxtrot.persistence.events import read_job_events
        from foxtrot.persistence.start_checkpoints import StartCheckpointError
        from foxtrot.persistence.tracking import list_tracking
        training = {"epochs": 6, "batch_size": 4, "image_size": 64, "seed": 1}
        first = self.submit("first", initialization="scratch", training=training)
        worker = self.worker("ultralytics")
        thread = threading.Thread(target=worker.run, kwargs={"once": True})
        thread.start()
        deadline = time.time() + 300
        while time.time() < deadline:  # cancel as soon as epoch 1 is reported (last.pt then exists)
            if any(e.event_type == "progress" and e.payload.get("epoch") == 1
                   for e in read_job_events(self.workspace, self.checkout, first)):
                request_cancel(self.workspace, self.checkout, first)
                break
            time.sleep(0.2)
        thread.join(300)
        self.assertEqual(self.record(first).execution_state, "cancelled")
        reached = max(e.payload["epoch"] for e in read_job_events(self.workspace, self.checkout, first)
                      if e.event_type == "progress")  # last.pt holds this epoch
        old_folder = self.workspace / "jobs" / first
        last = old_folder / "output" / "train" / "weights" / "last.pt"
        self.assertTrue(last.is_file())
        before = tree_hash(old_folder / "output")
        last_sha = hashlib.sha256(last.read_bytes()).hexdigest()

        with self.assertRaises(StartCheckpointError) as raised:  # settings other than runtime ones may not change
            self.submit("bad", initialization="resume", checkpoint_job_id=first, training={**training, "epochs": 8})
        self.assertIn("epochs", str(raised.exception))  # e.g. epochs 6 -> 8 is refused
        resumed = self.submit("resume", initialization="resume", checkpoint_job_id=first,
                              training={**training, "patience": 50})
        self.worker("ultralytics").run(once=True)
        record = self.record(resumed)
        self.assertEqual(record.execution_state, "finished", record.error_details)
        epochs = [e.payload["epoch"] for e in read_job_events(self.workspace, self.checkout, resumed)
                  if e.event_type == "progress"]
        self.assertEqual(epochs[0], reached + 1)  # continues right after the saved epoch
        self.assertEqual(epochs[-1], 6)
        self.assertEqual(tree_hash(old_folder / "output"), before)  # nothing written into the old job
        params = next(e.operation for e in list_tracking(self.workspace, self.checkout, resumed)
                      if e.operation.operation == "log_params").payload["params"]
        self.assertEqual((params["model.initialization"], params["model.start_weights_sha256"],
                          params["checkpoint_job_id"]), ("resume", last_sha, first))

    # Finished jobs and jobs without last.pt cannot be resumed; the reason is clear.
    def test_resume_refusals(self):
        from foxtrot.persistence.start_checkpoints import StartCheckpointError
        finished = self.submit("done", training={"epochs": 1})
        self.worker().run(once=True)
        failed = self.submit("fail", training={"epochs": 2})
        self.worker(fixture_fail_at_epoch=1).run(once=True)
        for job, message in ((finished, "only interrupted"), (failed, "no last.pt")):
            with self.subTest(message=message), self.assertRaises(StartCheckpointError) as raised:
                self.submit(f"resume-{message}", initialization="resume", checkpoint_job_id=job,
                            training={"epochs": 1 if job == finished else 2})
            self.assertIn(message, str(raised.exception))


# V7: upload errors named precisely, without exposing exception text.
class UploadErrorTests(unittest.TestCase):
    # Credentials, quota and connectivity are told apart; secrets in the error never appear.
    def test_classification(self):
        from foxtrot.tracking.uploader import classify_tracking_error

        class HttpError(Exception):
            def __init__(self, status, text):
                super().__init__(text)
                self.response = type("Response", (), {"status_code": status})()

        cases = [(HttpError(401, "token=SECRET123 rejected"), "tracking_credentials"),
                 (HttpError(429, "slow down"), "tracking_quota"),
                 (Exception("storage limit reached for SECRET123"), "tracking_quota"),
                 (ConnectionError("https://user:SECRET123@host"), "tracking_unreachable"),
                 (ValueError("odd"), "tracking_upload_failed")]
        for error, code in cases:
            details = classify_tracking_error(error)
            self.assertEqual(details.code, code)
            self.assertNotIn("SECRET123", details.message)


# V7: study and job state reconciled across a crash at each boundary.
class StudyCrashTests(RecoveryCase):
    # Crash after training finished but before the study recorded it: settled once on restart.
    def test_finished_trial_settles_once_after_restart(self):
        import optuna
        from foxtrot.config.study import StudySpec
        from foxtrot.studies.coordinator import StudyCoordinator
        from foxtrot.studies.store import create_study, read_trials, write_trial
        study = create_study(self.workspace, self.checkout, StudySpec.model_validate({
            "name": "c", "dataset_id": "squares", "tracking_destination_id": "local", "trials_per_model": 1,
            "base_training": {"epochs": 1}, "search_space": [{"name": "patience", "type": "int", "low": 2, "high": 9}]}))
        coordinator = StudyCoordinator(self.workspace, self.checkout)
        coordinator.step()
        coordinator.step()                        # trial saved and its job created
        self.worker(fixture_config_scores=True).run(once=True)   # training finishes...
        coordinator.stop()                        # ...and the coordinator "crashes" before noticing
        restarted = StudyCoordinator(self.workspace, self.checkout)
        self.addCleanup(restarted.stop)
        restarted.step()
        trial = read_trials(self.workspace, self.checkout, study)[0]
        self.assertEqual(trial["state"], "complete")
        # A second crash after Optuna was told but before the file was written must not tell twice.
        trial.update(state="running")
        write_trial(self.workspace, self.checkout, study, trial)
        restarted.step()
        storage = f"sqlite:///{self.workspace / 'studies' / study / 'optuna.db'}"
        optuna_study = optuna.load_study(study_name="yolov8n", storage=storage)
        self.assertEqual([t.state for t in optuna_study.trials], [optuna.trial.TrialState.COMPLETE])
        self.assertEqual(read_trials(self.workspace, self.checkout, study)[0]["state"], "complete")
        optuna_study._storage.engine.dispose() if hasattr(optuna_study._storage, "engine") else None


# V7: the recovery list; V8: cleanup, member attribution, setup check.
class RecoveryAndReleaseTests(RecoveryCase):
    # The recovery API lists stopped training (with resumability) and pending uploads.
    def test_recovery_listing(self):
        from foxtrot.api.app import create_app
        from test_api import call
        self.submit("ok", training={"epochs": 1})
        self.worker().run(once=True)
        failed = self.submit("fail", training={"epochs": 2})
        self.worker(fixture_fail_at_epoch=1).run(once=True)
        app = create_app(self.workspace, self.checkout, port=8765, session_token="test-session-token", static_directory=None)
        status, content, _ = call(app, "GET", "/api/recovery")
        report = json.loads(content)
        self.assertEqual(status, 200)
        stopped = {item["job_id"]: item for item in report["interrupted"]}
        self.assertFalse(stopped[failed]["resumable"])
        self.assertIn("last.pt", stopped[failed]["resume_reason"])
        self.assertEqual(stopped[failed]["request"]["dataset_id"], "squares")
        self.assertEqual(len(report["uploads"]), 2)  # nothing was uploaded in this test

    # Cleanup offers only safe items, keeps registered artifacts, and re-checks before deleting.
    def test_safe_cleanup(self):
        from foxtrot.persistence.artifacts import get_job_artifact
        from foxtrot.persistence.cleanup import apply_cleanup, plan_cleanup
        from foxtrot.persistence.lifecycle import transition_tracking
        from foxtrot.config.lifecycle import TrackingState
        done = self.submit("ok", training={"epochs": 1})
        self.worker().run(once=True)
        unsynced = self.submit("ok2", training={"epochs": 1, "seed": 2})
        self.worker().run(once=True)
        extra = self.workspace / "jobs" / done / "output" / "scratch.bin"
        extra.write_bytes(b"x" * 1000)
        (self.workspace / "jobs" / unsynced / "output" / "scratch.bin").write_bytes(b"y" * 10)
        self.assertEqual(plan_cleanup(self.workspace, self.checkout), [])  # nothing uploaded yet: nothing offered
        for state in (TrackingState.SYNCING, TrackingState.SYNCED):
            transition_tracking(self.workspace, self.checkout, done, state)
        plan = plan_cleanup(self.workspace, self.checkout)
        self.assertEqual([item["id"] for item in plan], [f"extras:{done}"])
        result = apply_cleanup(self.workspace, self.checkout, [f"extras:{done}", f"extras:{unsynced}"])
        self.assertEqual((result["removed"], result["skipped"]), ([f"extras:{done}"], [f"extras:{unsynced}"]))
        self.assertFalse(extra.exists())
        get_job_artifact(self.workspace, self.checkout, done, "checkpoint")  # still present and verified

    # The member name tags every run; the setup check reports a ready installation.
    def test_member_tag_and_system_check(self):
        from foxtrot.api.system import system_check
        from foxtrot.persistence.profile import save_profile
        from foxtrot.persistence.tracking import list_tracking
        with self.assertRaises(ValueError):
            save_profile(self.workspace, self.checkout, "bad/name")
        save_profile(self.workspace, self.checkout, "Member A")
        job = self.submit("tagged", training={"epochs": 1})
        self.worker().run(once=True)
        create = list_tracking(self.workspace, self.checkout, job)[0].operation
        self.assertEqual(create.payload["tags"]["foxtrot.member"], "Member A")
        report = system_check(self.workspace, self.checkout)
        names = {check["name"]: check["ok"] for check in report["checks"]}
        self.assertTrue(report["ok"])
        self.assertTrue(names["Libraries"] and names["Pinned versions"] and names["Dataset generator"])
        result = subprocess.run([sys.executable, "-B", "-m", "foxtrot.api.launcher", "--workspace", str(self.workspace),
                                 "--source-checkout", str(self.checkout), "--check"], capture_output=True, text=True,
                                timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK  Libraries", result.stdout)


if __name__ == "__main__":
    unittest.main()
