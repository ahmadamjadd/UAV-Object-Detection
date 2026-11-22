"""V5/V6 tuning studies: definitions, grid/random/TPE trials, model budgets, stop/resume,
failures, pruning, MLflow parent/child runs, recipe studies with frozen validation, repeats.

Trials run as real jobs through the fixture worker; its opt-in settings-based scores make
trials rank differently, so best-trial selection and pruning can be checked quickly.
"""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from PIL import Image, ImageDraw
from pydantic import ValidationError

from foxtrot.config.study import StudySpec, grid_points, planned_trials

BASE = {"name": "s", "dataset_id": "squares", "tracking_destination_id": "local", "base_training": {"epochs": 2}}


# Definition rules that need no workspace.
class StudySpecTests(unittest.TestCase):
    # Conditional grids skip inactive values; equal budgets multiply by the number of models.
    def test_grid_with_condition_and_budgets(self):
        spec = StudySpec(**BASE, sampler="grid", model_ids=["yolov8n", "yolo11n"], search_space=[
            {"name": "optimizer", "type": "categorical", "choices": ["SGD", "AdamW"]},
            {"name": "lr0", "type": "float", "low": 0.001, "high": 0.011, "step": 0.005,
             "condition": {"param": "optimizer", "values": ["SGD"]}}])
        self.assertEqual(len(grid_points(spec)), 4)  # SGD x 3 learning rates + AdamW
        self.assertEqual(planned_trials(spec), 8)
        combined = spec.model_copy(update={"model_budget": "combined"})
        self.assertEqual(planned_trials(StudySpec.model_validate(combined.model_dump())), 4)

    # Impossible or unfair definitions are refused before any trial runs.
    def test_bad_definitions_are_refused(self):
        bad = [
            {**BASE, "search_space": [{"name": "lr0", "type": "float", "low": 0.001, "high": 0.1, "log": True}]},
            {**BASE, "sampler": "grid", "search_space": [{"name": "lr0", "type": "float", "low": 0.001, "high": 0.1}]},
            {**BASE, "search_space": [{"name": "device", "type": "categorical", "choices": ["cpu"]}]},
            {**BASE, "search_space": [{"name": "recipe.noise", "type": "int", "low": 0, "high": 9}]},
            {**BASE, "search_space": [{"name": "epochs", "type": "int", "low": 1, "high": 80}], "base_training": {}},
            {**BASE, "pruning": True, "sampler": "grid"},
            {**BASE, "dataset_id": None},
        ]
        for definition in bad:
            with self.subTest(definition=definition), self.assertRaises(ValidationError):
                StudySpec.model_validate(definition)


# Shared workspace, fixture worker and coordinator for study tests.
class StudyCase(unittest.TestCase):
    # Before each test: a workspace with a small training dataset.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v5-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        source = self.root / "squares"
        for split, count in (("train", 6), ("val", 2)):
            (source / split / "images").mkdir(parents=True)
            (source / split / "labels").mkdir(parents=True)
            for index in range(count):
                image = Image.new("RGB", (64, 64), (30, 30, 30))
                ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill=(200, 60, 50))
                image.save(source / split / "images" / f"{index}.jpg")
                (source / split / "labels" / f"{index}.txt").write_text(f"{index % 2} 0.5 0.5 0.5 0.5\n")
        from foxtrot.persistence.dataset_import import import_dataset
        import_dataset(self.workspace, self.checkout, source=source, dataset_id="squares", class_names=("A", "B"))
        from foxtrot.studies.coordinator import StudyCoordinator
        self.coordinator = StudyCoordinator(self.workspace, self.checkout)
        self.addCleanup(lambda: self.coordinator.stop())  # closes Optuna's database connections

    # A fixture worker; `epoch_seconds` > 0 lets the coordinator watch epochs (pruning).
    def worker(self, **settings):
        from foxtrot.execution.worker import Worker
        from foxtrot.pipeline.registry import PipelineSettings
        options = {"adapter": "fixture", "fixture_epoch_seconds": 0.0, "fixture_config_scores": True, **settings}
        return Worker(self.workspace, self.checkout, pipeline=PipelineSettings(**options),
                      runtime_directory=self.root / "runtime")

    # Create a study from a definition.
    def create(self, **definition) -> str:
        from foxtrot.studies.store import create_study
        return create_study(self.workspace, self.checkout, StudySpec.model_validate({**BASE, **definition}))

    # Alternate coordinator steps and worker passes until the study ends.
    def drive(self, study_id: str, worker=None, limit: int = 200) -> dict:
        from foxtrot.studies.store import read_study
        worker = worker or self.worker()
        for _ in range(limit):
            self.coordinator.step()
            if read_study(self.workspace, self.checkout, study_id)[2]["state"] in ("finished", "stopped"):
                return self.details(study_id)
            worker.run(once=True)
        self.fail("study did not end")

    # Trials and summary of a study.
    def details(self, study_id: str) -> dict:
        from foxtrot.studies.store import read_study, read_trials
        from foxtrot.studies.summary import summarize
        spec, record, state = read_study(self.workspace, self.checkout, study_id)
        trials = read_trials(self.workspace, self.checkout, study_id)
        return {"state": state, "trials": trials, "summary": summarize(spec, record, state, trials)}


# V5 core: grid and random studies, best trial, stop/resume, failures.
class StudyRunTests(StudyCase):
    # A grid runs exactly its combinations, each as its own job; the best matches the evaluated checkpoint.
    def test_grid_runs_each_combination_once(self):
        from foxtrot.config.result import JobResult
        study = self.create(sampler="grid", search_space=[
            {"name": "batch_size", "type": "categorical", "choices": [4, 8]},
            {"name": "optimizer", "type": "categorical", "choices": ["SGD", "AdamW"]}])
        result = self.drive(study)
        trials = result["trials"]
        self.assertEqual(sorted((t["params"]["batch_size"], t["params"]["optimizer"]) for t in trials),
                         [(4, "AdamW"), (4, "SGD"), (8, "AdamW"), (8, "SGD")])
        self.assertEqual({t["state"] for t in trials}, {"complete"})
        self.assertEqual(len({t["job_id"] for t in trials}), 4)
        for trial in trials:
            resolved = json.loads((self.workspace / "jobs" / trial["job_id"] / "resolved.json").read_text())
            self.assertEqual((resolved["training"]["batch_size"], resolved["training"]["optimizer"]),
                             (trial["params"]["batch_size"], trial["params"]["optimizer"]))
            self.assertEqual(resolved["study"], {"study_id": study, "trial_number": trial["number"]})
        best = result["summary"]["best"]
        self.assertEqual(best["value"], max(t["value"] for t in trials))
        evaluated = JobResult.model_validate_json((self.workspace / "jobs" / best["job_id"] / "result.json").read_text())
        self.assertEqual((evaluated.evaluation.map50_95, evaluated.checkpoint.sha256),
                         (best["value"], best["checkpoint_sha256"]))
        self.assertEqual(result["state"]["reason"], "All planned trials are done")

    # The same seed gives the same random configurations.
    def test_seeded_random_is_repeatable(self):
        space = [{"name": "lr0", "type": "float", "low": 0.001, "high": 0.1, "log": True},
                 {"name": "patience", "type": "int", "low": 2, "high": 20}]
        values = []
        for _ in range(2):
            study = self.create(sampler="random", trials_per_model=3, seed=7, search_space=space,
                                base_training={"epochs": 2, "optimizer": "SGD"})
            values.append([t["params"] for t in self.drive(study)["trials"]])
        self.assertEqual(values[0], values[1])
        self.assertEqual(len(values[0]), 3)

    # Stopping keeps finished trials; resuming continues the remaining budget without repeating any.
    def test_stop_and_resume_preserve_completed_trials(self):
        from foxtrot.studies.coordinator import StudyCoordinator, resume_study, stop_study
        study = self.create(sampler="random", trials_per_model=4, search_space=[
            {"name": "patience", "type": "int", "low": 2, "high": 40}])
        worker = self.worker()
        self.coordinator.step()           # trial 0 saved and launched
        self.coordinator.step()
        stop_study(self.workspace, self.checkout, study, "finish")
        worker.run(once=True)             # the active trial is allowed to finish
        result = self.drive(study, worker)
        self.assertEqual(result["state"]["state"], "stopped")
        done = [(t["number"], t["job_id"]) for t in result["trials"]]
        self.assertEqual(len(done), 1)
        resume_study(self.workspace, self.checkout, study)
        self.coordinator.stop()
        self.coordinator = StudyCoordinator(self.workspace, self.checkout)  # like a restart
        result = self.drive(study, worker)
        self.assertEqual(len(result["trials"]), 4)
        self.assertEqual([(t["number"], t["job_id"]) for t in result["trials"][:1]], done)
        self.assertEqual(len({t["job_id"] for t in result["trials"]}), 4)

    # Failed trials stay visible with their reason; the "stop" policy ends the study.
    def test_failures_are_visible_and_policy_stops(self):
        study = self.create(sampler="random", trials_per_model=3, failure_policy="stop", search_space=[
            {"name": "patience", "type": "int", "low": 2, "high": 40}])
        result = self.drive(study, self.worker(fixture_fail_at_epoch=1))
        self.assertEqual([t["state"] for t in result["trials"]], ["failed"])
        self.assertIn("fixture_training_failed", result["trials"][0]["error"])
        self.assertEqual(result["state"]["state"], "stopped")

    # Restarting the coordinator mid-trial never creates a second job for that trial.
    def test_restart_reuses_the_trial_job(self):
        from foxtrot.persistence.jobs import list_job_records
        from foxtrot.studies.coordinator import StudyCoordinator
        from foxtrot.studies.store import read_trials, write_trial
        study = self.create(sampler="random", trials_per_model=1, search_space=[
            {"name": "patience", "type": "int", "low": 2, "high": 40}])
        self.coordinator.step()
        self.coordinator.step()
        trial = read_trials(self.workspace, self.checkout, study)[0]
        job = trial["job_id"]
        trial.update(state="asked", job_id=None)  # as if the job was created but not yet recorded
        write_trial(self.workspace, self.checkout, study, trial)
        self.coordinator.stop()
        self.coordinator = StudyCoordinator(self.workspace, self.checkout)
        result = self.drive(study)
        self.assertEqual(result["trials"][0]["job_id"], job)
        self.assertEqual(len(list_job_records(self.workspace, self.checkout)), 1)


# V6: adaptive search, model comparison, pruning, MLflow, recipes, repeats.
class AdaptiveStudyTests(StudyCase):
    # Equal budgets give each model the advertised trials (round-robin); TPE uses sampled values.
    def test_equal_budget_model_comparison_with_tpe(self):
        study = self.create(sampler="tpe", trials_per_model=3, model_ids=["yolov8n", "yolo11n"], search_space=[
            {"name": "lr0", "type": "float", "low": 0.0001, "high": 0.1, "log": True}],
            base_training={"epochs": 2, "optimizer": "AdamW"})
        result = self.drive(study)
        models = [t["model_id"] for t in result["trials"]]
        self.assertEqual(models, ["yolov8n", "yolo11n"] * 3)
        for trial in result["trials"]:
            resolved = json.loads((self.workspace / "jobs" / trial["job_id"] / "resolved.json").read_text())
            self.assertEqual((resolved["model_id"], resolved["training"]["lr0"]), (trial["model_id"], trial["params"]["lr0"]))
        summary = result["summary"]
        self.assertEqual(summary["comparison"], "equal budget per model")
        self.assertEqual(set(summary["best_per_model"]), {"yolov8n", "yolo11n"})

    # Pruned trials are stopped early and stay distinguishable; MLflow links children to the study.
    def test_pruning_and_mlflow_parent_child_runs(self):
        from foxtrot.persistence.tracking import list_tracking
        from foxtrot.studies.tracking import finalize_studies, trial_run_tags
        from foxtrot.tracking.fixture import RecordingTrackingClient
        from foxtrot.tracking.uploader import sync_job_tracking
        study = self.create(sampler="random", trials_per_model=8, pruning=True, seed=3,
                            base_training={"epochs": 6, "optimizer": "SGD"},
                            search_space=[{"name": "lr0", "type": "float", "low": 0.001, "high": 0.1, "log": True}])
        worker = self.worker(fixture_epoch_seconds=0.4)  # slow enough to observe epochs even on a busy machine
        stop = threading.Event()

        # Keep the worker processing jobs in the background, like the real launcher.
        def keep_working():
            while not stop.is_set():
                if worker.run(once=True) == 0:
                    time.sleep(0.05)
        thread = threading.Thread(target=keep_working)
        thread.start()
        try:
            from foxtrot.studies.store import read_study
            for _ in range(600):
                self.coordinator.step()
                if read_study(self.workspace, self.checkout, study)[2]["state"] == "finished":
                    break
                time.sleep(0.05)
        finally:
            stop.set()
            thread.join(30)
        result = self.details(study)
        states = [t["state"] for t in result["trials"]]
        self.assertEqual(len(states), 8)
        self.assertIn("pruned", states)
        self.assertIn("complete", states)
        pruned = next(t for t in result["trials"] if t["state"] == "pruned")
        self.assertLess(len(pruned["intermediate"]), 6)
        self.assertEqual(result["summary"]["counts"]["pruned"], states.count("pruned"))

        client = RecordingTrackingClient()
        for trial in result["trials"]:
            sync_job_tracking(self.workspace, self.checkout, trial["job_id"], client,
                              extra_run_tags=lambda tags: trial_run_tags(self.workspace, self.checkout, client, "local", tags))
        self.assertEqual(finalize_studies(self.workspace, self.checkout, lambda *_: client),
                         [{"study_id": study, "finalized": True}])
        parent_id, parent = next((i, r) for i, r in client.runs.items() if r["tags"].get("foxtrot.kind") == "study")
        children = [r for r in client.runs.values() if r["tags"].get("mlflow.parentRunId") == parent_id]
        self.assertEqual(len(children), 8)
        self.assertEqual(sorted(c["tags"]["foxtrot.trial_state"] for c in children), sorted(states))
        killed = [c for c in children if c["tags"]["foxtrot.trial_state"] == "pruned"]
        self.assertTrue(all(c["status"] == "KILLED" and "pruner" in c["tags"]["foxtrot.trial_reason"] for c in killed))
        self.assertEqual(parent["status"], "FINISHED")
        self.assertEqual(parent["params"]["best.trial"], str(result["summary"]["best"]["number"]))

    # Recipe tuning: validation is frozen once, training data excludes validation backgrounds, recipes are cached.
    def test_recipe_study_freezes_validation(self):
        backgrounds = self.root / "backgrounds"
        backgrounds.mkdir()
        for index in range(5):
            Image.new("RGB", (160, 120), (40 + 40 * index, 110, 90)).save(backgrounds / f"bg{index}.png")
        study = self.create(dataset_id=None, sampler="random", trials_per_model=3, seed=5,
                            recipe={"backgrounds": str(backgrounds), "recipe": {"images_per_class": 3, "canvas_size": 96}},
                            search_space=[{"name": "recipe.noise", "type": "categorical", "choices": [10, 60]}])
        deadline = time.time() + 300
        from foxtrot.studies.store import read_study
        worker = self.worker()
        while read_study(self.workspace, self.checkout, study)[2]["state"] != "finished" and time.time() < deadline:
            self.coordinator.step()
            worker.run(once=True)
            time.sleep(0.2)
        result = self.details(study)
        trials = result["trials"]
        self.assertEqual([t["state"] for t in trials], ["complete"] * 3, [t.get("error") for t in trials])
        folder = self.workspace / "studies" / study / "validation"
        frozen = json.loads((folder / "manifest.json").read_text())
        excluded = {entry["file"] for entry in frozen["validation_backgrounds"]}
        from foxtrot.persistence.datasets import dataset_content_sha256
        for trial in trials:
            data = self.workspace / "datasets" / trial["dataset_id"] / "data"
            provenance = json.loads((data.parent / "generation.json").read_text())
            self.assertEqual(provenance["frozen_validation_sha256"], frozen["content_sha256"])
            self.assertEqual(dataset_content_sha256(data / "val"), dataset_content_sha256(folder / "data"))
            self.assertFalse(excluded & {entry["file"] for entry in provenance["backgrounds"]["files"]})
            self.assertEqual(provenance["recipe"]["noise"], trial["params"]["recipe.noise"])
        by_noise = {}
        for trial in trials:
            by_noise.setdefault(trial["params"]["recipe.noise"], set()).add(trial["dataset_id"])
        self.assertTrue(all(len(ids) == 1 for ids in by_noise.values()))  # same recipe -> same cached dataset
        self.assertTrue(any(t.get("dataset_cached") for t in trials) or len(by_noise) == 3)

    # Repeating the best configuration under several seeds keeps every individual result.
    def test_repeat_best_under_seeds(self):
        from foxtrot.api.app import create_app
        from test_api import call, guarded
        study = self.create(sampler="grid", search_space=[{"name": "batch_size", "type": "categorical", "choices": [4, 8]}])
        self.drive(study)
        app = create_app(self.workspace, self.checkout, port=8765, session_token="test-session-token", static_directory=None)
        status, content, _ = call(app, "POST", f"/api/studies/{study}/repeat", body={"top": 1, "seeds": [1, 2, 3]},
                                  headers=guarded())
        self.assertEqual(status, 200, content)
        repeat = json.loads(content)["study_id"]
        result = self.drive(repeat)
        self.assertEqual([t["params"]["seed"] for t in result["trials"]], [1, 2, 3])
        group = result["summary"]["repeat_groups"][0]
        self.assertEqual([v["seed"] for v in group["values"]], [1, 2, 3])
        self.assertIsNotNone(group["mean"])
        self.assertEqual(len({t["job_id"] for t in result["trials"]}), 3)
        status, content, _ = call(app, "GET", f"/api/studies/{repeat}")
        self.assertEqual((status, json.loads(content)["summary"]["sampler"]), (200, "repeat"))


if __name__ == "__main__":
    unittest.main()
