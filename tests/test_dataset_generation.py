"""V2: generate a Foxtrot dataset with explicit inputs, split it, cache it, publish it safely.

Generation runs through the real runner process (as the website does), so the
original generator's module-level settings never leak into this test process.
"""

import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from pydantic import ValidationError

from foxtrot.api.generation import GenerationRuns
from foxtrot.config.generation import GenerationRecipe, GenerationRequest, SplitSettings
from foxtrot.persistence.datasets import list_datasets, load_dataset
from foxtrot.pipeline.contracts import PipelineFailure
from foxtrot.pipeline.foxtrot_split import plan_split
from foxtrot.tracking.plan import generation_params

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "src" / "data_generation" / "res"


# Hash every file under a folder (to prove protected repository folders are untouched).
def tree_hash(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(folder).as_posix().encode() + path.read_bytes())
    return digest.hexdigest()


# Settings and split rules: validation and stable fingerprints, no generation needed.
class RecipeAndSplitTests(unittest.TestCase):
    # Defaults equal params.yaml; the hash ignores key order and changes with any setting.
    def test_recipe_defaults_and_fingerprint(self):
        recipe = GenerationRecipe()
        self.assertEqual((recipe.images_per_class, recipe.canvas_size, recipe.seed), (100, 320, 42))
        same = GenerationRecipe.model_validate(json.loads(json.dumps(recipe.model_dump(), sort_keys=True)))
        self.assertEqual(recipe.sha256(), same.sha256())
        self.assertNotEqual(recipe.sha256(), GenerationRecipe(seed=43).sha256())

    # Impossible settings are refused before anything runs.
    def test_bad_settings_are_rejected(self):
        for bad in ({"images_per_class": 0}, {"canvas_size": 16}, {"noise": 1.5}, {"unknown": 1},
                    {"shape_scale": {"low": 0.7, "high": 0.6}}, {"blur": {"low": -1, "high": 2}},
                    {"letter_probability": 2}):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                GenerationRecipe.model_validate(bad)
        with self.assertRaises(ValidationError):
            GenerationRequest(backgrounds="relative/folder")
        with self.assertRaises(ValidationError):
            SplitSettings(train_ratio=1.0)

    # Grouped split: each background's images stay together; the same seed gives the same split.
    def test_grouped_split_keeps_backgrounds_together(self):
        sources = {f"{i}.jpg": f"bg{i % 5}.png" for i in range(40)}
        outcome = plan_split(sources, SplitSettings())
        self.assertFalse(set(outcome.train_groups) & set(outcome.validation_groups))
        self.assertEqual(len(outcome.train) + len(outcome.validation), 40)
        self.assertTrue(outcome.train and outcome.validation)
        self.assertEqual(outcome, plan_split(sources, SplitSettings()))
        self.assertNotEqual(outcome, plan_split(sources, SplitSettings(seed=1)))

    # Ungrouped split follows the ratio; grouping needs at least two backgrounds.
    def test_ungrouped_ratio_and_single_background(self):
        sources = {f"{i}.jpg": "only.png" for i in range(10)}
        outcome = plan_split(sources, SplitSettings(group_by_background=False))
        self.assertEqual((len(outcome.train), len(outcome.validation)), (8, 2))
        with self.assertRaises(PipelineFailure) as raised:
            plan_split(sources, SplitSettings())
        self.assertEqual(raised.exception.code, "split_one_background")

    # MLflow gets short provenance parameters; ordinary datasets get none.
    def test_generation_params_for_mlflow(self):
        self.assertEqual(generation_params(None), {})
        params = generation_params({"identity_sha256": "a" * 64, "generation_seed": 7,
                                    "recipe": {"images_per_class": 4, "canvas_size": 128},
                                    "split": {"train_ratio": 0.8, "seed": 42, "group_by_background": True},
                                    "counts": {"train": 6, "val": 2}, "backgrounds": {"count": 3}})
        self.assertEqual(params["dataset.generated"], "true")
        self.assertEqual(params["dataset.generation.seed"], "7")
        self.assertEqual(params["dataset.split.group_by_background"], "true")
        self.assertEqual(params["dataset.split.validation_images"], "2")


# Real generation into a temporary workspace through the website's run manager.
class GenerationRunTests(unittest.TestCase):
    # Before each test: an empty workspace and three small background photos outside it.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v2-generation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.workspace = self.root / "workspace"
        self.backgrounds = self.root / "backgrounds"
        self.backgrounds.mkdir()
        for index in range(3):
            image = np.full((200, 260, 3), (40 + 60 * index, 120, 90), np.uint8)
            self.assertTrue(cv2.imwrite(str(self.backgrounds / f"bg{index}.png"), image))
        self.runs = GenerationRuns(self.workspace, self.checkout)
        self.addCleanup(self.runs.stop)

    # A small request: 4 images per class on a 128 px canvas.
    def request(self, **recipe) -> GenerationRequest:
        settings = {"images_per_class": 4, "canvas_size": 128, **recipe}
        return GenerationRequest(recipe=GenerationRecipe(**settings), backgrounds=str(self.backgrounds))

    # Start a run, wait for its process, and return its final status.
    def run_to_end(self, kind: str, request: GenerationRequest, *, cancel_first: bool = False) -> dict:
        status = self.runs.start(kind, request)
        if cancel_first:
            self.runs.cancel(status["run_id"])
        self.runs._active[1].wait(timeout=180)
        return self.runs.status(status["run_id"])

    # Leftover hidden build folders would mean a failed run left half a dataset behind.
    def hidden_builds(self) -> list[str]:
        return [p.name for p in (self.workspace / "datasets").iterdir() if p.name.startswith(".")]

    # Full flow: generate → split → check → publish; then reuse; a changed seed makes a new dataset.
    def test_generate_publish_reuse_and_new_identity(self):
        protected = (tree_hash(ROOT / "src" / "data_generation"), tree_hash(ROOT / "data" / "generated"))
        first = self.run_to_end("dataset", self.request())
        self.assertEqual(first["state"], "finished", first)
        result = first["result"]
        self.assertFalse(result["cached"])
        self.assertEqual(result["counts"]["images"], 8)
        self.assertEqual(result["counts"]["train"] + result["counts"]["val"], 8)
        self.assertEqual(result["counts"]["objects_per_class"], {"Hexagon": 4, "Triangle": 4})
        dataset = load_dataset(self.workspace, self.checkout, result["dataset_id"])  # verifies the content hash
        self.assertEqual(dataset.metadata.validation_status, "valid")
        self.assertEqual(dataset.metadata.class_names, ("Hexagon", "Triangle"))
        data = Path(dataset.root)
        for split in ("train", "val"):
            for image in (data / split / "images").iterdir():
                self.assertEqual(cv2.imread(str(image)).shape[:2], (128, 128))  # canvas size, not training size
                self.assertTrue((data / split / "labels" / f"{image.stem}.txt").is_file())
        provenance = json.loads((data.parent / "generation.json").read_text())
        self.assertEqual(provenance["content_sha256"], dataset.metadata.content_sha256)
        self.assertEqual(len(provenance["backgrounds"]["files"]), 3)
        self.assertFalse(set(provenance["groups"]["train"]) & set(provenance["groups"]["val"]))
        self.assertTrue((self.workspace / "generation" / "letters" / "letters-128").is_dir())

        again = self.run_to_end("dataset", self.request())
        self.assertEqual((again["state"], again["result"]["cached"], again["result"]["dataset_id"]),
                         ("finished", True, result["dataset_id"]))
        changed = self.run_to_end("dataset", self.request(seed=7))
        self.assertEqual(changed["state"], "finished", changed)
        self.assertNotEqual(changed["result"]["dataset_id"], result["dataset_id"])
        self.assertEqual(len(list_datasets(self.workspace, self.checkout)), 2)
        self.assertEqual(self.hidden_builds(), [])
        # The repository's generator, letter cache and generated data were only read.
        self.assertEqual(protected, (tree_hash(ROOT / "src" / "data_generation"), tree_hash(ROOT / "data" / "generated")))
        self.assertFalse((RESOURCES / "letters128").exists())

    # Preview uses the same generator with its own output folder and reports every box.
    def test_preview_images_and_boxes(self):
        status = self.run_to_end("preview", self.request(images_per_class=10))
        self.assertEqual(status["state"], "finished", status)
        preview = status["result"]
        self.assertEqual(len(preview["images"]), 6)  # 3 per class at most
        self.assertEqual(preview["seed"], 42)
        for item in preview["images"]:
            self.assertTrue(self.runs.preview_image(status["run_id"], item["image"]).is_file())
            self.assertEqual(len(item["boxes"]), 1)
            box = item["boxes"][0]
            self.assertIn(box["class_name"], ("Hexagon", "Triangle"))
            self.assertTrue(0 < box["width"] <= 1 and 0 < box["height"] <= 1)
        self.assertEqual(list_datasets(self.workspace, self.checkout), ())  # previews never publish
        with self.assertRaises(KeyError):
            self.runs.preview_image(status["run_id"], "../request.json")

    # Bad input fails before generation and leaves nothing behind.
    def test_invalid_backgrounds_fail_without_output(self):
        empty = self.root / "empty"
        empty.mkdir()
        status = self.run_to_end("dataset", GenerationRequest(recipe=GenerationRecipe(images_per_class=2),
                                                              backgrounds=str(empty)))
        self.assertEqual((status["state"], status["error"]["code"]), ("failed", "backgrounds_missing"))
        self.assertEqual(list_datasets(self.workspace, self.checkout), ())
        self.assertEqual(self.hidden_builds(), [])

    # Cancelling stops the run and removes its temporary build folder.
    def test_cancel_leaves_no_dataset(self):
        status = self.run_to_end("dataset", self.request(), cancel_first=True)
        self.assertEqual(status["state"], "cancelled", status)
        self.assertEqual(list_datasets(self.workspace, self.checkout), ())
        self.assertEqual(self.hidden_builds(), [])

    # A published dataset whose files changed is never overwritten by a new run.
    def test_changed_cached_dataset_is_not_overwritten(self):
        first = self.run_to_end("dataset", self.request())
        dataset_id = first["result"]["dataset_id"]
        label = next((self.workspace / "datasets" / dataset_id / "data" / "train" / "labels").iterdir())
        label.write_text("0 0.5 0.5 0.1 0.1\n")
        second = self.run_to_end("dataset", self.request())
        self.assertEqual((second["state"], second["error"]["code"]), ("failed", "cached_dataset_changed"))
        self.assertEqual(label.read_text(), "0 0.5 0.5 0.1 0.1\n")

    # Two runs can never build the same dataset at once (the per-dataset lock).
    def test_conflicting_build_is_refused(self):
        from foxtrot.persistence.dataset_generation import generation_identity
        from foxtrot.pipeline.foxtrot_generator import list_backgrounds

        request = self.request()
        dataset_id = generation_identity(request, list_backgrounds(self.backgrounds))["dataset_id"]
        self.runs._runs()  # initializes the workspace and its generation folder
        lock = self.workspace / "generation" / "locks" / f"{dataset_id}.lock"
        lock.parent.mkdir()
        with lock.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            status = self.run_to_end("dataset", request)
        self.assertEqual((status["state"], status["error"]["code"]), ("failed", "dataset_busy"))
        self.assertEqual(list_datasets(self.workspace, self.checkout), ())

    # Only one run at a time; a second start is refused while the first is running.
    def test_one_run_at_a_time(self):
        from foxtrot.api.generation import GenerationBusy

        self.runs.start("dataset", self.request(images_per_class=30))
        with self.assertRaises(GenerationBusy):
            self.runs.start("preview", self.request())

    # Training on a generated dataset queues its generation/split record for MLflow.
    def test_training_job_logs_generation_provenance(self):
        from foxtrot.execution.worker import Worker
        from foxtrot.persistence.jobs import create_recorded_job, get_job_record
        from foxtrot.persistence.tracking import list_tracking
        from foxtrot.pipeline.registry import PipelineSettings

        built = self.run_to_end("dataset", self.request())
        dataset_id = built["result"]["dataset_id"]
        job = create_recorded_job(self.workspace, self.checkout, {
            "dataset_id": dataset_id, "tracking_destination_id": "team", "training": {"epochs": 1},
        }, "generated-one").name
        worker = Worker(self.workspace, self.checkout, runtime_directory=self.root / "runtime",
                        pipeline=PipelineSettings(adapter="fixture", fixture_epoch_seconds=0.0))
        self.assertEqual(worker.run(once=True), 1)
        self.assertEqual(get_job_record(self.workspace, self.checkout, job).execution_state, "finished")
        operations = [entry.operation for entry in list_tracking(self.workspace, self.checkout, job)]
        params = next(o for o in operations if o.operation == "log_params").payload["params"]
        self.assertEqual(params["dataset.generated"], "true")
        self.assertEqual(params["dataset.generation.images_per_class"], "4")
        self.assertEqual(params["dataset.split.group_by_background"], "true")
        artifacts = [o.payload["name"] for o in operations if o.operation == "log_artifact"]
        self.assertIn("dataset-provenance", artifacts)
        saved = json.loads((self.workspace / "jobs" / job / "dataset-provenance.json").read_text())
        self.assertEqual(saved["dataset_id"], dataset_id)
        # The upload step sends the provenance file into the run's "dataset" folder.
        from foxtrot.tracking.fixture import RecordingTrackingClient
        from foxtrot.tracking.uploader import sync_job_tracking

        client = RecordingTrackingClient()
        self.assertEqual(sync_job_tracking(self.workspace, self.checkout, job, client).tracking_state, "synced")
        (run,) = client.runs.values()
        self.assertIn("dataset-provenance.json", {name for path, name, _ in run["artifacts"] if path == "dataset"})


# The website endpoints for generation (requests go through the real app, not a browser).
class GenerationApiTests(unittest.TestCase):
    # Before each test: an app over an empty workspace.
    def setUp(self):
        from foxtrot.api.app import create_app

        temporary = tempfile.TemporaryDirectory(prefix="foxtrot-v2-api-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        (root / "checkout").mkdir()
        self.app = create_app(root / "workspace", root / "checkout", port=8765,
                              session_token="test-session-token", static_directory=None)
        self.addCleanup(self.app.state.generation.stop)

    # Options give the params.yaml defaults and say the generator is available in this checkout.
    def test_options(self):
        from test_api import call

        status, content, _ = call(self.app, "GET", "/api/generation/options")
        options = json.loads(content)
        self.assertEqual(status, 200)
        self.assertTrue(options["available"])
        self.assertEqual(options["recipe"]["images_per_class"], 100)
        self.assertEqual(options["split"]["train_ratio"], 0.8)
        self.assertEqual(options["default_backgrounds"], str(ROOT / "data" / "raw"))

    # Starting needs the session; bad settings get 422; unknown runs get 404.
    def test_start_requires_session_and_valid_settings(self):
        from test_api import call, guarded

        body = {"backgrounds": "/tmp", "recipe": {"images_per_class": 2}}
        status, _, _ = call(self.app, "POST", "/api/generation/preview", body=body)
        self.assertEqual(status, 403)
        status, _, _ = call(self.app, "POST", "/api/generation/preview", headers=guarded(),
                            body={"backgrounds": "/tmp", "recipe": {"images_per_class": 0}})
        self.assertEqual(status, 422)
        status, content, _ = call(self.app, "POST", "/api/generation/preview", headers=guarded(), body=body)
        self.assertEqual((status, json.loads(content)["kind"]), (200, "preview"))
        status, _, _ = call(self.app, "GET", "/api/generation/runs/" + "0" * 32)
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
