"""Training data for dataset-recipe studies (V6).

Fair recipe tuning needs one fixed validation set; otherwise a recipe could win
by making its own validation images easier. So, for each study:

1. The backgrounds are divided once (by the study seed) into validation and
   training backgrounds. The validation images are generated once with the
   base recipe and *frozen* in ``studies/<id>/validation/``.
2. Each trial generates only training images, from training backgrounds only
   (validation backgrounds are excluded), and is published as a normal dataset
   whose val/ split is a copy of the frozen validation set.
3. A trial dataset is identified by its recipe, training backgrounds, generator
   and frozen validation; an identical recipe reuses the published dataset.

Runs as a separate process (the generator keeps settings in globals):

    python -m foxtrot.studies.recipes --workspace W --source-checkout C --study ID --trial N
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import random
import shutil
import sys
from uuid import uuid4

from foxtrot.config.dataset import DatasetMetadata
from foxtrot.config.generation import GenerationRecipe
from foxtrot.persistence.dataset_generation import (
    PROVENANCE, _DatasetLock, _hash_json, generation_root, generator_identity,
)
from foxtrot.persistence.datasets import DATA, MANIFEST, _dataset_directory, dataset_content_sha256, load_dataset
from foxtrot.persistence.files import fingerprint, sync_directory, write_new_text
from foxtrot.pipeline.contracts import PipelineCancelled, PipelineFailure
from foxtrot.pipeline.dataset_validation import validate_dataset
from foxtrot.pipeline.foxtrot_generator import generate_images, list_backgrounds

from .store import read_json, read_study, study_folder, write_json

SPLITS = {"train": "train/images", "val": "val/images"}


# Minimal progress reporter for this process; cancellation is the coordinator's job (it stops us).
class _Quiet:
    def progress(self, payload): pass
    def log(self, payload): pass
    def wait_cancelled(self, seconds=0.0): return False


# Divide the backgrounds once and generate the frozen validation images (only if not done yet).
def freeze_validation(workspace: Path, checkout: Path, study_id: str) -> dict:
    spec, _, _ = read_study(workspace, checkout, study_id)
    folder = study_folder(workspace, checkout, study_id)
    target = folder / "validation"
    with _DatasetLock(generation_root(workspace.resolve()) / "locks" / f"study-{study_id}-validation.lock"):
        if (target / "manifest.json").is_file():
            return read_json(target / "manifest.json")
        backgrounds = list_backgrounds(spec.recipe.backgrounds)
        if len(backgrounds) < 2:
            raise PipelineFailure("too_few_backgrounds", "A recipe study needs at least 2 background images")
        order = sorted(backgrounds, key=lambda p: p.name)
        random.Random(spec.seed).shuffle(order)
        count = min(len(order) - 1, max(1, round(len(order) * spec.validation_share)))
        validation_backgrounds = sorted(order[:count], key=lambda p: p.name)
        base = spec.recipe.recipe
        share = spec.validation_share
        per_class = max(4, round(base.images_per_class * share / (1 - share)))
        recipe = base.model_copy(update={"images_per_class": per_class, "seed": base.seed + 1})
        staging = folder / f".validation-{uuid4().hex}"
        try:
            generate_images(recipe, validation_backgrounds, staging / "data",
                            generation_root(workspace.resolve()) / "letters", _Quiet())
            shutil.rmtree(staging / "data" / "unused", ignore_errors=True)
            manifest = {  # the hash covers data/ only; the manifest sits beside it
                "content_sha256": dataset_content_sha256(staging / "data"), "recipe": recipe.model_dump(mode="json"),
                "validation_backgrounds": [{"file": p.name, "sha256": fingerprint(p)[0]} for p in validation_backgrounds],
                "training_backgrounds": sorted(p.name for p in order[count:]),
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            write_new_text(staging / "manifest.json", json.dumps(manifest, indent=2))
            sync_directory(staging)
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return manifest


# Build (or reuse) one trial's training dataset: new training images + the frozen validation copy.
def build_trial_dataset(workspace: Path, checkout: Path, study_id: str, recipe: GenerationRecipe) -> tuple[str, bool]:
    spec, _, _ = read_study(workspace, checkout, study_id)
    frozen = freeze_validation(workspace, checkout, study_id)
    validation = study_folder(workspace, checkout, study_id) / "validation" / "data"
    if dataset_content_sha256(validation) != frozen["content_sha256"]:
        raise PipelineFailure("validation_changed", "The study's frozen validation images changed")
    excluded = {entry["file"] for entry in frozen["validation_backgrounds"]}
    backgrounds = [p for p in list_backgrounds(spec.recipe.backgrounds) if p.name not in excluded]
    background_hashes = [{"file": p.name, "sha256": fingerprint(p)[0]} for p in backgrounds]
    generator = generator_identity()
    identity = _hash_json({"format": "study-recipe-1", "recipe": recipe.model_dump(mode="json"),
                           "training_backgrounds": background_hashes, "generator_sha256": generator["sha256"],
                           "frozen_validation_sha256": frozen["content_sha256"]})
    dataset_id = f"gen-t-{identity[:16]}"
    root = workspace.resolve()
    directory = _dataset_directory(root, dataset_id)
    with _DatasetLock(generation_root(root) / "locks" / f"{dataset_id}.lock"):
        if directory.exists():
            load_dataset(root, checkout, dataset_id)  # must still match its manifest; never overwritten
            return dataset_id, True
        staging = root / "datasets" / f".generate-{uuid4().hex}"
        try:
            raw = staging / "raw"
            generated = generate_images(recipe, backgrounds, raw, generation_root(root) / "letters", _Quiet())
            data = staging / DATA
            (data / "train").mkdir(parents=True)
            raw.joinpath("images").rename(data / "train" / "images")
            raw.joinpath("labels").rename(data / "train" / "labels")
            shutil.copytree(validation / "images", data / "val" / "images")
            shutil.copytree(validation / "labels", data / "val" / "labels")
            shutil.rmtree(raw)
            report = validate_dataset(data, generated.class_names, SPLITS, missing_labels="error")
            train_objects = report.splits["train"].objects_per_class
            if not report.valid or any(train_objects.get(n) != recipe.images_per_class for n in generated.class_names):
                raise PipelineFailure("generated_invalid", "Generated trial data failed its checks")
            metadata = DatasetMetadata(dataset_id=dataset_id, class_names=generated.class_names,
                                       train_images_ref=SPLITS["train"], validation_images_ref=SPLITS["val"],
                                       content_sha256=dataset_content_sha256(data), validation_status="valid")
            provenance = {
                "format": 1, "source": "study-recipe", "dataset_id": dataset_id, "identity_sha256": identity,
                "content_sha256": metadata.content_sha256, "study_id": study_id,
                "recipe": recipe.model_dump(mode="json"), "recipe_sha256": recipe.sha256(),
                "generation_seed": recipe.seed, "frozen_validation_sha256": frozen["content_sha256"],
                "validation_backgrounds_excluded": sorted(excluded),
                "backgrounds": {"folder": spec.recipe.backgrounds, "count": len(backgrounds), "files": background_hashes},
                "counts": {"images": len(generated.sources), "train": len(generated.sources),
                           "val": report.splits["val"].images},
                "generator": generator,
            }
            write_new_text(staging / MANIFEST, metadata.model_dump_json(indent=2))
            write_new_text(staging / PROVENANCE, json.dumps(provenance, indent=2))
            sync_directory(staging)
            staging.rename(directory)
            sync_directory(directory.parent)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return dataset_id, False


# Command line used by the coordinator: build the dataset for one trial, write the outcome file.
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--study", required=True)
    parser.add_argument("--trial", required=True, type=int)
    arguments = parser.parse_args(argv)
    folder = study_folder(arguments.workspace, arguments.source_checkout, arguments.study) / "trials"
    trial = read_json(folder / f"{arguments.trial}.json")
    outcome_path = folder / f"{arguments.trial}.prepare.json"
    try:
        dataset_id, cached = build_trial_dataset(arguments.workspace, arguments.source_checkout, arguments.study,
                                                 GenerationRecipe.model_validate(trial["recipe"]))
        write_json(outcome_path, {"ok": True, "dataset_id": dataset_id, "cached": cached})
        return 0
    except (PipelineFailure, PipelineCancelled) as failure:
        write_json(outcome_path, {"ok": False, "code": getattr(failure, "code", "cancelled"),
                                  "message": getattr(failure, "message", "Cancelled")})
    except Exception as error:
        print(f"recipe preparation failed: {error!r}", file=sys.stderr)
        write_json(outcome_path, {"ok": False, "code": "internal", "message": str(error)[:300]})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
