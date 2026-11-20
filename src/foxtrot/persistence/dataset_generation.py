"""Build a generated dataset safely and reuse it when nothing changed (V2).

Flow: check backgrounds → compute the dataset's identity → if that dataset is
already published and unchanged, reuse it → otherwise generate into a hidden
temporary folder, split, check every label and count, write the manifest and
provenance, and publish with one rename. A failed, cancelled or interrupted
build never replaces or half-creates a dataset.

Identity = recipe + split settings + every background's hash + the generator's
code/resources/library versions. Same identity → same dataset ID ("gen-…").
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from uuid import uuid4

from foxtrot.config.dataset import DatasetMetadata
from foxtrot.config.generation import GenerationRecipe, GenerationRequest
from foxtrot.pipeline.contracts import DatasetInput, PipelineFailure, ProgressReporter
from foxtrot.pipeline.dataset_validation import DatasetReport, validate_dataset
from foxtrot.pipeline.foxtrot_generator import generate_images, generator_files, list_backgrounds
from foxtrot.pipeline import foxtrot_generator, foxtrot_split

from .datasets import DATA, MANIFEST, DatasetUnavailable, _dataset_directory, dataset_content_sha256, load_dataset
from .files import fingerprint, sync_directory, write_new_text
from .workspace import initialize_workspace

PROVENANCE = "generation.json"
REPORT = "validation.json"
SPLITS = {"train": "train/images", "val": "val/images"}
RUNTIME_PACKAGES = ("opencv-python", "numpy", "pillow")  # their versions can change the pixels


# The result of a build: the dataset, whether it was reused, and its recorded provenance.
@dataclass(frozen=True)
class GeneratedDataset:
    dataset: DatasetInput
    cached: bool
    provenance: dict
    report: DatasetReport | None


# Workspace folder for generation helpers (letter cache, locks, run folders).
def generation_root(root: Path) -> Path:
    folder = root / "generation"
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise DatasetUnavailable("Workspace generation folder is unsafe")
    folder.mkdir(exist_ok=True)
    return folder


# A stable hash of any JSON-compatible value.
def _hash_json(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Hash of everything in the generator that decides the pictures, plus library versions.
def generator_identity() -> dict:
    files = [*generator_files(), Path(foxtrot_generator.__file__), Path(foxtrot_split.__file__)]
    entries = [{"file": path.name, "sha256": fingerprint(path)[0]} for path in files]
    runtime = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    return {"sha256": _hash_json({"files": entries, "runtime": runtime}), "files": entries, "runtime": runtime}


# Work out the dataset's identity and ID without generating anything.
def generation_identity(request: GenerationRequest, backgrounds: list[Path]) -> dict:
    background_entries = [{"file": path.name, "sha256": fingerprint(path)[0]} for path in backgrounds]
    generator = generator_identity()
    identity = {
        "format": 1,
        "recipe": request.recipe.model_dump(mode="json"),
        "split": request.split.model_dump(mode="json"),
        "backgrounds": background_entries,
        "generator_sha256": generator["sha256"],
    }
    digest = _hash_json(identity)
    return {"sha256": digest, "dataset_id": f"gen-{digest[:16]}", "backgrounds": background_entries,
            "generator": generator}


# Hold an exclusive lock for one dataset ID, so two runs never build the same dataset at once.
class _DatasetLock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise PipelineFailure("dataset_busy", "Another run is already building this exact dataset") from None
        return self

    def __exit__(self, *_):
        self.stream.close()  # closing releases the lock


# Read a generated dataset's recorded provenance, if it has one that matches the manifest.
def load_generation_provenance(dataset: DatasetInput) -> dict | None:
    path = Path(dataset.root).parent / PROVENANCE
    if path.is_symlink() or not path.is_file():
        return None
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metadata = dataset.metadata
    if not isinstance(provenance, dict) or (provenance.get("dataset_id"), provenance.get("content_sha256")) != (
            metadata.dataset_id, metadata.content_sha256):
        return None
    return provenance


# Make sure the generated files match what the recipe promised before publishing.
def _check_generated(report: DatasetReport, recipe: GenerationRecipe, class_names: tuple[str, ...]) -> None:
    if not report.valid:
        first = report.errors[0].message if report.errors else "unknown problem"
        raise PipelineFailure("generated_invalid", f"Generated dataset failed checks ({report.error_count}): {first}")
    for name in class_names:
        objects = sum(split.objects_per_class.get(name, 0) for split in report.splits.values())
        if objects != recipe.images_per_class:
            raise PipelineFailure("generated_counts",
                                  f"Expected {recipe.images_per_class} {name} objects, found {objects}")


# Generate (or reuse) the dataset described by `request` and return it.
def build_generated_dataset(
    workspace_root: str | Path, source_checkout: str | Path, request: GenerationRequest,
    reporter: ProgressReporter,
) -> GeneratedDataset:
    root = initialize_workspace(workspace_root, source_checkout)
    helpers = generation_root(root)
    reporter.progress({"phase": "checking backgrounds"})
    backgrounds = list_backgrounds(request.backgrounds)
    identity = generation_identity(request, backgrounds)
    dataset_id = identity["dataset_id"]
    directory = _dataset_directory(root, dataset_id)
    with _DatasetLock(helpers / "locks" / f"{dataset_id}.lock"):
        if directory.exists() or directory.is_symlink():
            try:
                dataset = load_dataset(root, source_checkout, dataset_id)
            except DatasetUnavailable as error:
                raise PipelineFailure("cached_dataset_changed", f"Dataset {dataset_id} exists but cannot be reused "
                                      f"({error}). It is left untouched.") from error
            return GeneratedDataset(dataset=dataset, cached=True, report=None,
                                    provenance=load_generation_provenance(dataset) or {})
        staging = root / "datasets" / f".generate-{uuid4().hex}"
        try:
            raw = staging / "raw"
            generated = generate_images(request.recipe, backgrounds, raw, helpers / "letters", reporter)
            reporter.progress({"phase": "splitting"})
            outcome = foxtrot_split.plan_split(generated.sources, request.split)
            foxtrot_split.write_split(raw, staging / DATA, outcome)
            shutil.rmtree(raw)
            reporter.progress({"phase": "validating"})
            report = validate_dataset(staging / DATA, generated.class_names, SPLITS, missing_labels="error")
            _check_generated(report, request.recipe, generated.class_names)
            metadata = DatasetMetadata(
                dataset_id=dataset_id, class_names=generated.class_names,
                train_images_ref=SPLITS["train"], validation_images_ref=SPLITS["val"],
                content_sha256=dataset_content_sha256(staging / DATA), validation_status="valid",
            )
            provenance = {
                "format": 1, "dataset_id": dataset_id, "identity_sha256": identity["sha256"],
                "content_sha256": metadata.content_sha256,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "class_names": list(generated.class_names),
                "recipe": request.recipe.model_dump(mode="json"), "recipe_sha256": request.recipe.sha256(),
                "generation_seed": request.recipe.seed,
                "split": request.split.model_dump(mode="json"),
                "counts": {"images": len(generated.sources), "train": len(outcome.train),
                           "val": len(outcome.validation),
                           "objects_per_class": {name: request.recipe.images_per_class
                                                 for name in generated.class_names}},
                "backgrounds": {"folder": request.backgrounds, "count": len(backgrounds),
                                "files": identity["backgrounds"]},
                "groups": {"train": list(outcome.train_groups), "val": list(outcome.validation_groups)},
                "image_backgrounds": {
                    **{f"train/images/{n}": generated.sources[n] for n in outcome.train},
                    **{f"val/images/{n}": generated.sources[n] for n in outcome.validation},
                },
                "generator": identity["generator"],
            }
            write_new_text(staging / MANIFEST, metadata.model_dump_json(indent=2))
            write_new_text(staging / PROVENANCE, json.dumps(provenance, indent=2))
            write_new_text(staging / REPORT, report.model_dump_json(indent=2))
            sync_directory(staging)
            staging.rename(directory)  # publish: complete or not at all
            sync_directory(directory.parent)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return GeneratedDataset(dataset=DatasetInput(metadata=metadata, root=str(directory / DATA)),
                            cached=False, provenance=provenance, report=report)


# Read one YOLO label file into boxes for the preview (class name + centre/size, 0–1).
def read_boxes(label: Path, class_names: tuple[str, ...]) -> list[dict]:
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5:
            index = int(fields[0])
            x, y, w, h = (float(value) for value in fields[1:])
            name = class_names[index] if 0 <= index < len(class_names) else str(index)
            boxes.append({"class_name": name, "x": x, "y": y, "width": w, "height": h})
    return boxes


# Make a few sample images with the same code path as a full run, into `output`.
def generate_preview(
    workspace_root: str | Path, source_checkout: str | Path, request: GenerationRequest,
    output: Path, reporter: ProgressReporter, *, per_class: int = 3,
) -> dict:
    root = initialize_workspace(workspace_root, source_checkout)
    backgrounds = list_backgrounds(request.backgrounds)
    recipe = request.recipe.model_copy(update={"images_per_class": min(per_class, request.recipe.images_per_class)})
    generated = generate_images(recipe, backgrounds, output, generation_root(root) / "letters", reporter)
    images = []
    for name in sorted(generated.sources, key=lambda item: int(Path(item).stem)):
        images.append({"image": name, "background": generated.sources[name],
                       "boxes": read_boxes(output / "labels" / f"{Path(name).stem}.txt", generated.class_names)})
    return {"seed": recipe.seed, "canvas_size": recipe.canvas_size, "class_names": list(generated.class_names),
            "background_count": len(backgrounds), "images": images}
