# Generating a dataset in the browser (V2)

The **Generate** screen makes a new Foxtrot shapes dataset (hexagons and
triangles, optionally with letters, drawn onto real background photos), splits it
into train/validation, checks it, and registers it so **New training** can use it.

## Using it

1. Start Foxtrot from the source checkout:
   `foxtrot --workspace /abs/ws --pipeline-adapter ultralytics`.
2. Open **Generate**. The defaults are the same as `params.yaml`; the backgrounds
   folder defaults to the checkout's `data/raw` (only read, never changed).
3. **Preview samples** makes up to 3 images per class with the same code and
   settings, and draws each label box over its image.
4. **Generate dataset** shows progress and can be cancelled. When it finishes,
   **Train on this dataset** opens New training with it selected.

*Images per class* is the old `n` (2 classes × 100 = 200 images). *Canvas size*
is the generated picture size (old `size`); YOLO's training image size is set
separately in New training.

## What happens

| Step | Where | Guarantee |
| --- | --- | --- |
| Settings | `config/generation.py` | Strict recipe, split and request models; bad values are refused before anything runs |
| Generator | `pipeline/foxtrot_generator.py` | Runs the unchanged `src/data_generation` code with explicit settings, backgrounds and output folder. Output is byte-identical to the original script for the same settings and seed |
| Letter cache | `<ws>/generation/letters/letters-<size>/` | Built in a temporary folder and renamed; never written into `src/data_generation/res/` |
| Split | `pipeline/foxtrot_split.py` | Reproducible from the split seed. With *grouping* on, all images from one background photo land in the same split |
| Build and publish | `persistence/dataset_generation.py` | Generated into hidden `datasets/.generate-<id>/`, split, label-checked, counts compared with the recipe, then published with one rename. Failure, cancel or a crash never leaves a usable-looking partial dataset |
| Run process | `pipeline/generation_runner.py`, `api/generation.py` | A separate process per run, one at a time; `generation/runs/<run>/` keeps request, status, log and preview images |
| MLflow | `tracking/plan.py`, `execution/outcomes.py` | Jobs on a generated dataset log `dataset.generation.*`/`dataset.split.*` parameters and `dataset/dataset-provenance.json` |

## Identity and reuse

The dataset ID is `gen-` plus the first 16 hex digits of a hash over the recipe,
the split settings, every background file's SHA-256, and the generator's code,
resources and OpenCV/NumPy/Pillow versions. The same request reuses the
published dataset (after re-checking its content hash). Any change produces a
new dataset ID. A per-ID lock refuses two builds of the same dataset at once. A
published dataset whose files were changed is reported and left untouched;
it is never overwritten.

Each dataset folder holds `manifest.json`, `generation.json` (recipe, seeds,
backgrounds with hashes, train/val background groups, source background for
every image, generator identity, counts) and `validation.json`.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/generation/options` | Defaults, default backgrounds folder, whether the generator is available |
| `POST /api/generation/preview`, `POST /api/generation/datasets` | Start a run (session required; 409 while another run is active) |
| `GET /api/generation/runs/{id}` | State, phase, progress, error, result |
| `POST /api/generation/runs/{id}/cancel` | Ask the run to stop |
| `GET /api/generation/runs/{id}/images/{n}.jpg` | One preview image |

## Limits

- The generator is found at `src/data_generation` next to the package, so
  generation needs the source checkout (planned Docker image includes it). An
  installed wheel without it shows "generator not available".
- Classes are the shared `src/class_names.py` shapes; the letter-crop dataset
  mode of the original script is not offered.
- A run killed with SIGKILL may leave a hidden `.generate-*` or `.letters-*`
  folder. These are ignored and can be deleted.
