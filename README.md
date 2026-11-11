# Team Foxtrot: Local YOLO Training

Synthetic UAV detection data generation and local YOLO training, with DVC for
pipeline/data provenance and MLflow for experiment tracking.

The local web application is planned in [implementation_plan.md](implementation_plan.md).
It has not been implemented yet.

The H1 development baseline (package skeleton, pinned dependencies, and safe
verification commands) is documented in [docs/development.md](docs/development.md).
Use that setup before running the commands below. Current implementation and
approval state are recorded in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
The H2 configuration contracts and H3 local persistence now support validated job
files, database records, queue reservations, and exact artifact references.
See [docs/persistence.md](docs/persistence.md) for storage behavior and checks.
H4 adds independent execution/upload states, committed progress history, and
restart reconciliation. See [docs/lifecycle.md](docs/lifecycle.md) for the
contracts, Linux process evidence, and verification fixtures. H5 adds the local
worker, fixed subprocess runner, durable cancellation, output capture, and
bounded child cleanup; see [docs/execution.md](docs/execution.md). It currently
runs explicit infrastructure fixtures. Real adapters and the browser application
remain later work.

## Project files

| Path | Purpose |
| --- | --- |
| `implementation_plan.md` | Current architecture and delivery steps: horizontal foundation, then vertical slices |
| `research.md` | Architecture research; issues 3–7 were subsequently fixed |
| `repository_verification.md` | Findings, implemented fixes, and recorded verification |
| `project_roadmap.md` | Historical roadmap and original scope |
| `params.yaml` | Current generation, split, and training configuration |
| `src/data_generation/` | Active generator, helpers, and required resources |
| `src/class_names.py` | Shared class IDs |
| `src/split_dataset.py` | Current deterministic train/validation split |
| `src/train.py` | Local YOLO training and existing MLflow integration |
| `src/test.py` | Optional visual inference on real-flight images |
| `tests/` | Generator/training-import regressions, configuration contracts, and local persistence checks |
| `dvc.yaml`, `dvc.lock`, `.dvc/` | Existing DVC pipeline, recorded state, and configuration |

## Run the current pipeline

Use the verified Ubuntu/Python baseline described in the development guide.
Activate its environment, then install the locked dependencies if needed:

```bash
python -m pip install --require-hashes --no-build-isolation -r requirements.txt
```

Place backgrounds in `data/raw/` and edit `params.yaml`. The active script entry
point reads YAML; the retained legacy argument parser is not the current
configuration interface.

Run the stages from the repository root:

```bash
python src/data_generation/DataSetGeneration.py
python src/split_dataset.py
python src/train.py
python src/test.py
```

Alternatively, use `dvc repro` with the configured DVC environment and inputs.
DVC may reuse cached stage outputs. Source cleanup can invalidate a recorded stage
dependency; no pipeline or lockfile regeneration is performed by this cleanup.

Training currently uses YOLOv8n and the existing DagsHub tracking destination.
Configure your own tracking credentials locally using
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`, either in the
environment or the ignored `.env` file. See `src/train.py` for current behavior;
configurable tracking and broader model selection belong to the implementation plan.

Training recreates the ignored root `data.yaml`. Generation can recreate ignored
letter-resource caches. Neither needs to be committed.

Optional real-flight images belong in `data/test_real/images/`. The final script
skips if none are present. Its predictions are visual examples; quantitative
accuracy requires labeled evaluation data.

## Current generation behavior

- `data_generation.n` means images per class.
- `data_generation.size` is a positive integer and produces exactly
  `size × size` detection images. Rectangular backgrounds are stretched to that
  square before objects and labels are drawn. This differs from training `imgsz`.
- PNG, JPG, and JPEG extensions are accepted case-insensitively. Backgrounds are
  validated and sorted before selection; corrupt images are skipped with a warning.
- Missing/unusable backgrounds and invalid output sizes fail before previous
  generated outputs are cleared.
- Incomplete letter-resource caches are rebuilt automatically. Generation fails
  explicitly if its retry budget cannot produce the requested count.
- Class IDs are `0 = Hexagon` and `1 = Triangle`, defined in
  `src/class_names.py`. Changing class order changes existing label meanings.

## Regression checks

With the project dependencies installed, run:

```bash
python -m unittest discover -s tests -v
```

The tests use temporary fixtures rather than regenerating the project's datasets
or starting training. Training-import checks require genuine YOLO dependencies.

The issue 3–7 corrections can change generated images and boxes. Existing datasets
and checkpoints are not rewritten automatically. Compare results using the same
code, inputs, configuration, and dependency versions. See
[repository_verification.md](repository_verification.md) for recorded evidence.
