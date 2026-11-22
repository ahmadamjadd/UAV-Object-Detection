# Training

Training started as V1, the first complete workflow: import a YOLO detection dataset, train YOLOv8n
for real, follow progress, and find the exact run in MLflow. It builds on the
foundation (H1–H7); see [application.md](application.md) for the website/API
shell and [adapters.md](adapters.md) for the pipeline/tracking boundaries.

## Start

```bash
foxtrot --workspace /abs/foxtrot-workspace --pipeline-adapter ultralytics
# equivalent, from the checkout:
.venv/bin/python -B -m foxtrot.api.launcher --workspace /abs/foxtrot-workspace --pipeline-adapter ultralytics
```

`--source-checkout` is optional; it defaults to the folder that holds the running
Foxtrot code, and the workspace must be outside it. Since V3 each job chooses how
it starts (see below); `--initialization pretrained|scratch` only forces one mode
for every job (testing), and the default `job` follows each job's choice.

## In the browser

1. **Setup → Pretrained weights → Install** (once per model you want to start from
   pretrained weights). Downloads the official `<model>.pt` into `workspace/models/`
   and records its SHA-256.
2. **Setup → Tracking destinations** (optional). For DagsHub: ID `dagshub`, kind
   MLflow server, Tracking URI `https://dagshub.com/<user>/<repo>.mlflow`,
   username, token. **Test connection** checks it.
3. **Datasets → Import a dataset** (once per dataset): absolute folder path, a new
   ID, class names in label order (0, 1, …). The folder is checked, then copied.
4. **New training**: dataset, model, how it starts, tracking destination, and
   grouped settings (see "Models and settings (V3)"). Image size 320 suits the
   Foxtrot generator's 320×320 images.
5. **Activity**: progress per epoch, Cancel, official Results, Upload state,
   **View in MLflow** once synced, **Retry upload** if it failed.

## What happens

| Stage | Module | Guarantee |
| --- | --- | --- |
| Dataset check | `pipeline.dataset_validation` | Readable images; 5-value labels; class IDs in range; boxes inside 0–1; non-empty splits; duplicates and orphans reported |
| Dataset import | `persistence.dataset_import` | Invalid datasets are not imported; valid ones are copied to a staging folder, fingerprinted, published in one rename; the source is never changed |
| Weights | `persistence.models` | Download or local file; zip/size check; SHA-256 recorded; training refuses changed weights and logs the hash it used |
| Training | `pipeline.ultralytics_adapter.UltralyticsTrainer` | Job's own `data.yaml` and output folder; private offline Ultralytics settings with its MLflow/DVC/analytics off; one progress event per real epoch; cancel after any batch; exact `best.pt` + hash |
| Evaluation | `UltralyticsEvaluator` | Re-scores the exact fingerprinted `best.pt` on the validation split at its training image size; per-class mAP logged |
| Upload plan | `tracking.plan` | Every epoch's metrics (step = epoch); official `eval/*` once at the last epoch; failed runs keep completed epochs |
| Upload | `tracking.mlflow_client`, `tracking.sync`, `api.uploads` | Separate process (credentials never in the website); resumable; finds an already-created run by job tag; one uploader per workspace (lock); back-off 20 s → 10 min; Retry resets |

Dataset fingerprints ignore YOLO's `*.cache` files, which Ultralytics writes into
dataset folders. MLflow telemetry is disabled with `MLFLOW_DISABLE_TELEMETRY=true`
(`DO_NOT_TRACK=1` alone does not disable it).

## Command-line equivalents

```bash
python -m foxtrot.persistence.models --workspace W --source-checkout C install yolov8n [--from file.pt]
python -m foxtrot.tracking.sync --workspace W --source-checkout C check dagshub
python -m foxtrot.tracking.sync --workspace W --source-checkout C run [--job ID]
```

`sync run` exits with code 3 and prints `{"busy": true}` while another upload runs.

## Models and settings (V3)

- **Models** (`config/models.py`): YOLOv8, YOLO11, YOLO12 and YOLO26 detection,
  sizes n/s/m/l/x (20 models). Each architecture was built offline with the
  pinned Ultralytics; pretrained weights come from its v8.4.0 release.
- **Start from** (per job): *pretrained* (installed, fingerprint-checked weights),
  *scratch* (random weights from the architecture file), or *checkpoint* (the
  best.pt of an earlier finished job of the same model, checked at submission and
  again by hash before training). MLflow records `initialization`,
  `model.initialization` and `model.start_weights_sha256`.
- **Settings** (`config/training.py`): training, optimizer/learning rate, loss
  weights, augmentation early stopping and runtime groups (39 Ultralytics keys plus the early-stopping switch). Each field
  carries its group, description and Ultralytics key; the form is built from the
  schema and "Edit as JSON" sends the same request through the same checks.
  `optimizer: auto` picks its own `lr0`, `momentum` and `warmup_bias_lr`
  (verified in the pinned code), so setting them with `auto` is refused; a named
  optimizer gets Ultralytics' defaults filled in and saved. All values are logged
  as `training.*` parameters; the job's `output/train/args.yaml` shows what YOLO used.
- **Coverage** (`pipeline/coverage.py`, `GET /api/coverage`, shown beside the
  form): every key of the pinned Ultralytics `default.yaml` is exposed (39),
  managed by Foxtrot (15, e.g. data/project/name/resume), unsupported with a
  reason (13, e.g. evaluation thresholds kept fixed, unverified options), or
  inapplicable to detection training (46). Generator options are listed the same
  way. A test fails if a new runtime key is unclassified or an exposed key does
  not reach training.

## Early stopping, GPU and experiment names

- **Early stopping** (Train › Early stopping): an on/off switch and *patience*. After
  every epoch YOLO scores the validation images (mAP50-95); if the score has not improved
  for *patience* epochs, training stops and the best checkpoint is kept. Off sends
  `patience=0` (never stop early); a patience of at least the epoch count never triggers.
  A run that stopped early says so on the Runs page and gets the MLflow tag
  `foxtrot.stopped_early`.
- **Run on** (Train and tuning studies): CPU or a detected NVIDIA GPU (`cuda:N`).
  `api/gpu.py` asks PyTorch in a short separate process (the website never imports
  PyTorch), caches the answer, and explains a missing GPU in plain words: CPU-only
  PyTorch, no NVIDIA driver, or a driver that is installed but not running (for example
  rejected by Secure Boot after a kernel update). Settings › GPU shows the result and
  *Check again*. A GPU that is not present is refused at submission; the trainer checks
  again before starting (`gpu_unavailable`); evaluation runs on the training device.
  *Mixed precision* (`amp`) is a setting: faster on a GPU, ignored on the CPU.
- **MLflow experiment = member name**: an experiment is a folder of runs. Every run is
  uploaded into an experiment named after the member who trained it (Settings › Member;
  recorded when the job finishes, so renaming later does not move it). The destination's
  experiment name is only a fallback until a member name is set. A study's trials always
  join the experiment of the study's parent run. The run inside carries the run's name.
- **Comparison parameters**: every run logs `run.name` and `run.short_id` (different for each
  run, useful as the first axis of MLflow's parallel-coordinates chart); a study's parent run
  also logs `model_id` and `dataset_id`. For runs uploaded before these existed:
  `.venv/bin/python -m foxtrot.tracking.sync --workspace <ws> --source-checkout . backfill`
  adds only the missing keys (never changes an existing value; safe to repeat).

## Limits

- DVC datasets and real-image evaluation: see [evaluation.md](evaluation.md).
- Raw stdout/stderr logs are kept locally and not uploaded.
- No automatic redirect to MLflow; the View in MLflow button appears once synced.
- Uploads run while Foxtrot is open; a job finished while it was closed is sent at the next start.
- Verified on this Ubuntu x86_64 machine (Python 3.14.4, CPU); GPU training is implemented but not yet verified on a working GPU. A second machine
  is deferred to the planned Docker image after the project is complete.
