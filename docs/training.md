# V1: train an existing dataset from the browser and open it in MLflow

V1 is the first complete workflow: import a YOLO detection dataset, train YOLOv8n
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
Foxtrot code, and the workspace must be outside it. `--initialization scratch`
trains from random weights (no pretrained file needed, much weaker results).

## In the browser

1. **Setup → Pretrained weights → Install** (once). Downloads the official
   `yolov8n.pt` into `workspace/models/` and records its SHA-256.
2. **Setup → Tracking destinations** (optional). For DagsHub: ID `dagshub`, kind
   MLflow server, Tracking URI `https://dagshub.com/<user>/<repo>.mlflow`,
   username, token. **Test connection** checks it.
3. **Datasets → Import a dataset** (once per dataset): absolute folder path, a new
   ID, class names in label order (0, 1, …). The folder is checked, then copied.
4. **New training**: dataset, epochs, batch size, image size (320 for the Foxtrot
   generator's 320×320 images), seed, tracking destination ID.
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

## Limits (later slices)

- One model (`yolov8n`), CPU device, a small set of training settings (V3).
- Dataset generation in the browser (V2); DVC datasets and real-image evaluation (V4).
- Raw stdout/stderr logs are kept locally and not uploaded.
- No automatic redirect to MLflow; the View in MLflow button appears once synced.
- Uploads run while Foxtrot is open; a job finished while it was closed is sent at the next start.
- Verified on this Ubuntu x86_64 machine (Python 3.14.4, CPU). A second machine
  is deferred to the planned Docker image after the project is complete.
