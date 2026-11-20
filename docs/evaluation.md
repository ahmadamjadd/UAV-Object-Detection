# Versioned datasets and real-image evaluation (V4)

Synthetic validation scores only show how well a model handles more synthetic
images. V4 adds honest checks on **real labeled images**, previews on unlabeled
photos, and training on an **exact DVC dataset version**.

## Dataset roles

Every dataset has a `role` (older datasets are `training`):

| Role | Used for | Rules |
| --- | --- | --- |
| `training` | Training jobs | Imported, generated, or from DVC |
| `validation` | Scoring/comparing models on real labeled images | Never trained on |
| `test` | The final, honest score | Never trained on; `can_be_tuning_objective()` is false, so V5 studies cannot optimise against it |

The website refuses a training job on a validation/test dataset and an
evaluation on a training dataset; the worker enforces the same rule
(`dataset_role`) for jobs queued without the website.

## Registering real labeled images

**Datasets → Import → Use: Real labeled images.** Layout: `images/` and
`labels/` (YOLO format); every image needs a label file. Optional `groups.json`
maps each image to a flight or scene, e.g. `{"0012.jpg": "flight-3"}`.
"Also set aside a final test set (%)" creates `<id>-val` and `<id>-test`,
keeping each group in only one of them (`persistence/dataset_import.py`,
`import_evaluation_dataset`). The group map is stored as `data/groups.json`.

## Evaluating a model

**Evaluate → Evaluate on labeled images**: choose a finished training job and a
validation/test dataset. This creates an **evaluation job** (`kind: "evaluate"`)
that appears in Activity, can be cancelled, and uploads to MLflow like training:

- the earlier job's `best.pt` is copied into the evaluation job and verified by
  SHA-256 (`model.start_weights_sha256`); no training runs (preparing → evaluating);
- the model's classes must equal the dataset's (names and order), checked at submission;
- scores use the fixed configuration **`foxtrot-eval-v1`** (`config/evaluation.py`:
  conf 0.001, IoU 0.7, batch 16, max 300 detections, the checkpoint's own image
  size), logged as `evaluation.*` parameters;
- MLflow gets precision, recall, mAP50, mAP50-95, `eval/class/<name>/mAP50-95`,
  tags `foxtrot.kind=evaluate`, `dataset.role`, and plots under `evaluation/`
  (confusion matrix, PR/P/R/F1 curves). The weights are not uploaded again.

## Previews on unlabeled photos

**Evaluate → Preview on unlabeled photos**: a checkpoint, a folder and a minimum
confidence. Runs in the background (`pipeline/prediction.py`) and shows up to 24
images with predicted boxes and confidences. The result is `measured: false` and
the page says "Preview only: not an accuracy measurement".

## Datasets from Git/DVC

**Datasets → From Git/DVC**: repository (default: this checkout), revision
(commit, tag or branch), DVC-tracked folder, ID, classes and role. DVC's Python
API downloads that version into a hidden workspace folder; the repository and
its `.dvc/cache` are not changed. The dataset is checked and published with
`dvc.json` (repository, revision, exact commit, DVC md5). Jobs on it log
`dataset.dvc.repository/revision/commit/path/md5` to MLflow. For a private DVC
remote (e.g. DagsHub), "Remote login from tracking destination" reuses that
destination's saved username/token without storing it in the dataset.

The command-line pipeline (`dvc repro`, `src/train.py`, `src/test.py`) is unchanged.

## API

`POST /api/datasets` (`role`, `images_ref`, `test_share`), `POST /api/datasets/dvc`,
`POST /api/predictions`, `GET /api/generation/runs/{id}` (shared run status),
`POST /api/jobs` with `kind: "evaluate"`; `/api/checkpoints` lists training jobs only.

## Limits

- Verified with sample images standing in for real drone photos; no real labeled
  flight data is in the repository yet.
- Evaluation is per dataset; per-flight scores are not computed.
- DVC downloads use the repository's own remote configuration; only basic
  username/token authentication is offered.
