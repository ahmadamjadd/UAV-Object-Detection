# Foxtrot 0.9.0: release record (V8)

Status: **release candidate**. Every slice V1–V8 is implemented and verified on the
development machine; the two-member pilot on separate computers is still to be done by
the team (see "Not yet verified").

## Install and start

Requirements: Ubuntu Linux x86_64, Python 3.14, about 7 GB for the environment, 5 GB+
free for work. No Node or frontend tools are needed (the interface ships inside the package).

```bash
git clone <repository> && cd MlOps
python3.14 -m venv .venv
.venv/bin/pip install --require-hashes --no-build-isolation -r requirements.txt
.venv/bin/pip install --no-deps --no-build-isolation -e .
.venv/bin/foxtrot --workspace ~/foxtrot-workspace --check      # setup check
.venv/bin/foxtrot --workspace ~/foxtrot-workspace --pipeline-adapter ultralytics
```

Then in **Setup**: run the system check, set your **member name** (shown on every run in
shared MLflow), add a tracking destination and **Test connection**, install pretrained
weights. Dataset generation needs the source checkout (it runs `src/data_generation`).

## Supported workflows

| Workflow | Screen | Docs |
| --- | --- | --- |
| Import a YOLO dataset (training, or real validation/test images) | Datasets | [training.md](training.md), [evaluation.md](evaluation.md) |
| Dataset from a Git/DVC revision | Datasets → From Git/DVC | [evaluation.md](evaluation.md) |
| Generate a Foxtrot dataset with previews | Generate | [generation.md](generation.md) |
| Train (20 models, 3 start modes, 38 settings, JSON editor) | New training | [training.md](training.md) |
| Evaluate a checkpoint on labeled real images; previews on unlabeled photos | Evaluate | [evaluation.md](evaluation.md) |
| Tuning studies: grid / random / TPE, model comparison, recipe tuning, pruning, repeats | Studies | [studies.md](studies.md) |
| Follow progress, cancel, results, MLflow links | Activity | [application.md](application.md) |
| Resume, retry, new experiment, continue study, upload problems | Recovery | [recovery.md](recovery.md) |
| System check, member name, storage cleanup | Setup | this file |

**Models:** YOLOv8, YOLO11, YOLO12, YOLO26 detection × n/s/m/l/x, from scratch, from
official pretrained weights (fingerprinted), from an earlier job's checkpoint, or resumed.

**Parameter coverage** (pinned Ultralytics `default.yaml`, 113 keys, shown on the New
training screen): 39 exposed, 15 managed by Foxtrot, 46 for other modes/tasks, **13 not
offered, each with a recorded reason**: `time`, `save_period`, `single_cls`, `rect`,
`profile`, `multi_scale`, `compile`, `channels_last`, `nbs`, `conf`, `iou`,
`max_det`, `save_json` (fixed evaluation thresholds, GPU-only or unverified options).
Coverage is therefore **not complete**; these are recorded gaps, not hidden ones.
Generator: all `params.yaml` settings are exposed; command-line-only options are listed
with reasons.

## Requirement audit (implementation plan section 4)

| Section | Status |
| --- | --- |
| 4.1 Free operation, tracking destinations | Local MLflow (SQLite), team-hosted MLflow server and DagsHub supported; nothing requires a paid service |
| 4.2 Reusable pipeline components | Generator wrapper, split, trainer, evaluator, uploader are separate, explicit-input components; the legacy CLI/DVC path is unchanged |
| 4.3 Workspaces, immutable configuration | Per-job folders, hashed submitted/resolved configurations, idempotent submission |
| 4.4 Parameter coverage, schema | Schema-driven form and JSON editor with shared validation; coverage report (gaps above) |
| 4.5 Model registry | 20 detection models; checkpoint hash and start mode recorded for every run |
| 4.6 Dataset preparation | Import, generate, DVC; image/label/class/box/split/duplicate checks; group-aware splits; identity caching |
| 4.7 Fair evaluation, checkpoint identity | Fixed `foxtrot-eval-v1`; exact best.pt scored and hashed; frozen validation for recipe tuning; test data never a tuning objective |
| 4.8 Job lifecycle | Durable states/events, owned processes, cancellation, recovery holds, restart reconciliation |
| 4.9 Hyperparameter tuning | Grid/random/TPE, log/conditional spaces, equal/combined budgets, pruning within a model, ask-and-tell with one coordinator, resume of studies |
| 4.10 MLflow logging contract | Params, per-epoch metrics, evaluation, artifacts, provenance, study parent/child runs, member tag |
| 4.11 Outages and redirects | Durable queue, back-off, exact run association, classified errors, "upload pending" state, optional redirect after sync |
| 4.12 Screens and API | Setup, Datasets, Generate, New training, Studies, Evaluate, Activity, Recovery |
| 4.13 DVC integration | Exact revision/commit/md5 recorded; browser training never returns a cached training result |

## Tracking destinations: privacy and cost

| Destination | Collaboration | Privacy | Cost |
| --- | --- | --- | --- |
| Local MLflow (`sqlite:///…/mlflow.db`) | single computer | stays on the machine | free |
| Team-hosted MLflow server (`http://host:port`) | whole team | on the team's own machine/network | free (open source) |
| DagsHub (`https://dagshub.com/<user>/<repo>.mlflow`) | whole team, browser access | hosted service; private repositories need access control on their side | free tier; storage limits apply (quota errors are shown as such) |

Note: MLflow 3.17 refuses plain-folder `file://` stores; use `sqlite:///…`.
Credentials are stored write-only in the workspace (`settings/secrets.json`, mode 600)
and are never shown, logged or uploaded.

## Storage and cleanup

Setup → Storage shows disk use per area and offers only safe removals: finished
background-run folders, interrupted build leftovers older than six hours, and
intermediate files of jobs that **finished and are fully uploaded** (best.pt, data.yaml
and results are kept). Active, held, resumable or not-yet-uploaded work and all datasets
are never offered; the plan is recomputed before anything is deleted.

## Upgrades

Workspace format `foxtrot-workspace-v1`, database schema 6 (unchanged since H6).
V2–V8 added optional fields with defaults, so older datasets read as `training`, older
jobs as `pretrained` with `optimizer: auto`. Verified by opening a copy of a real V1/V2
workspace: datasets verified against their hashes, jobs loaded, system check passed.
Upgrade = stop Foxtrot (so no job is running), update the checkout, reinstall from
`requirements.txt`, start again. Do not swap the environment while jobs are running.

## Evidence (20 November 2026, Ubuntu 24.04 x86_64, CPU, Python 3.14.4)

- `scripts/verify_baseline.py`: all tests pass; 1,216 protected files unchanged (see
  docs/project/implementation_status.md for the exact counts per slice).
- Wheel `foxtrot_local-0.9.0` (84 files, interface included) builds and imports outside
  the checkout.
- Live checks with real YOLO: training, DVC import of this repository's `data/split`,
  evaluation with plots, previews, random and TPE studies with MLflow parent/child runs,
  recipe studies, resume after cancel.
- Two independent workspaces ("Member A", "Member B") uploaded to one team-hosted MLflow
  server: same experiment, runs tagged with the right member, artifacts uploaded over HTTP,
  deep links working. A server outage left the job finished locally with
  `tracking_unreachable`; after the server returned the same job uploaded once.

## Not yet verified (documented, not claimed)

- Other operating systems, ARM and Docker.
- **GPU training**: device choice, detection and refusal are implemented and tested; a
  real run on a working GPU is pending (the development laptop's NVIDIA driver is not
  loaded under Secure Boot).
- A two-member pilot on two separate computers (simulated on one machine above).
- Real labeled drone images (evaluation was exercised with generated stand-ins).
- DagsHub with the new V2–V8 features (V1 uploads to DagsHub were verified by the user).
