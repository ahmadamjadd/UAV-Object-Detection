# The local application (website, API and worker)

One command starts the local website/API and the worker. This document began with the
H7 shell; each slice's additions are listed below in order, and the current pages are
described under *Interface redesign*.

## Start it

From the repository root, with an absolute workspace outside the checkout:

```bash
.venv/bin/python -B -m foxtrot.api.launcher --workspace /abs/foxtrot-workspace --source-checkout /abs/MlOps --pipeline-adapter fixture
```

It initializes the workspace/database, starts the worker as a separate process,
serves `http://127.0.0.1:8765/`, and opens a browser tab (`--no-browser` to skip).
`--port` changes the port, `--no-worker` serves the website/API only. Omitting
`--pipeline-adapter` runs the H5 infrastructure fixture instead. Ctrl+C stops the
website, then asks the worker to stop and waits for its bounded cleanup; queued
jobs stay queued. Closing the browser never stops anything.

If another worker already holds the lease, the launcher says so and keeps serving;
that worker processes the queue.

## Processes

```text
launcher ──┬── uvicorn (in-process): FastAPI API + built interface, 127.0.0.1 only
           └── worker (separate process, own session) ── supervisor ── child per job
```

The API only records and reads state in SQLite; it never trains.

## Code responsibilities

| Module | Responsibility |
| --- | --- |
| `api.app.create_app` | Routes, security headers, static interface, fallback page when not built |
| `api.security` | Loopback Host check, same-origin + session token for state changes |
| `api.settings` | Tracking destinations; credentials in private `settings/secrets.json`; log redaction |
| `api.launcher` | One command: initialize, start worker, serve, coordinated shutdown |
| `frontend/src` | React shell: Setup, New training, Activity (polling with event cursor) |
| `api/static/` | Vite build output shipped in the Python package |

## API

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Liveness and database schema version |
| `GET /api/capabilities` | Version, models, devices, adapters, whether real training exists, destinations |
| `GET /api/schemas` | JSON Schema of the job specification (forms read defaults from it) |
| `GET /api/session` | Per-launch session token (same-origin pages only) |
| `GET /api/datasets` | Registered dataset manifests |
| `GET /api/jobs`, `GET /api/jobs/{id}` | Job list; detail with resolved settings and tracking run link |
| `GET /api/jobs/{id}/events?after=N` | Events after the last seen sequence |
| `POST /api/jobs` | Submit `{submission_key, request}`; repeats return the same job; field errors are 422 |
| `POST /api/jobs/{id}/cancel` | Durable cancellation request |
| `GET/PUT /api/tracking/destinations[/{id}]` | Public destination list; save destination and write-only credentials |

## Security decisions

- Binds to `127.0.0.1` only; there is no option to listen on the network.
- `Host` must be `127.0.0.1` or `localhost` (blocks DNS rebinding).
- POST/PUT/PATCH/DELETE require the `X-Foxtrot-Session` token, and any `Origin`
  must be this app. No CORS headers are sent, so other sites cannot read the token.
- CSP `default-src 'self'; frame-ancestors 'none'`, `nosniff`, `no-referrer`,
  `no-store` on API responses. Interactive API docs are disabled.
- Credentials are write-only: the browser sees only `has_credentials`. They live
  in `settings/secrets.json` (0600), never in job specs, `tracking.json`, events,
  or exports. URLs with embedded `user:pass@` are rejected. A logging filter
  replaces known credential values with `***`.
- Scope: one OS user on one machine. Other local accounts are out of scope.

## Frontend development

```bash
cd frontend && npm run build    # type-check, then build into src/foxtrot/api/static
cd frontend && npm run dev      # Vite dev server; proxies /api to a running launcher on 8765
```

Team members do not need Node: the built files ship with the package.

## Added in V1

`POST /api/datasets` (import), `GET /api/models` and `POST /api/models/{id}/install`
(pretrained weights), `POST /api/tracking/destinations/{id}/check` (Test connection),
`POST /api/jobs/{id}/retry-sync` and `GET /api/uploads` (automatic uploads); job
detail now includes the official `result`. The launcher also starts the background
uploader, and `--source-checkout` is optional. See [training.md](training.md).

## Added in V2

A **Generate** screen and `/api/generation/...` endpoints (options, preview,
datasets, run status, cancel, preview images). Each run is a separate process;
the launcher cancels an active run on shutdown. See [generation.md](generation.md).

## Added in V3

`GET /api/capabilities` adds `model_details` (20 registered models);
`GET /api/schemas` adds `training_groups`; `GET /api/checkpoints` lists finished
jobs a new job can start from; `GET /api/coverage` gives the parameter coverage
report. `POST /api/jobs` additionally refuses an unusable start checkpoint (422).
The New training screen is built from the schema and has a JSON editor. See
[training.md](training.md#models-and-settings-v3).

## Added in V4

Dataset `role` (training / validation / test); evaluation jobs (`kind: "evaluate"`);
`POST /api/datasets/dvc` and `POST /api/predictions` run in the shared background
runner; an **Evaluate** screen. See [evaluation.md](evaluation.md).

## Added in V5/V6

A **Studies** screen and `/api/studies/...` endpoints; the launcher also starts the study
coordinator (one per workspace). Study trial jobs carry `study` in their specification. See
[studies.md](studies.md).

## Added in V7/V8

A **Recovery** screen (`GET /api/recovery`) with resume / retry / new experiment /
continue study; classified upload errors; "Training finished — upload pending" and an
optional redirect after sync. Setup gains a system check (`GET /api/system/check`,
`foxtrot --check`), the member name (`GET/PUT /api/profile`, `foxtrot.member` tag) and
safe storage cleanup (`GET /api/storage`, `POST /api/storage/cleanup`). See
[recovery.md](recovery.md) and [release.md](release.md).

## Limits

- Progress uses short polling with the event cursor (no Server-Sent Events).
- One job runs at a time per workspace; further jobs wait in the queue.

## Interface redesign (21 November 2026)

The browser interface follows the "foxtrot-ui-design" package (FoxBrain-style shell:
near-black + fox orange, light/dark, sidebar collapsing to icons under 800 px). Six pages,
hash-routed (`#/home`, `#/datasets?add=generate`, `#/train?mode=study`, `#/runs?job=<id>`,
`#/runs?tab=attention`, `#/evaluate?tab=preview`, `#/settings`):

| Page | Contains |
| --- | --- |
| Home | live training card (epoch, mAP50-95 so far, time left, sparkline), needs attention, recent results, setup status, quick actions, first-run steps |
| Datasets | datasets with roles, image counts and warnings; "Add dataset": Generate synthetic (preview with label boxes) / Import a folder / From Git-DVC |
| Train | Single run (schema-built settings, start-from, JSON editor, coverage) and Tuning study (list, new-study form with plan check, study detail, repeat) |
| Runs | all runs with live detail (training and upload shown separately) and Needs attention (resume / retry / new experiment / continue study / holds) |
| Evaluate | score on labeled images (validation preferred; test warned) and preview on unlabeled photos |
| Settings | system check, GPU, tracking destinations (with experiment name), member name, pretrained weights, storage cleanup, about |

Code: `frontend/src/App.tsx` (shell), `ui.tsx` (shared components, routing, polling),
`live.ts` (live job events), `pages/*.tsx`. No new npm packages; fonts are the system
font (or Inter if installed) because the security policy forbids downloaded web fonts.
`GET /api/jobs` now also returns `kind`, `model_id`, `dataset_id`, `study`;
`GET /api/datasets` returns `images` (per split) and `warning_count`.

### Names and training output (21 November 2026)

- **Names**: datasets and runs can carry a member-chosen name (`settings/labels.json`),
  set when importing/generating a dataset or starting a run, and renamed any time
  (`PUT /api/datasets/{id}/name`, `PUT /api/jobs/{id}/name`, optional `name` in
  `POST /api/jobs` and `POST /api/datasets`). IDs and fingerprints remain the identity.
  Lists return `name`; a run's name becomes its MLflow run name (`foxtrot.name` tag) and
  its dataset's name is logged as `dataset.name`. Study trials are named
  "<study> · trial N"; evaluations "Evaluation of <model> on <dataset>".
- **Training output**: Runs shows the selected run's output in a large terminal panel under
  the table (colour/cursor codes removed, progress-bar redraws collapsed, epoch summaries
  highlighted, follows new lines unless scrolled up).

### Early stopping, GPU and experiment names (21 November 2026)

- Train: *Run on* (CPU or a detected GPU, the GPU chosen by default when present) and an
  *Early stopping* group (switch + patience). Details in [training.md](training.md#early-stopping-gpu-and-experiment-names).
- `GET /api/system/gpu?refresh=true|false`: detected GPUs (`id`, `name`, `memory_gb`) or the
  reason there is none; `GET /api/capabilities` lists the same devices; the setup check
  shows it as an informational line. Submissions and studies asking for an absent GPU get 422.
- MLflow experiment = the member name (22 November 2026): each job uploads into the experiment
  named after the member recorded when it finished; a destination's experiment name is only the
  fallback when no member name is set; Test connection names the experiment. The Train page shows
  "Uploaded into the MLflow experiment “X” as run “name”".
