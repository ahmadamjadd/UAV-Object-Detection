# Team Foxtrot implementation handoff

Updated: **22 November 2026, Asia/Karachi** (all steps H1–H7 and V1–V8 complete; interface redesign, names, early stopping, GPU selection, member-named experiments, repository tidy-up; committed and pushed on 22 November 2026).

## Objective and scope

Build Option C: a local Python application serving the browser UI and API, with training on each member's machine and results in configured MLflow. Follow [implementation_plan.md](implementation_plan.md): H1–H7, foundation exit check, then V1–V8 in order. Jetson deployment and cloud training CI/CD are excluded. Preserve existing CLI/DVC behavior and issue 3–7 corrections.

## Working rules and authorization

- H1, the remaining H3 work, all H4, and all H5 implementation/testing were expressly authorized as milestone-specific exceptions, including agent edits and command execution. On 10 November 2026 the user asked the agent to "implement H6 and then explain" it; the agent treated that as a whole-H6 exception. On 11 November 2026 the user approved switching to Python 3.14 ("yes switch it") and asked the agent to "complete h7". **These exceptions are complete.**
- **Vertical slices (from 18 November 2026):** each slice is split into small substeps. By default the agent gives the user the code (comment above every class/function; long files split into numbered parts with expected line counts), placement and purpose, the user pastes it, and they confirm understanding before the next substep. The agent tests every substep in a scratch copy first. During V1 the user repeatedly asked the agent to apply specific substeps itself ("paste code and complete step", "complete 1.9 fully", "complete the whole v1"); each such request covers only that substep or slice. On 19 November 2026 the user asked the agent to "complete whole v2" and explain it afterwards (after asking for at most 3–4 substeps per slice); V2 was implemented and verified by the agent. On 20 November 2026 the user asked for V3 in 3–4 substeps under the manual workflow; after reviewing 3.1 they asked the agent to "paste code and test" it, then to "complete v3 on your own" and explain it afterwards. V3 was implemented and verified by the agent. Later on 20 November 2026 the user asked the agent to "use sample images for now and complete v4 and explain as before"; V4 was implemented and verified by the agent. The user then asked the agent to "implement whole of v5" and, while it was in progress, "implement v6 too", to be explained and tested together with V4; V5 and V6 were implemented and verified by the agent. Later on 20 November 2026 the user asked to "push to github and then complete v7 and v8 too": V4–V6 were committed (`941451d`) and pushed, then V7 and V8 were implemented and verified by the agent. No further slice exists; the remaining open items are the two-computer pilot, other OS/GPU, and real flight data.
- From H6, return to the manual workflow with approximately **4–5 meaningful steps per H milestone**, rather than many very small steps. State the milestone, proposed outcome, purpose, and expected files; wait for explicit approval. Normally supply code for the user to copy and commands for them to run. Do not independently edit source or run tests/builds/scripts/installations/servers unless newly authorized.
- Agent read-only repository inspection remains allowed: pwd/ls/rg, cat/head/tail/read-only sed, and read-only Git status/diff/log/show. No commits, Git mutations, unrelated refactors, deletion, or dependency changes without explicit instruction.
- Explain changes and unfamiliar terms simply in chat. For commands requested of the user, provide working directory, purpose, important flags, effects, and expected output; wait for results and explain them. Never request secrets.
- Passing tests alone do not authorize advancement. A reported failure requires a focused fix within the current step. Stop for review after each step; an existing instruction to continue remains valid when awaiting clarification of its test result.
- Update this handoff at **whole-milestone completion**, rather than after each edit/message. Keep it factual for agents; mentoring explanations belong in chat. Keep the plan's concise progress checklist consistent.
- At a fresh session, read this file, the plan, and applicable AGENTS.md instructions; recheck actual files/local changes because handoffs can be stale. A handoff never grants permission.
- Subagents, if expressly used, must follow the same scope, command, approval, and verification restrictions, with nonoverlapping edits and parent review. None were used for H3/H4/H5 completion.

## Current milestone and approval state

**Everything in the plan is implemented.** Foundation H1–H7 and slices V1–V8 are committed and
pushed on `foundation-h4-h7` (last slice commit `24b2300`, handoff `ebdea14`); `main` is unmerged.
The work after V8 (interface redesign, names, terminal, early stopping, GPU selection,
experiment names, tidy-up; sections below) was committed and pushed on 22 November 2026 at the
user's request ("push everything to github"); 299/299 tests, 1,216 protected files unchanged.

Remaining open items are verification only: a two-member pilot on two computers, a training
run on a working NVIDIA GPU, and real labeled flight images.

Slice history (details in the plan's progress list and the verification section):

| Step | Completed | Commit |
| --- | --- | --- |
| H1–H7, Python 3.14 switch | 15 Oct – 11 Nov 2026 | `f60b96d` |
| V1 real training + MLflow (DagsHub verified by the user) | 18 Nov 2026 | `6075f7f`, `dc7222a` |
| V2 dataset generation, V3 models and settings | 19–20 Nov 2026 | `1dfd3e6` |
| V4 evaluation and DVC, V5/V6 tuning studies | 20 Nov 2026 | `941451d` |
| V7 recovery, V8 release 0.9.0 | 20 Nov 2026 | `24b2300` |
| Interface redesign and follow-ups, tidy-up | 21–22 Nov 2026 | pushed 22 Nov 2026 |

## Interface redesign and follow-ups — 21 November 2026

At the user's request, the browser interface was rebuilt from the supplied design package
(`foxtrot-ui-design.zip`, now kept outside the repository in the user's home folder). Six pages
(Home, Datasets, Train, Runs, Evaluate, Settings) replace the eight V1–V8 tabs; all mock data
replaced by the real API; two small API additions (job list summary fields, dataset image counts
and warnings). Verified in headless Chrome on a seeded workspace (real training, study,
evaluation and preview submitted through the pages, no JavaScript errors, light/dark/narrow
layouts); 292/292.

Follow-ups the same day, each at the user's request:

- Member-chosen names for datasets and runs (`persistence/labels.py`, API, MLflow run names; `tests/test_labels.py`) and a large training-output terminal under the runs table (294/294).
- Early stopping as its own group (switch + patience; off sends `patience=0`; `foxtrot.stopped_early` tag).
- "Run on": CPU or a detected GPU (`api/gpu.py`, separate-process PyTorch probe, cached, plain reasons; `GET /api/system/gpu`; absent GPUs refused at submission, study creation and in the trainer; evaluation on the training device); `amp` exposed (coverage 39/15/13/46).
- Editable MLflow experiment name per tracking destination; the Train page names the target experiment and run.
- `tests/test_device_early_stop.py` (3); the study pruning test made robust under load. Full suite 297/297.

The user's laptop has an RTX 3050 6GB, but its NVIDIA driver is not loaded: Secure Boot rejects
the DKMS-built module for the new kernel (unenrolled MOK). Foxtrot correctly reports no GPU; the
user was given the commands to install Canonical-signed proprietary modules (`nvidia-driver-595`
with `linux-modules-nvidia-595-generic-hwe-26.04`).

## Repository tidy-up — 22 November 2026

At the user's request ("remove every file not required … make everything tidy"):

- Removed: the accidentally committed ImageMagick screenshots `tempfile` and `unittest`
  (11 MB each), `build/`, `src/foxtrot_local.egg-info/` (the editable install keeps its own
  metadata), a stray `weights/yolo26n.pt` (now ignored with `/weights/` and `/*.pt`), `__pycache__`
  folders; the design zip moved to `~/foxtrot-ui-design.zip`.
- Moved: the plan and this handoff to `docs/project/` (this file renamed to
  `implementation_status.md`), research/roadmap/verification notes to `docs/history/`;
  all links updated and checked. README rewritten as the entry point (start, pages, docs, layout).
- Comments: every class and function in `src/foxtrot` and the frontend has a plain comment
  (161 Python and 32 TypeScript added); test classes and helpers too (109); `test_*` methods
  are named for what they check.
- Removed unused interface code (`useLoad`, `formatTime`, two CSS rules) and rebuilt the interface.
- Docs updated: training.md (early stopping, GPU, experiment), application.md, release.md
  (coverage 39/13, GPU status), plan status, screens and acceptance status.
- Verified: `.venv/bin/python -B scripts/verify_baseline.py` **297/297 passed** (277.7 s), **1,216 protected files unchanged**; frontend type check (no unused code) and build pass; all Markdown links resolve.

## Implemented work

| Area / paths | Current outcome |
| --- | --- |
| Redesign: frontend/src/App.tsx, ui.tsx, live.ts, pages/{Home,Datasets,Train,Runs,Evaluate,Settings,Studies}.tsx, public/fox-logo.svg; old tab files removed; static rebuilt | Six-page interface with hash routing, shared components, live job events, names, terminal, device choice, early-stopping help, experiment hint |
| Follow-ups: persistence/labels.py, api/gpu.py, api/app.py, api/system.py, config/training.py, pipeline/{ultralytics_adapter,coverage,contracts}.py, execution/child.py, tracking/plan.py, studies/coordinator.py | Member names, GPU detection and refusal, early-stopping switch, `amp`, device-aware evaluation, `foxtrot.stopped_early` and `foxtrot.name` tags, trial names |
| Follow-up tests: tests/test_labels.py (2), tests/test_device_early_stop.py (3) | Names stored/cleaned/renamed; early-stop switch, device values, absent GPU refused with its reason |
| H1: pyproject.toml, requirements.txt, requirements-dev.in, .python-version, .nvmrc, frontend manifests/lock, src/foxtrot package boundaries | Installable foxtrot-local 0.1.0; pinned Linux x86_64 / Python 3.12 baseline and frontend development dependencies |
| H2: src/foxtrot/config/{training,job,dataset,snapshot,checkpoint,evaluation,result,event}.py; tests/test_config_contracts.py | Strict/frozen versioned request/result contracts; submitted/resolved snapshots; dataset/checkpoint/evaluation identities; finite JSON event payloads |
| H3: src/foxtrot/persistence/workspace.py | External workspace validation/marker; managed jobs/settings/models/datasets/tracking-outbox directories; rejects redirected directories |
| H3–H5: src/foxtrot/persistence/database.py | Transactional migrations through schema 5; jobs/artifacts/events, ownership/recovery, cancellation flag, exit code, final timestamp; unsupported/redirected databases rejected |
| H3: src/foxtrot/persistence/files.py | Canonical UUID/job-file paths, symlink/traversal guards, exclusive flushed writes, Linux directory fsync, SHA-256/size fingerprinting with concurrent-change checks |
| H3: src/foxtrot/persistence/jobs.py | Owned job folders, submitted/resolved/provenance files, idempotent recorded submission, manifest hashes, frozen JobRecord/JobClaim snapshots, transactional queued→preparing reservation, explicit unregistered-folder cleanup |
| H3: src/foxtrot/persistence/artifacts.py | Exact completed artifact/checkpoint path/hash/size registration and verified retrieval; rejects changed identities, aliases, managed configuration paths, and external paths |
| H4: src/foxtrot/config/lifecycle.py | Execution/tracking enums and immutable legal-transition tables; strict frozen ErrorDetails |
| H4: src/foxtrot/persistence/events.py, lifecycle.py | Ordered durable events with replay cursors and optional duplicate keys; atomic status/error/event updates; claim-owner write checks; upload and execution remain independent |
| H4: src/foxtrot/persistence/processes.py, recovery.py; jobs.py integration | Linux boot/namespace/start identity, launch-intent and child identity recording, explicit startup reconciliation; surviving/uncertain executions block new claims |
| H5: src/foxtrot/execution/{worker,supervisor,child,protocol,locking,process_control}.py | Sequential worker, fixed structured runner/START handshake, separate progress/log pipes, exact process/session ownership, kernel leases, PID-safe bounded termination; explicit fixture mode only |
| H5: src/foxtrot/persistence/commands.py; jobs.py, lifecycle.py, recovery.py extensions | Durable cancellation requests; atomic final outcome/exit metadata; invalid saved configurations fail visibly; live-owner failed-spawn settlement; uncertain cleanup becomes interrupted with a hold |
| Tests: tests/test_lifecycle_rules.py, test_workspace_lifecycle.py, test_workspace_recovery.py; tests/fixtures/lifecycle_owner.py | 43 added H4 checks, including concurrent events, rollback, migration, fresh-process reads, real owner loss, surviving children, and uncertain launches. Workspace total now 103. Two old sequence assertions updated for submission/claim events. Original generator regressions unchanged |
| H5 tests: tests/test_execution_contracts.py, tests/test_execution_worker.py, tests/fixtures/asgi_health.py | 29 checks for commands/migration/path safety, leases, spawn/attachment/protocol failures, real success/failure/cancel/crash cleanup, restart, output draining, shutdown, and bounded ASGI responsiveness |
| H6: src/foxtrot/pipeline/{contracts,yolo_dataset,fixture,registry}.py | Explicit-input generation/split/train/evaluate contracts and protocols; per-job YOLO data.yaml from recorded classes; code-owned fixture trainer/evaluator; trusted PipelineSettings |
| H6: src/foxtrot/persistence/datasets.py | Workspace datasets/<id>/manifest.json + data/; content SHA-256 over paths and bytes; symlink/ID/path guards; manifests `pending` until V1 validation |
| H6: src/foxtrot/persistence/database.py, tracking.py; src/foxtrot/config/tracking.py | Schema 6: tracking_runs association and ordered, acknowledged tracking_outbox; validated tracking operations |
| H6: src/foxtrot/tracking/{contracts,plan,uploader,fixture}.py | TrackingClient protocol; upload plan from saved results; resumable sanitized uploader; in-memory fixture client |
| H6: src/foxtrot/execution/{protocol,child,supervisor,worker,outcomes}.py | Pipeline request/runner beside the H5 fixture; dataset verified before launch; parent re-verifies result and checkpoint hash; H2 result.json; tracking plan enqueued; `--pipeline-adapter fixture` CLI |
| H6 tests: tests/test_pipeline_contracts.py, test_pipeline_worker.py, test_tracking_outbox.py | 23 checks: contracts, datasets, data.yaml, fixture adapters, real worker success/failure/cancel/missing-or-changed dataset, report tampering, migration 5→6, outbox idempotency, outage retry, ambiguous create, changed artifact |
| H7: src/foxtrot/api/{app,security,settings,launcher}.py | Loopback FastAPI app and API; Host/Origin/session protections; security headers; write-only private credentials with log redaction; one-command launcher with separate worker and coordinated shutdown |
| H7: frontend/{index.html,vite.config.mjs,tsconfig.json,src/*}, frontend/package.json scripts, src/foxtrot/api/static/ | React shell (Setup, New training, Activity with event-cursor polling), built by Vite into the Python package; pyproject package-data ships it |
| H7: persistence jobs.list_job_records, datasets.list_datasets | Read helpers for the activity and training screens |
| H7 tests: tests/test_api.py, tests/test_api_launcher.py | 11 checks: endpoints, host/origin/session refusals, idempotent and field-validated submission, events cursor, cancel, credential privacy/redaction, static serving, real launch → finished job → clean Ctrl+C, port-in-use message |
| V1.1–V1.3: pipeline/dataset_validation.py, persistence/dataset_import.py, api POST /api/datasets, frontend Datasets.tsx | Dataset checker (images, 5-value labels, class range, box bounds, empty splits, missing-label policy, duplicates, orphans; bounded report); staged copy + fingerprint + atomic publish; invalid datasets not imported |
| V1.4–V1.6: pipeline/ultralytics_adapter.py, registry.py, persistence/models.py, worker/launcher options, frontend Weights.tsx | Ultralytics trainer/evaluator; private per-job `YOLO_CONFIG_DIR` + `YOLO_OFFLINE`, integrations off; epoch reporting via on_train_epoch_end flag (Ultralytics re-fires on_fit_epoch_end for the final best.pt check with epoch+1); cancel raised from on_train_batch_end; `--pipeline-adapter ultralytics`, `--initialization pretrained|scratch`; weights from official v8.4.0 URL or local file, SHA-256 record, trainer verifies; dataset hash ignores `*.cache` |
| V1.7–V1.9: tracking/plan.py, execution/outcomes.py, tracking/mlflow_client.py, tracking/sync.py, api/uploads.py, app/launcher/UI | Per-epoch metric points; MLflow client (experiment create/restore, batched params/metrics, artifacts, run URL); sync/check in a separate process with redacted errors and `MLFLOW_DISABLE_TELEMETRY=true`; uploader lock (exit 3 busy); background scheduler 20 s → 600 s back-off; Retry; Test connection; Results/MLflow/local time in Activity |
| V1.10: pyproject `[project.scripts] foxtrot`, launcher default `--source-checkout` | Wheel (63 files, static UI included, no leaked files) installs and runs the whole workflow outside the checkout |
| V1 tests | test_dataset_validation (7), test_dataset_import (6), test_api_datasets (4), test_ultralytics_trainer (4), test_ultralytics_evaluator (2), test_pipeline_ultralytics_worker (3), test_model_weights (6), test_tracking_epoch_metrics (4), test_mlflow_sync (4), test_api_tracking_check (2), test_upload_scheduler (4), test_api_job_result (1); one H6 assertion updated (failed runs upload completed epochs) |
| V2: config/generation.py | Strict GenerationRecipe (params.yaml defaults, `images_per_class`, `canvas_size`, seed, ranges; canonical SHA-256), SplitSettings (ratio, seed, group_by_background), GenerationRequest (absolute backgrounds folder) |
| V2: pipeline/foxtrot_generator.py | Loads the unchanged legacy generator in a separate process and sets its globals from the recipe; explicit background list, output folder, workspace letter cache (temp + rename); cv2/os stand-ins report each saved image (progress, cancel, source background) and skip terminal clearing. Output byte-identical to the original script (200/200 images and labels) |
| V2: pipeline/foxtrot_split.py | Reproducible split; grouped mode keeps each background's images in one split; refuses grouping with one background |
| V2: persistence/dataset_generation.py | Identity = recipe + split + background hashes + generator files/resources + OpenCV/NumPy/Pillow versions → `gen-<16 hex>`; per-ID flock; cache reuse after content verification; changed cache refused, never overwritten; hidden `.generate-*` build → split → validate (missing labels = error) → per-class counts = recipe → manifest/generation.json/validation.json → rename; preview helper |
| V2: pipeline/generation_runner.py, api/generation.py, app/launcher | Separate runner process per run (status.json atomic, cancel file); one run at a time; `/api/generation/{options,preview,datasets,runs/{id},runs/{id}/cancel,runs/{id}/images/{n}.jpg}`; launcher cancels an active run on shutdown |
| V2: tracking/plan.py, execution/outcomes.py | `dataset.generation.*`/`dataset.split.*` params and `dataset-provenance` artifact (copied into the job folder and registered) for generated datasets |
| V2: frontend/src/Generate.tsx, App.tsx, api.ts, Datasets.tsx, styles.css; static rebuilt | Generate screen: settings (advanced under "More settings"), preview gallery with label boxes, progress/cancel, dataset summary and report, "Train on this dataset" preselects it in New training |
| V2 tests: tests/test_dataset_generation.py (15) | Recipe/split rules, MLflow params, real runs: publish/reuse/new identity, protected folders untouched, preview, invalid backgrounds, cancel, changed cache, lock conflict, one run at a time, fixture training job logs and uploads provenance, API options/session/422/404 |
| V7: config/job.py, persistence/start_checkpoints.py, pipeline/ultralytics_adapter.py, pipeline/contracts.py | `initialization: "resume"` with `checkpoint_job_id`; `check_resume_source` (interrupted/failed/cancelled, no hold, same model/dataset, settings equal except workers/cache/patience, last.pt present); trainer copies and hash-checks last.pt, refuses another Ultralytics version or a completed checkpoint, passes `resume` + `save_dir` so output stays in the new job; progress uses real epoch numbers |
| V7: tracking/uploader.py, tracking/sync.py | `classify_tracking_error` (credentials / quota / unreachable / not configured / generic; exception text never shown); `record_destination_failure` shows the reason on queued jobs when the destination cannot be opened (pending→syncing→retrying) |
| V7: api/app.py `/api/recovery`; frontend Recovery.tsx, App.tsx (upload-pending label, optional same-tab redirect after sync), Training.tsx (`initialRequest` for New experiment) | Interrupted jobs with resumability and reason, holds, uploads not in MLflow with errors, studies with remaining trials; resume / retry / new experiment / continue study actions |
| V8: api/system.py, api/launcher.py `--check`, persistence/profile.py, persistence/cleanup.py, api/generation.py `active_run`, execution/outcomes.py + tracking/plan.py member tag, frontend SetupExtras.tsx, pyproject version 0.9.0 | Setup check (Python, libraries via find_spec, pins vs installed, generator, disk, schema, weights, interface); member name → `foxtrot.member`; storage usage and safe cleanup (finished runs, stale build leftovers > 6 h, intermediate files of finished+synced jobs only; plan recomputed before delete) |
| V7/V8 tests: tests/test_recovery_release.py (7); test_tracking_outbox, test_mlflow_sync, test_upload_scheduler assertions updated for named/visible upload errors | Real cancel→resume (continues at the next epoch, old folder unchanged, provenance), resume refusals, error classification without secrets, study settle-once after crash, recovery listing, safe cleanup, member tag, system check incl. CLI |
| Docs: docs/recovery.md, docs/release.md (new), docs/application.md, README.md, implementation_plan.md | Recovery actions and boundaries; install, workflows, coverage gaps, requirement audit, destinations privacy/cost, cleanup, upgrades, evidence, not-yet-verified list |
| V5/V6: pyproject.toml, requirements.txt, docs/development.md | `optuna==5.0.0` + `colorlog==6.12.0` with PyPI hashes (5.1.0 skipped as 4 days old); installed with `--require-hashes --no-deps`; `pip check` clean |
| V5/V6: config/study.py | StudySpec (fixed dataset or recipe; 1–5 models, equal/combined; grid/random/tpe/repeat; trials per model, time limit, max epochs, objective, failure policy, pruning), SearchParam (int/float/log/step/categorical, conditions), recipe scalars and low+span ranges; validates names/types, finite grids ≤500, conditions, and every grid point or range-end combination through TrainingSettings/GenerationRecipe; `grid_points`, `planned_trials`, `resolve_settings` |
| V5/V6: studies/store.py, coordinator.py, recipes.py, summary.py, tracking.py; config/job.py `study` | Study folders (study.json, state.json, optuna.db, trials/N.json, validation/); coordinator (workspace lease, cached Optuna storages closed on stop, ask-and-tell, grid via enqueue_trial because GridSampler calls study.stop() under ask-and-tell, orphan RUNNING trials failed, trial saved before its job, key `study-<id>-trial-<n>`, round-robin equal budgets, time limit, failure policy, median pruning via per-epoch events, waits for result.json after "finished", stop finish/cancel, resume); recipe trials built in a subprocess with frozen validation from held-out backgrounds and identity cache; summaries incl. repeat mean/stdev; MLflow parent run + child tags, uploads deferred until the trial outcome is recorded, parent finalized with best/counts/summary |
| V5/V6: tracking/uploader.py, tracking/sync.py, api/uploads.py, tracking/plan.py, pipeline/fixture.py, pipeline/registry.py | `extra_run_tags` hook; study tags; scheduler treats deferred trial uploads as healthy and wakes for unfinalized studies; opt-in `fixture_config_scores` for tests (default fixture outputs unchanged) |
| V5/V6: api/app.py, api/launcher.py; frontend Studies.tsx (new), App.tsx, api.ts, styles.css; static rebuilt | `/api/studies` options/plan/create/list/detail/stop/resume/repeat; launcher starts/stops the coordinator; Studies screen (definition form with Check plan, list, detail with trials, best overall/per model, stop/resume, repeat) |
| V5/V6 tests: tests/test_studies.py (11) | Spec rules; grid exactness and best-trial identity; seeded random repeatability; stop/resume preserving trials; failures and stop policy; restart reusing the trial job; equal-budget TPE comparison; pruning with MLflow parent/child statuses; recipe study frozen validation/exclusion/cache; repeat seeds via API |
| Docs: docs/studies.md (new), docs/application.md, README.md, implementation_plan.md | V5/V6 usage, mechanics, MLflow, API, limits |
| V4: config/dataset.py, config/job.py, config/evaluation.py, config/lifecycle.py | Dataset `role` training/validation/test (default training) and `can_be_tuning_objective()` (false for test); JobSpec `kind` train/evaluate (evaluate needs initialization checkpoint); `EVALUATION_CONFIG` "foxtrot-eval-v1" (conf 0.001, IoU 0.7, batch 16, max_det 300); optional `per_class_map50_95`; new PREPARING→EVALUATING edge |
| V4: persistence/dataset_import.py (`import_evaluation_dataset`), persistence/start_checkpoints.py (`check_dataset_use`), execution/supervisor.py | Labeled real sets (labels required, optional groups.json, group-preserving `<id>-val`/`<id>-test` division via plan_split); submission rules: training only on training datasets, evaluation only on validation/test with identical class names; worker `dataset_role` failure for jobs queued outside the website |
| V4: execution/child.py, execution/outcomes.py, pipeline/contracts.py, pipeline/ultralytics_adapter.py, tracking/plan.py | Evaluate jobs copy and hash-verify the earlier best.pt, skip training, evaluate with plots; parent registers `plot-*` PNGs from output/evaluation; plan adds `foxtrot.kind` tag, `dataset.role`, `evaluation.*` params, per-class metrics, plot artifacts, no second weights upload; DVC provenance params |
| V4: persistence/dvc_datasets.py, pipeline/prediction.py, pipeline/generation_runner.py, api/generation.py, api/app.py | DVCFileSystem import at an exact revision into a hidden workspace folder (repository/cache untouched; optional remote login from a tracking destination with secret redaction), validation, `dvc.json`; prediction previews (`measured: false`); runner kinds dvc/predict; `/api/datasets/dvc`, `/api/predictions`, import roles; `/api/checkpoints` lists training jobs only; capabilities `dvc_default_repository` |
| V4: frontend Evaluate.tsx (new), Datasets.tsx, Generate.tsx (BoxGallery), Training.tsx, App.tsx, api.ts, styles.css; static rebuilt | Evaluate screen (evaluation job form, unlabeled preview gallery with banner), Datasets roles/test share/groups help and From Git/DVC form, training form shows training datasets only, Activity labels evaluation jobs |
| V4 tests: tests/test_evaluation.py (7) | Roles/tuning guard, evaluate rules and lifecycle edge, group split and required labels, role/class rules incl. worker refusal, fixture evaluation job identities/plan, real YOLO evaluation plots + per-class metrics + prediction preview, DVC two revisions/bad revision/untouched repo/MLflow commit |
| Docs: docs/evaluation.md (new), docs/application.md, docs/training.md, README.md, implementation_plan.md | V4 usage, rules, API, limits |
| V3: config/models.py | Registry: yolov8/yolo11/yolo12/yolo26 × n/s/m/l/x (detect, ultralytics), architecture `<id>.yaml`, weights from the v8.4.0 release; every architecture built offline and forward-passed with the pinned runtime |
| V3: config/job.py | `model_id` = any registry ID; `initialization` pretrained/scratch/checkpoint with `checkpoint_job_id` required exactly for checkpoint (schema_version stays 1; older resolved files load with defaults) |
| V3: config/training.py | 38 settings in groups (training, optimizer, loss, augmentation, runtime) with `group`/`ultralytics` schema metadata and descriptions; image size multiple of 32; `optimizer=auto` refuses lr0/momentum/warmup_bias_lr (verified in pinned trainer code); named optimizers get Ultralytics defaults saved; `ultralytics_arguments()` (cache none→False, freeze 0→None, None omitted) |
| V3: persistence/start_checkpoints.py | Submission check (finished, same model, registered checkpoint unchanged) and training-time lookup from the earlier job's result.json with SHA-256 re-check |
| V3: pipeline/ultralytics_adapter.py, contracts.py, registry.py, worker/launcher | Trainer follows the job's start mode (worker `--initialization job` default; pretrained/scratch force a mode); all settings forwarded plus managed keys (data/project/name/exist_ok/plots/verbose/val/save/resume/pretrained); `training_arguments` log event; TrainingResult `initialization` and `start_weights_sha256`; `ultralytics_workers` None = job setting |
| V3: tracking/plan.py | None values not logged; `model.initialization` and `model.start_weights_sha256` params |
| V3: pipeline/coverage.py, api/app.py | All 113 keys of the pinned `cfg/default.yaml` classified (38 exposed, 15 managed, 14 unsupported with reasons, 46 inapplicable) plus generator options; `/api/coverage`, `/api/checkpoints`, `training_groups`, `model_details`; submit returns 422 for an unusable checkpoint |
| V3: frontend Training.tsx (new), Weights.tsx, App.tsx, api.ts, styles.css; static rebuilt | Schema-built grouped form (model optgroups, start-from choice with installed-weights hint and checkpoint picker, only changed settings sent, Reset), "Edit as JSON" with the same backend validation, coverage card; Setup weights card lists installed models and installs any missing one |
| V3 tests: tests/test_training_settings.py (8); tests/test_api.py one assertion updated | Registry, optimizer rules, limits, argument mapping, start-mode rules, setting metadata, coverage completeness; real YOLO11 scratch job proving args.yaml and MLflow params, then a checkpoint job tracing `model.start_weights_sha256`; API 422s and /api/checkpoints |
| Docs: docs/training.md, docs/application.md, implementation_plan.md | V3 models, start modes, settings, coverage, endpoints |
| Docs: docs/generation.md (new), docs/application.md, README.md, implementation_plan.md | V2 usage, guarantees, identity, API, limits |
| Docs: docs/training.md, docs/application.md, docs/adapters.md, docs/execution.md, docs/lifecycle.md, docs/persistence.md, docs/development.md, README.md, implementation_plan.md | Current storage/state/supervisor contracts, development commands, verified scope, and progress |

No dependencies, generator/training/split code, DVC files, existing datasets, or model outputs were changed during H3/H4/H5 completion.

## Decisions and limits

- Submission identity compares validated **submitted choices**, ignoring object-key order. Adding an explicitly supplied default is a different submission. A new key always permits a new job. Matching retries preserve the original resolved defaults and current state.
- Store configurations before publishing a queued row. Files and SQLite are separate durability domains: failed preparation may leave an owned unregistered folder, which is never treated as queued work. No automatic deletion occurs.
- Configuration SHA-256 values are stored in the job record. Reads/retries/claims validate both manifests, explicit-value consistency, and hashes. New jobs also have file-level provenance.json; runtime/source/training provenance is expanded by later pipeline workflows.
- Version-2 jobs have no prior hashes. First successful validated configuration access pins their existing files and preserves their resolved defaults. Earlier file contents cannot be authenticated retroactively.
- Claim lookup/update uses BEGIN IMMEDIATE plus a one-active-job constraint. Active rows and any recovery hold block new claims. H4 records the calling process's birth identity as well as its claim token. Reopening storage never resets or relaunches work.
- SQLite is the canonical event history so state/error/event/cursor commit together. No separate events.jsonl is written; it is an optional future export. Event keys deduplicate identical active updates. Terminal execution cannot restart through ordinary transitions.
- H5 implements claim → persist launch intent → fixed child in its own session → attach exact identity → READY/START → supervise → stop/wait for all session children → finalize. The H4 helpers remain storage/inspection; H5 owns launching/signals. The child verifies assigned paths and resolved configuration hash. No arbitrary shell/module commands are accepted.
- Host lease defaults to a private system-temp foxtrot-worker-<uid> directory; workspace lease is settings/worker/worker.lock. Kernel flock releases on process exit; files are not unlinked and descriptors are not inherited by children. Default coordination is per OS user across workspaces, not across different users/machines. A custom runtime directory is only for isolated tests.
- Progress uses a dedicated bounded JSON pipe; stdout/stderr are drained independently into complete local files plus bounded event previews. Normal fixture outcomes register exact logs/result/summary references. Fixture results explicitly carry fixture=true, not real checkpoint/evaluation identities. Filesystem failure can prevent a file summary after the database outcome commits; it never authorizes rerun.
- Cancellation persists a command; the worker grants 1s graceful stop, then TERM (1s), then KILL (2s), with bounded exception cleanup if needed. Linux pidfds protect signal targets from PID reuse. Session enumeration includes separate child process groups and surviving children after leader death. A racing cancellation wins atomic finalization. Cleanup uncertainty becomes interrupted/held and stops the worker.
- Worker SIGINT/TERM/HUP stops new claims and interrupts active work after cleanup; queued work remains. Browser lifetime is independent. H7 must implement the combined API/worker launcher and explicit application shutdown. Leases/claims do not depend on a browser tab.
- Reconcile explicitly before startup claims. Matching live owners retain supervision responsibility. Stopped owners become interrupted. Unknown ownership, legacy active records without evidence, surviving children, and ambiguous launches create recovery holds. Re-proven stopped owners/groups can release a hold without requeueing the interrupted job. Reconciliation never launches, signals, or changes configurations.
- PID liveness uses Linux /proc boot ID, namespace, and start ticks; PID reuse is not accepted as the old owner. Other operating systems and Docker are unverified. Children deliberately leaving their owned session are outside this contract.
- An ambiguous launch or unknown legacy owner stays held for explicit V7 recovery. H5 permits only the original live owner to settle its own failed spawn when no child started or its known session stopped. No public force-clear exists. Live owners are not declared dead by elapsed time. Worker startup/idle reconciliation never retries interrupted jobs. Automatic tracking recovery remains V7.
- Schema 4 imports each legacy job's current state at its prior cursor + 1, explicitly marking earlier event history unavailable. It preserves configurations/artifacts and rolls back completely on migration failure.
- A version-2 database with multiple active rows cannot satisfy the new constraint: migration fails and rolls back without losing records. It is never silently repaired or discarded.
- Only completed immutable artifacts should be registered after their writers stop. Required paths/hashes are explicit; no scanning for a newest checkpoint. Mutable logs are not registered as immutable artifacts while still being written.
- Temporary cleanup requires an exact UUID, owned marker, and absence of a database record; it refuses **all registered jobs**, including finished/unsynchronized ones. Stop standalone preparation helpers before cleanup. Normal submissions/cleanup use the same database write lock. Nested symlink targets survive removal. Unmarked legacy folders are not adopted for cleanup.
- Initialize workspace/database before starting concurrent API/worker requests. Recursive cleanup and directory durability are verified on the H1 Linux target; other platforms are unverified.
- H1 target: Ubuntu 24.04.4 x86_64, Intel i5-13420H CPU, Python 3.12.3. Node 22.17.0/npm 10.9.2 are development tools. User chose their machine for now; second-machine installation and Docker are deferred. No GPU support has been verified. Keep current dependency pins; Optuna waits until V5.

## Verification

### V7/V8 completion checks — 20 November 2026

Repository root, `.venv` (CPython 3.14.4), CPU.

| Command / check | Outcome |
| --- | --- |
| `.venv/bin/python -B -W error::ResourceWarning -m unittest discover -s tests -p 'test_recovery_release.py'` | **7/7 passed**; resume tests repeated twice more, stable (first run exposed a wrong test assumption: a cancel can land after epoch 2, so resume must continue after the last reported epoch) |
| `.venv/bin/python -B scripts/verify_baseline.py` | **292/292 passed** (189.2 s), exit 0; installed skeleton 0.9.0; **1,216 protected files unchanged** |
| Wheel | `foxtrot_local-0.9.0-py3-none-any.whl`, 84 files, studies/DVC/system/cleanup modules and the interface included, nothing unexpected, `Requires-Dist: optuna==5.0.0` |
| Upgrade on a copy of `~/foxtrot-workspace-v1` (secrets excluded, database copied with the SQLite backup API; the original untouched) | 2 datasets read (V1 import as training, V2 generated) and verified against their hashes; 2 old jobs load with defaults; `foxtrot --check` all OK (17 pins) |
| Team-hosted MLflow server (`mlflow server`, served artifacts) with two fresh workspaces "Member A"/"Member B" | Both runs in experiment 1 tagged with the right member, artifacts uploaded over HTTP, deep links valid. Server stopped: a new job finished locally, upload `retrying` with `tracking_unreachable` on the job; server restarted: same job synced as exactly one run with its metrics |
| Headless Chrome | Recovery screen listed the interrupted YOLO26 job as resumable with Resume / Retry / New experiment; Setup showed system check (ready), member, storage; storage list overflow fixed afterwards |
| Incidents | `pkill -f` matched the agent's own shell once (it only stopped the test server); processes are now stopped by exact PID |

### V5/V6 completion checks — 20 November 2026

Repository root, `.venv` (CPython 3.14.4 + optuna 5.0.0), CPU.

| Command / check | Outcome |
| --- | --- |
| `.venv/bin/python -B -W error::ResourceWarning -m unittest discover -s tests -p 'test_studies.py'` | **11/11 passed** (≈17–23 s), three consecutive runs; earlier runs found and fixed leaked Optuna database connections, unreaped recipe subprocesses, and a race reading result.json right after "finished" |
| `.venv/bin/python -B scripts/verify_baseline.py` | **285/285 passed** (163.1 s), exit 0; **1,216 protected files unchanged** |
| Live random study (real YOLOv8n, DVC dataset, lr0 log range, 3 trials) | finished in 65 s; best = highest value with its checkpoint hash; `auto`+`lr0` plan refused (422); automatic upload: parent run FINISHED with `best/map50_95`, counts, `study/summary.json`, 3 child runs with `mlflow.parentRunId` and `foxtrot.trial_state` |
| Live V6 study (recipe.noise 10/60, yolov8n vs yolo11n, equal budget, TPE, 4 trials) | round-robin models; frozen validation from 79 of 397 backgrounds, training from the other 318; repeated recipes reused their datasets (`cached`) |
| Headless Chrome on Studies | Detail view (trials, best, repeat) and form ("12 training trials at most (3 models × 4 trials each)") rendered; form column widened afterwards |

### V4 completion checks — 20 November 2026

Repository root, `.venv` (CPython 3.14.4), CPU.

| Command / check | Outcome |
| --- | --- |
| `.venv/bin/python -B -m unittest discover -s tests -p 'test_evaluation.py'` | **7/7 passed** (39.5s), incl. real YOLO evaluation and a throwaway Git+DVC repository with a local remote |
| `.venv/bin/python -B scripts/verify_baseline.py` | **274/274 passed** (241.9s), exit 0; **1,216 protected files unchanged** |
| Live launcher, scratch workspace with isolated lease | Sample "real" set (24 generator images, seed 99, grouped by background as flights) → `sample-real-val`/`sample-real-test`; DVC import of this repository's `data/split` at HEAD (`cd92767`, md5 `3164ad9e…` = dvc.lock); repository `.dvc/cache` (476 files) and git status unchanged; training on a validation set refused; pretrained yolov8n trained on the DVC dataset (mAP50-95 0.872 synthetic val) |
| Headless Chrome on Evaluate | Preview of 6 unlabeled images showed predicted boxes with confidences and the "Preview only" banner; evaluation submitted from the page finished and synced: mAP50-95 0.802 (Hexagon 0.886, Triangle 0.719), `model.start_weights_sha256` = trained best.pt, `evaluation.config` foxtrot-eval-v1, 6 plots in MLflow |
| Bug found and fixed | Confidence input had min 0.01 with step 0.05, so the browser silently refused even the default 0.25; step changed to 0.01 |

### V3 completion checks — 20 November 2026

Repository root, `.venv` (CPython 3.14.4), CPU.

| Command / check | Outcome |
| --- | --- |
| All 20 registry architectures | 7 representative ones (yolov8n/s, yolo11n/m, yolo12n, yolo26n/x) built and forward-passed offline; all 20 IDs follow the same Ultralytics family files; 3 weight URLs checked (HTTP 200) |
| `.venv/bin/python -B -W error::ResourceWarning -m unittest discover -s tests -p 'test_training_settings.py'` | **8/8 passed** (14.6s), including two real YOLO11 jobs |
| `.venv/bin/python -B scripts/verify_baseline.py` | **267/267 passed** (122.3s), exit 0; **1,216 protected files unchanged** |
| Live launcher on a scratch workspace (`--runtime-directory` isolated lease, because the user's own Foxtrot worker was running on `~/foxtrot-workspace-v1`) | Generated 60-image dataset; `auto`+`lr0` refused (422); YOLO26n scratch job with AdamW lr0 0.002, degrees 10, cache ram finished and synced to SQLite MLflow with those `training.*` params and no "None" values; args.yaml matched |
| Headless Chrome on New training | Chose YOLO26 nano, "earlier job's checkpoint", SGD, epochs 1; JSON editor showed the exact request; Start training created the job, which finished/synced with `model.start_weights_sha256` equal to the earlier job's best.pt hash. Layout fixed after the first screenshot (inputs overflowed their columns) |
| Note | The user's running Foxtrot (started 15:42 on `~/foxtrot-workspace-v1`) still runs pre-V3 code until restarted |

### V2 completion checks — 19 November 2026

Repository root, `.venv` (CPython 3.14.4), CPU.

| Command / check | Outcome |
| --- | --- |
| `.venv/bin/python -B -W error::ResourceWarning -m unittest discover -s tests -p 'test_dataset_generation.py'` | **15/15 passed** (12.2s) |
| `.venv/bin/python -B scripts/verify_baseline.py` | **259/259 passed** (118.8s), exit 0; **1,216 protected files unchanged** |
| Wrapper vs original script (copied checkout, params.yaml defaults, `data/raw`, seed 42) | 200/200 images and 200/200 labels byte-identical. The committed `data/generated` predates the issue 3–7 fixes and differs from both |
| Live launcher on a scratch workspace (`--pipeline-adapter ultralytics --initialization scratch`) | Preview 6 images over HTTP; 80-image dataset `gen-cbbd13ead68ab3df` (64/16, backgrounds 55/16, disjoint); 2-epoch YOLOv8n job finished; upload to SQLite MLflow synced with all `dataset.generation.*`/`dataset.split.*` params and `dataset/dataset-provenance.json`; clean Ctrl+C, no leftover processes |
| Headless Chrome on the Generate screen | Preview gallery rendered 6 images with 6 boxes placed on the shapes |
| Note | MLflow 3.17 rejects `file://` tracking stores ("maintenance mode"); local destinations must use `sqlite:///…/mlflow.db` |

### V1 completion checks — 18 November 2026

Repository root, `.venv` (CPython 3.14.4), CPU.

| Command / check | Outcome |
| --- | --- |
| `.venv/bin/python -B scripts/verify_baseline.py` | **244/244 passed** (101.1s), exit 0; **1,216 protected files unchanged**; no stray `mlruns`/`mlartifacts` |
| Same suite with a fresh venv containing only the built wheel (third-party libraries borrowed read-only from `.venv`; a second full 6.6 GB copy did not fit the 7.4 GB `/tmp` tmpfs) | **244/244 passed**; `foxtrot` imported from the wheel, not `src/` |
| Installed `foxtrot` command from `/tmp`: weights from local file, dataset import, real pretrained YOLOv8n 3 epochs at 320, automatic upload to local MLflow | finished + synced in 42 s; precision 0.761, mAP50 0.765, mAP50-95 0.644, identical to the earlier editable run (seed 42) |
| Real runs on the user's 200-image dataset | scratch 3 epochs: mAP50-95 ≈ 0; pretrained 3 epochs: 0.644 |
| User-reported | dataset import, training and DagsHub sync of job eab50a55 (run 7c2b843e…) from the browser |

Plan V1 completion checks: real run with resolved settings, metrics, best checkpoint, evaluation and dataset identity in MLflow (tests + DagsHub); repeated clicks do not duplicate (submission key); reopen preserves progress (event cursor); cancellation stops the owned execution (tests); tracking failure keeps local results and is not shown as synced (tests). **Second-member installation not performed**: deferred by the user to a Docker image after project completion.

### Agent-executed H7 checks — 11 November 2026

Scratch venv (system Python 3.14.4; pydantic 2.13.5, PyYAML 6.0.3, fastapi 0.142.2, starlette 1.7.0, uvicorn 0.54.0 constrained to the lock), `PYTHONPATH=<repo>/src`.

| Command / check | Outcome |
| --- | --- |
| `cd frontend && npm run build` | tsc passed; Vite built index.html, 2.7 kB CSS, 203 kB JS into src/foxtrot/api/static |
| `python -B -W error::ResourceWarning -m unittest discover -s tests -p 'test_api*.py' -v` | **11/11 passed** (first run: 2 launcher failures — uvicorn re-raises Ctrl+C and exits 3 on bind failure — fixed in launcher.py) |
| Manual launch on port 8799 + two POSTed fixture jobs + headless Chrome screenshot | Interface rendered from FastAPI; both jobs shown finished; SIGINT exit 0 |

### Foundation exit check on Python 3.14 — 11 November 2026

Rebuilt `.venv` (CPython 3.14.4), repository root.

| Command / check | Outcome |
| --- | --- |
| `pip install --require-hashes --no-build-isolation -r requirements.txt`; `pip install --no-deps --no-build-isolation -e .`; `pip check` | All 186 locked packages + foxtrot-local installed; hash-checked; no broken requirements; `.venv` 6.6 GB |
| `.venv/bin/python -B scripts/verify_baseline.py` | **197/197 passed in 41.6s**, exit 0; installed-package smoke passed; **1,216 protected files unchanged** |
| YOLOv8n built from `yolov8n.yaml` (no download), CPU forward pass + torchvision NMS | torch 2.14.1+cu130; 3,157,200 parameters; outputs boxes (1,64,2100), scores (1,80,2100); NMS correct |
| `cd frontend && npm run check` | React/TypeScript/Vite temporary build PASS |

Exit-check items: isolated validated job workspace (H2/H3), state survives restart (H3/H4), owned subprocess with events/results/cancel (H5/H6), duplicate submission → one job (H3/H7), browser shell ↔ API (H7), regressions pass (above). GPU, other OS, Docker remain unverified.

### Agent-executed H6 checks — 10 November 2026

Repository root; throwaway venv at the session scratchpad (Python 3.14.x, pydantic 2.13.5, PyYAML 6.0.3 only), `PYTHONPATH=<repo>/src`.

| Command / check | Outcome |
| --- | --- |
| `python -B -W error::ResourceWarning -m unittest discover -s tests -p 'test_pipeline*.py'` | **15/15 passed** in 2.7s |
| `python -B -m unittest discover -s tests -p 'test_tracking*.py'` | **8/8 passed** |
| `python -B -W error::ResourceWarning -m unittest discover -s tests -p 'test_[elptw]*.py'` (execution, lifecycle, pipeline, tracking, training-import, workspace) | 162 run, **160 passed**; 2 failed only for missing packages in the scratch venv: `test_training_import` (ultralytics), H5 HTTP fixture (fastapi) |
| Worker CLI `--pipeline-adapter fixture --once` on a disposable workspace | exit 0; finished job with data.yaml, checkpoint, pipeline-result.json, result.json; `--fixture` with `--pipeline-adapter` rejected |
| Not run | `scripts/verify_baseline.py`, generator regressions, protected-file hash comparison, Python 3.12 target: blocked by the broken `.venv` |

**Fix, 11 November 2026:** child.py's cancel listener read `request.fixture.ignore_cancel`, which crashed for pipeline requests (`fixture` is None), so pipeline cancellation only succeeded via the supervisor's SIGTERM escalation. Fixed; the cancellation test now also asserts no termination signal and exit code 130 (it failed on the old code, passes on the fix). Rerun of `test_[ept][xir]*.py`: 53 run, 51 passed, same 2 environment-only failures.

No source outside H6 scope, legacy CLI/DVC/training/generator file, dataset, or dependency was changed. No training, MLflow call, project install, or commit ran.

### Agent-executed H5 checks — 21 October 2026

All commands used repository root /home/muhammad_ahmad/Documents/Foxtrot/MlOps.

| Command / check | Outcome |
| --- | --- |
| .venv/bin/python -B -m unittest discover -s tests -p 'test_*lifecycle*.py' -q before edits | 25/25 passed in 2.693s |
| .venv/bin/python -B -m unittest discover -s tests -p 'test_workspace*.py' -q after initial integration | 103/103 passed in 11.700s |
| timeout 120s .venv/bin/python -B -W error::ResourceWarning -m unittest discover -s tests -p 'test_execution*.py' -q after final code changes | **29/29 passed in 24.460s**, no resource warning |
| timeout 300s .venv/bin/python -B scripts/verify_baseline.py | **163/163 passed in 151.271s**: prior 134 + 29 H5 checks; installed-package smoke passed |
| Protected-file hashes | **1,216 unchanged**, including original issue 3–7 sources/resources, datasets/outputs, classes, legacy train/test/split, params and DVC files |
| Static review | Ownership/handshake, cancellation/exit atomicity, bounded cleanup, leases, scope reviewed; git diff --check passed |

Real-process fixtures demonstrate graceful cancel, forced KILL of a stubborn
root/loader in separate process groups, cleanup after leader crash, sequential
jobs, rejection of a competing worker across workspaces, preserved queue on
shutdown, and no repeat of completed work on restart. Large stdout/stderr drain
without deadlock. Ten bounded ASGI HTTP dispatches plus persisted job reads
remain responsive during execution; no product/network server was implemented.

Two preliminary focused runs were interrupted when the pinned TestClient harness
hung. Replaced it with the bounded ASGI fixture, without dependency changes.
A later unreaped-process ResourceWarning was fixed and reverified above.
No failure remains. Only documentation changed after the full verifier.
No model training, MLflow call, package install, frontend build, or commit ran.

### Agent-executed H4 checks — 19 October 2026

All commands used repository root /home/muhammad_ahmad/Documents/Foxtrot/MlOps.

| Command / check | Outcome |
| --- | --- |
| .venv/bin/python -B -m unittest discover -s tests -p 'test_workspace*.py' -q before edits | 66/66 passed in 3.741s |
| .venv/bin/python -B -m unittest discover -s tests -p 'test_workspace*.py' -v after all code/test changes | **103/103 passed in 11.657s** |
| .venv/bin/python -B scripts/verify_baseline.py | **134/134 passed in 138.665s**: 103 workspace + 6 lifecycle rules + 4 H2 + 21 original regressions; installed-package smoke passed |
| Protected-file hashes from full verifier | **1,216 unchanged**, including generator/resources, data/outputs, classes, legacy train/test/split, params and DVC files |
| Static review | State tables, transaction/ownership boundaries, migration rollback, and scope reviewed; git diff --check passed |

Controlled fixtures demonstrate ordered progress survives reopening/fresh Python;
loss of a real owner becomes interrupted with its reason/history retained;
surviving children and launch uncertainty block claims; a proven-stopped child
allows hold release. A surviving session child is detected after its leader
exits. No automatic relaunch, training, service startup, or MLflow call occurred.
No H4 test failure remained. Only documentation changed after the full run.
These are historical H4 results; H5 evidence above supersedes their totals.

### Agent-executed H3 checks — 19 October 2026

From the repository root, `.venv/bin/python -B scripts/verify_baseline.py`
passed **91/91 in 71.802s** (66 workspace, 4 H2, 21 regressions), with 1,216
protected files unchanged and installed-package smoke passing. Fresh-process
fixtures verified two jobs/configurations/checkpoint identities, idempotent
submission, concurrent claims, rollback after deliberate exit 23, migrations,
and owned cleanup. No failure remained. H4's full run above supersedes these
totals; fixture checkpoint bytes are not real models.

### Earlier evidence

- H1 agent checks under its separate exception: editable environment 21/21 regressions in 58.542s; fresh same-machine wheel environment 21/21 in 57.674s. Both preserved 1,216 protected files. Dependency checks, CPU Torch/Torchvision operations, and temporary frontend build passed.
- H2 user reported individual checks and the completion verifier passed; full original output/timing was not supplied. H3's full run now independently verifies the retained H2 checks.
- Historical 10 October remediation evidence remains in [docs/history/repository_verification.md](../history/repository_verification.md); it is distinct from current H3 evidence.
- User manually reported H3 checks passing through H3.8, including all 39 tests. Agent independently reran those before completing H3.

## Repository state and changes to preserve

Base `349954e` on `main`. Branch `foundation-h4-h7` (GitHub): `f60b96d` foundation H4–H7 + Python 3.14, `d36695a` plan/status, `6075f7f` V1.1–V1.7, `dc7222a` V1.8–V1.10 + docs (V1 complete); all pushed to GitHub. `main` (tracking DagsHub `origin`) is unchanged and not merged.

- Issue 3–7 corrections: sorted/seeded generation, square-size and glyph validation, corrected geometry/boxes, bounded background preflight preserving old outputs, lightweight shared class metadata and no training-import .env reads.
- Keep n as images per class and generator size separate from training imgsz. Preserve all six original regression files.
- Legacy `src/train.py`, `src/test.py`, `src/split_dataset.py`, `src/class_names.py`, generator, params and DVC files are unchanged and remain the CLI path.
- The built interface in `src/foxtrot/api/static/` is committed deliberately; rebuild it with `cd frontend && npm run build` after frontend changes, and restart the launcher after backend changes.
- User workspace `~/foxtrot-workspace-v1` was deleted entirely at the user's request on 22 November 2026 (fresh start), and the 11 Foxtrot runs on DagsHub were deleted (moved to MLflow's trash); the 5 original-pipeline runs in `YOLOv8_UAV_Detection` were kept. A new workspace holds the user's data and private DagsHub credentials; do not modify it without asking.
- No applicable AGENTS.md was found in the repository or ancestor chain during inspection.

## Exact next proposed action

Everything is committed and pushed on `foundation-h4-h7` (GitHub); `main` is unmerged. Next, when
the user wants: merge into `main` (needs explicit approval), a training run on a working NVIDIA GPU
once the driver loads, the two-computer pilot, and real labeled flight images.

Open items: GPU training, other operating systems and Docker unverified; second-machine install deferred to Docker; ambiguous ownership remains held for V7 recovery; uploads only run while Foxtrot is open.
