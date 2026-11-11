# Team Foxtrot implementation handoff

Updated: **11 November 2026, Asia/Karachi** (H7, Python 3.14 switch).

## Objective and scope

Build Option C: a local Python application serving the browser UI and API, with training on each member's machine and results in configured MLflow. Follow [implementation_plan.md](implementation_plan.md): H1–H7, foundation exit check, then V1–V8 in order. Jetson deployment and cloud training CI/CD are excluded. Preserve existing CLI/DVC behavior and issue 3–7 corrections.

## Working rules and authorization

- H1, the remaining H3 work, all H4, and all H5 implementation/testing were expressly authorized as milestone-specific exceptions, including agent edits and command execution. On 10 November 2026 the user asked the agent to "implement H6 and then explain" it; the agent treated that as a whole-H6 exception. On 11 November 2026 the user approved switching to Python 3.14 ("yes switch it") and asked the agent to "complete h7". **These exceptions are complete. V1 is not authorized.**
- From H6, return to the manual workflow with approximately **4–5 meaningful steps per H milestone**, rather than many very small steps. State the milestone, proposed outcome, purpose, and expected files; wait for explicit approval. Normally supply code for the user to copy and commands for them to run. Do not independently edit source or run tests/builds/scripts/installations/servers unless newly authorized.
- Agent read-only repository inspection remains allowed: pwd/ls/rg, cat/head/tail/read-only sed, and read-only Git status/diff/log/show. No commits, Git mutations, unrelated refactors, deletion, or dependency changes without explicit instruction.
- Explain changes and unfamiliar terms simply in chat. For commands requested of the user, provide working directory, purpose, important flags, effects, and expected output; wait for results and explain them. Never request secrets.
- Passing tests alone do not authorize advancement. A reported failure requires a focused fix within the current step. Stop for review after each step; an existing instruction to continue remains valid when awaiting clarification of its test result.
- Update this handoff at **whole-milestone completion**, rather than after each edit/message. Keep it factual for agents; mentoring explanations belong in chat. Keep the plan's concise progress checklist consistent.
- At a fresh session, read this file, the plan, and applicable AGENTS.md instructions; recheck actual files/local changes because handoffs can be stale. A handoff never grants permission.
- Subagents, if expressly used, must follow the same scope, command, approval, and verification restrictions, with nonoverlapping edits and parent review. None were used for H3/H4/H5 completion.

## Current milestone and approval state

**H1–H7 implemented (fixture adapters only). Foundation exit check passed on Python 3.14 (11 November 2026): 197/197 tests, 1,216 protected files unchanged. Awaiting user review and approval for V1. V1–V8 are not implemented.**

**Python 3.14 switch (11 November 2026):** `pyproject.toml` requires `>=3.14,<3.15`; `.python-version` is 3.14.4; `requirements.txt` regenerated with uv 0.8.22 constrained to the identical 186 H1 versions (only hashes changed; PyPI hashes for all files). The broken 3.12 environment was deleted at the user's request after verification; new `.venv` links to `/usr/bin/python3.14` explicitly. Docs updated. Docker for teammates is planned after project completion, using the same Python 3.14 and lock.

**Git (11 November 2026):** all H4–H7 work and the Python 3.14 switch are committed as `f60b96d` ("Complete foundation H4–H7 and move to Python 3.14") on branch `foundation-h4-h7`, pushed to the `github` remote at the user's request. `main` (tracking DagsHub `origin`) is unchanged and not merged. Working tree clean after the commit.

## Implemented work

| Area / paths | Current outcome |
| --- | --- |
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
| Docs: docs/application.md, docs/adapters.md, docs/execution.md, docs/lifecycle.md, docs/persistence.md, docs/development.md, README.md, implementation_plan.md | Current storage/state/supervisor contracts, development commands, verified scope, and progress |

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
- Historical 10 October remediation evidence remains in repository_verification.md; it is distinct from current H3 evidence.
- User manually reported H3 checks passing through H3.8, including all 39 tests. Agent independently reran those before completing H3.

## Repository state and changes to preserve

Base: `349954e` (Implementing software) on `main`. Foundation commit: `f60b96d` on
`foundation-h4-h7` (GitHub). Do not restore deleted legacy files or reset source
to the older H1 revision.

- Issue 3–7 corrections: sorted/seeded generation, square-size and glyph validation, corrected geometry/boxes, bounded background preflight preserving old outputs, lightweight shared class metadata and no training-import .env reads.
- Keep n as images per class and generator size separate from training imgsz. Preserve all six original regression files.
- Legacy `src/train.py`, `src/test.py`, `src/split_dataset.py`, `src/class_names.py`, generator, params and DVC files are unchanged and remain the CLI path.
- The built interface in `src/foxtrot/api/static/` is committed deliberately; rebuild it with `cd frontend && npm run build` after frontend changes.
- No applicable AGENTS.md was found in the repository or ancestor chain during inspection.

## Exact next proposed action

Foundation is complete. Wait for explicit user approval to start **V1 — train an
existing dataset and open its MLflow result**. V1 scope (plan section 3): real
Ultralytics trainer/evaluator adapter behind the H6 contracts, registering an
existing local dataset from the browser, real MLflow `TrackingClient` behind the
H6 outbox, epoch-level metrics, and the "View in MLflow" result link. Also decide
with the user whether to merge `foundation-h4-h7` into `main` first.

Open items: GPU execution, other operating systems, and Docker remain unverified;
ambiguous ownership remains held for explicit V7 recovery; one uploader at a time.
