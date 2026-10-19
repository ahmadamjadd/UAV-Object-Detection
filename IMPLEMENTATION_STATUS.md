# Team Foxtrot implementation handoff

Updated: **19 October 2026, Asia/Karachi**.

## Objective and scope

Build Option C: a local Python application serving the browser UI and API, with training on each member's machine and results in configured MLflow. Follow [implementation_plan.md](implementation_plan.md): H1–H7, foundation exit check, then V1–V8 in order. Jetson deployment and cloud training CI/CD are excluded. Preserve existing CLI/DVC behavior and issue 3–7 corrections.

## Working rules and authorization

- H1 and the remaining H3 work were expressly authorized as whole milestones, including agent command execution. **Those exceptions are complete. H4 is not authorized.**
- From H4, return to the user's manual tiny-step workflow. State the milestone, one proposed outcome, purpose, and expected file changes; wait for explicit approval. Normally supply code for the user to copy and commands for them to run. Do not independently edit source or run tests/builds/scripts/installations/servers unless newly authorized.
- Agent read-only repository inspection remains allowed: pwd/ls/rg, cat/head/tail/read-only sed, and read-only Git status/diff/log/show. No commits, Git mutations, unrelated refactors, deletion, or dependency changes without explicit instruction.
- Explain changes and unfamiliar terms simply in chat. For commands requested of the user, provide working directory, purpose, important flags, effects, and expected output; wait for results and explain them. Never request secrets.
- Passing tests alone do not authorize advancement. A reported failure requires a focused fix within the current step. Stop for review after each step; an existing instruction to continue remains valid when awaiting clarification of its test result.
- Update this handoff at **whole-milestone completion**, rather than after each edit/message. Keep it factual for agents; mentoring explanations belong in chat. Keep the plan's concise progress checklist consistent.
- At a fresh session, read this file, the plan, and applicable AGENTS.md instructions; recheck actual files/local changes because handoffs can be stale. A handoff never grants permission.
- Subagents, if expressly used, must follow the same scope, command, approval, and verification restrictions, with nonoverlapping edits and parent review. None were used for H3 completion.

## Current milestone and approval state

**H1, H2, and H3 implemented; H3 completion verified by the agent. Awaiting user review. H4–H7 and V1–V8 are not implemented.**

The user manually completed H3.1–H3.8 and confirmed 39 workspace tests passed. They then explicitly asked the agent to finish all H3 and run any needed commands, returning to manual tiny steps from H4. Agent completion covers claiming, exact file identities, cleanup, migrations, and acceptance evidence below. No website, HTTP server, actual worker, training execution, or MLflow upload exists yet.

## Implemented work

| Area / paths | Current outcome |
| --- | --- |
| H1: pyproject.toml, requirements.txt, requirements-dev.in, .python-version, .nvmrc, frontend manifests/lock, src/foxtrot package boundaries | Installable foxtrot-local 0.1.0; pinned Linux x86_64 / Python 3.12 baseline and frontend development dependencies |
| H2: src/foxtrot/config/{training,job,dataset,snapshot,checkpoint,evaluation,result,event}.py; tests/test_config_contracts.py | Strict/frozen versioned request/result contracts; submitted/resolved snapshots; dataset/checkpoint/evaluation identities; finite JSON event payloads |
| H3: src/foxtrot/persistence/workspace.py | External workspace validation/marker; managed jobs/settings/models/datasets/tracking-outbox directories; rejects redirected directories |
| H3: src/foxtrot/persistence/database.py | Transactional schema migrations 1→2→3, jobs/artifacts tables, foreign-key-enabled connections; one-active-job unique constraint; rejects unsupported schemas and symlinked SQLite files/sidecars |
| H3: src/foxtrot/persistence/files.py | Canonical UUID/job-file paths, symlink/traversal guards, exclusive flushed writes, Linux directory fsync, SHA-256/size fingerprinting with concurrent-change checks |
| H3: src/foxtrot/persistence/jobs.py | Owned job folders, submitted/resolved/provenance files, idempotent recorded submission, manifest hashes, frozen JobRecord/JobClaim snapshots, transactional queued→preparing reservation, explicit unregistered-folder cleanup |
| H3: src/foxtrot/persistence/artifacts.py | Exact completed artifact/checkpoint path/hash/size registration and verified retrieval; rejects changed identities, aliases, managed configuration paths, and external paths |
| Tests: existing tests/test_workspace*.py preserved; new tests/test_workspace_persistence.py | 27 added acceptance checks for claims, artifacts, migration/transaction failures, cleanup, and restart persistence; 66 workspace tests total |
| Docs: docs/persistence.md, docs/development.md, README.md, implementation_plan.md | Storage contracts, entry points, migration/cleanup boundaries, verification commands, and current progress |

No dependencies, generator/training/split code, DVC files, existing datasets, or model outputs were changed during H3 completion.

## Decisions and limits

- Submission identity compares validated **submitted choices**, ignoring object-key order. Adding an explicitly supplied default is a different submission. A new key always permits a new job. Matching retries preserve the original resolved defaults and current state.
- Store configurations before publishing a queued row. Files and SQLite are separate durability domains: failed preparation may leave an owned unregistered folder, which is never treated as queued work. No automatic deletion occurs.
- Configuration SHA-256 values are stored in the job record. Reads/retries/claims validate both manifests, explicit-value consistency, and hashes. New jobs also have file-level provenance.json; runtime/source/training provenance is expanded by later pipeline workflows.
- Version-2 jobs have no prior hashes. First successful validated configuration access pins their existing files and preserves their resolved defaults. Earlier file contents cannot be authenticated retroactively.
- Claim lookup/update uses BEGIN IMMEDIATE plus a database one-active-job constraint. One preparing/training/evaluating row is permitted per workspace. Claim tokens/timestamps are reservations, not process-liveness proof. Reopening never resets or relaunches them.
- An active stale record blocks another claim. H4 will add lifecycle transitions/events/reconciliation; H5 will add actual process ownership and cancellation. Storage currently provides only the initial queued→preparing transition.
- A version-2 database with multiple active rows cannot satisfy the new constraint: migration fails and rolls back without losing records. It is never silently repaired or discarded.
- Only completed immutable artifacts should be registered after their writers stop. Required paths/hashes are explicit; no scanning for a newest checkpoint. Mutable logs are not registered as immutable artifacts while still being written.
- Temporary cleanup requires an exact UUID, owned marker, and absence of a database record; it refuses **all registered jobs**, including finished/unsynchronized ones. Stop standalone preparation helpers before cleanup. Normal submissions/cleanup use the same database write lock. Nested symlink targets survive removal. Unmarked legacy folders are not adopted for cleanup.
- Initialize workspace/database before starting concurrent API/worker requests. Recursive cleanup and directory durability are verified on the H1 Linux target; other platforms are unverified.
- H1 target: Ubuntu 24.04.4 x86_64, Intel i5-13420H CPU, Python 3.12.3. Node 22.17.0/npm 10.9.2 are development tools. User chose their machine for now; second-machine installation and Docker are deferred. No GPU support has been verified. Keep current dependency pins; Optuna waits until V5.

## Verification

### Agent-executed H3 checks — 19 October 2026

All commands used repository root /home/muhammad_ahmad/Documents/Foxtrot/MlOps.

| Command / check | Reported outcome |
| --- | --- |
| .venv/bin/python -B -m unittest discover -s tests -p 'test_workspace*.py' -v before edits | 39/39 passed in 0.434s |
| Same focused command after initial additions | 63/63 passed in 2.759s; later additions are included in the full run below |
| .venv/bin/python -B scripts/verify_baseline.py after all implementation changes | **91/91 passed in 71.802s**: 66 workspace tests + 4 H2 contract tests + 21 original issue 3–7 regressions |
| Protected-file comparison performed by verify_baseline.py | **1,216 protected files unchanged**, including datasets/outputs, generator/resources, shared classes, legacy train/test/split, params and DVC files |
| Installed-package import smoke within verify_baseline.py | foxtrot-local 0.1.0 imported outside checkout; package-boundary imports did not load training, MLflow, or dotenv |
| Final static review | Syntax/whitespace checks passed for 7 H3 Python files; git diff --check -- README.md passed; plan/handoff approval state reviewed for consistency |

Acceptance fixtures demonstrate:
- Two jobs retain independent resolved settings and exact checkpoint references across a **fresh Python process**; retry returns the original job and a saved claim remains unchanged.
- Concurrent same-key submissions create one queued row/folder. Concurrent claims reserve one active job, even with multiple queued jobs.
- A child exits deliberately with code 23 during an uncommitted SQLite transaction; reopening recovers the queued state. Code 23 is expected fixture behavior, not a failed check.
- Mid-preparation write failure publishes no queued row; explicit cleanup removes only the owned unregistered folder. Registered jobs, unrelated folders, and external symlink targets are preserved.
- Migration failures roll back schema changes and retain legacy records. Version-2 manifests acquire hashes without changing old resolved defaults.
- Valid-JSON configuration tampering, missing/redirected files, changed/replaced artifacts, and unsafe paths fail visibly.
- Fixture checkpoint bytes are not real models. No application server, actual training, tracking call, or model-quality evaluation was run.

No H3 test failure remained at completion. Final documentation was updated after the full run; no production code changed afterward.

### Earlier evidence

- H1 agent checks under its separate exception: editable environment 21/21 regressions in 58.542s; fresh same-machine wheel environment 21/21 in 57.674s. Both preserved 1,216 protected files. Dependency checks, CPU Torch/Torchvision operations, and temporary frontend build passed.
- H2 user reported individual checks and the completion verifier passed; full original output/timing was not supplied. H3's full run now independently verifies the retained H2 checks.
- Historical 10 October remediation evidence remains in repository_verification.md; it is distinct from current H3 evidence.
- User manually reported H3 checks passing through H3.8, including all 39 tests. Agent independently reran those before completing H3.

## Existing local changes to preserve

HEAD remains **030c45fa75fb272bb6d0c0641209a9775d7125d0** (Completed step 7). These are uncommitted working-tree changes; do not reset/restore/clean them.

- Pre-existing modifications: .gitignore, README.md, dvc.yaml, project_roadmap.md, requirements.txt, src/data_generation/DataSetGeneration.py, src/data_generation/Shapes.py, src/train.py.
- Pre-existing deletions: data.yaml, mlops_journey.md, notebooks/train_on_colab.ipynb, src/data_generation/LetterDataSetGeneration.py, src/data_generation/divide_dataset.py, src/data_generation/divide_letter_dataset.py, src/data_generation/res/Color.json, step7b_plan.md, yolov8n.pt. Do not recreate them automatically.
- Preserve untracked docs/plans/research, H1/H2/H3 package/tests, helper modules, frontend/scripts, and the existing files named tempfile and unittest. H3 did not inspect or remove those two files.
- Issue 3–7 corrections: sorted/seeded generation, square-size and glyph validation, corrected geometry/boxes, bounded background preflight preserving old outputs, lightweight shared class metadata and no training-import .env reads.
- Keep n as images per class and generator size separate from training imgsz. Preserve all six original regression files.
- No applicable AGENTS.md was found in the repository or ancestor chain during inspection.

## Exact next proposed action

Wait for user review of completed H3. If explicitly asked to continue, propose the first **tiny H4** step (define execution/tracking states and legal transitions), explain its purpose and expected file paths, and wait for implementation approval as required. Provide code and commands for the user to apply/run; agent commands return to read-only inspection. Do not implement H4 or treat this handoff as authorization.

No H3 blocker remains. The foundation exit check is still unmet because H4–H7 have not been delivered.
