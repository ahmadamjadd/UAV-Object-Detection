# H5 worker and process supervisor

H5 executes controlled fixtures through a real worker/subprocess lifecycle.
It proves orchestration, not YOLO training or model quality. H6 adds a second,
code-registered pipeline runner (`--pipeline-adapter fixture`) through this same
supervisor; see [adapters.md](adapters.md). The first browser-to-YOLO workflow is V1.
No HTTP product server, browser launcher, or `foxtrot serve` command exists yet.

## How work moves

```text
Validated job and immutable settings (H2/H3)
    → worker holds leases and claims one queued job
    → write explicit execution.json and managed output/log folders
    → save launch intent
    → launch a fresh fixed Python child in its own session
    → save its exact process identity
    → receive READY; send START
    → save phases, progress, stdout and stderr
    → wait for the whole execution session to stop
    → save final state/exit code, exact artifact references, and summary
```

`execution.json` contains the resolved job, its hash, absolute assigned paths,
and trusted fixture settings. The child checks these against the actual managed
job folder and `resolved.json`. It cannot choose a different output directory.
Neither the job schema nor runner accepts an executable, shell command, or
arbitrary module. `Popen` receives a fixed argument list with no shell.

## Code responsibilities

| Module / function | Responsibility |
| --- | --- |
| `execution.worker.Worker.run` | Hold leases, reconcile stopped owners, claim sequentially, supervise, then poll for more work |
| `execution.locking.worker_lock` | Nonblocking kernel lease; refuses a competing worker and releases automatically on process exit |
| `execution.protocol` | Strict versioned request, fixture settings, and READY/phase/update/completion messages |
| `execution.child.load_request` | Validate exact job paths and configuration identity before execution |
| `execution.child.run_fixture` | Produce controlled progress/output/failures and optional loader children; never train a model |
| `execution.supervisor.Supervisor.run` | Prepare owned files, record launch intent/identity, launch the fixed child, and handle setup failures |
| `Supervisor._monitor` | Drain all pipes, persist events, consume cancellation, enforce stop deadlines, capture exit, and finalize |
| `execution.process_control` | Find verified members of the owned session and signal them using Linux process handles |
| `persistence.commands.request_cancel` | Persist a cancellation command; unstarted jobs cancel immediately, active jobs wait for their owner |
| `persistence.lifecycle.transition_execution` | Commit final state, sanitized reason, exit code, finish time, and event atomically; cancellation wins a racing completion |
| `persistence.recovery.settle_failed_launch` | Let a still-live owner settle its own failed spawn after no child started or its known child session was stopped |
| `persistence.recovery.hold_owned_execution` | Mark interrupted with a recovery hold when bounded cleanup cannot be verified |

The child waits for START, so an immediate success/failure cannot race durable
ownership recording. A dedicated event pipe carries JSON messages. Stdout and
stderr are separate raw log pipes, so normal library output cannot become a
control message. Selectors drain all three without waiting for newline-terminated
logs. Messages have a 64 KiB limit; log-event previews have a 4 KiB limit per
chunk, while the complete raw output remains in the log files. Disk capacity and
log retention controls are later workflows; raw logs are not silently discarded.

The parent owns database writes. The child reports phases/progress through the
pipe; it never impersonates the claim owner. Final log/result artifacts are
registered only after their writers and session children stop. Normal fixture
outcomes save `logs/stdout.log`, `logs/stderr.log`, `output/fixture-result.json`
on success, and `summary.json`. Fixture results explicitly say `fixture: true`;
they are not H2's real training/checkpoint/evaluation result contract.

Setup/supervision errors save a safe error class/reason and the observed exit
code when available. Available raw logs remain local. A filesystem failure can
prevent an optional file summary/artifact registration after a database outcome
commits; that outcome is not rerun or changed to manufacture a different history.
Required real-training artifact policies will be implemented through H6/V1.
H7 will add application log redaction; fixtures contain no credential values.

## One worker and one execution

On the verified Linux target, the default host lease is
`<system-temp>/foxtrot-worker-<uid>/worker.lock`. A lease is an exclusive OS lock
that is released when its owner exits. It permits one worker per OS user
across workspaces. Each workspace also has `settings/worker/worker.lock`. The
directories/files must be private, owned regular paths, not symlinks or hard-link
aliases. Leases use `flock`, not a guessed PID or stale-file timestamp. They are
not inherited by execution children and are never unlinked on release.

SQLite still enforces one active job per workspace. H4 recovery holds also block
claims. This is a local, single-user application; coordination across different
OS accounts or machines is not implemented. Use one OS account/workspace on each
member's machine. The test-only runtime-directory override must not be varied
to run competing production workers. Docker and other operating systems remain
unverified; Linux with pidfd support is required for H5.

## Cancellation and shutdown

For a running job:

1. Persist `cancel_requested`; return promptly without waiting for execution.
2. Worker sends a structured cancel command and grants 1 second for a graceful
   stop/checkpoint opportunity. A real adapter must implement that command in H6.
3. If children remain, signal all verified session members with SIGTERM and
   grant 1 second.
4. If children still remain, use SIGKILL with a 2-second deadline.
5. Record cancelled only after the session is stopped. Unknown/failed cleanup
   becomes interrupted with a hold; the worker exits and does not start more work.

Timeouts are configurable through trusted `SupervisorLimits` in code. An
exception cleanup path may make one additional bounded TERM/KILL attempt.
Linux pidfds pin signal targets against PID reuse. Session enumeration includes
children in different process groups and children whose original leader exited.
Children must remain inside the owned session; deliberately detached sessions
are outside the adapter contract. Failed process creation settles launch intent
only in the original live owner; ambiguous launch intent after a crash remains
held for explicit recovery, as H4 specifies.

SIGINT, SIGTERM, or SIGHUP to the worker stops new claims, cleans up active work,
and records an interruption. An explicit job cancellation takes precedence over
a simultaneous shutdown or completion. Queued jobs remain queued. An idle
worker exits cleanly. SIGKILL/power loss cannot run graceful cleanup: restart
reconciliation retains H4's conservative ownership/hold rules and never retries
an interrupted job automatically.

Browser closure/disconnection does not send any worker shutdown command. H7's
launcher must start the API and worker as separate processes and keep browser
lifetime independent. Explicit application shutdown should request worker stop
and wait for its bounded cleanup; it must not silently orphan active training.
This contract is defined here; the combined API/browser launcher remains H7.

## Development commands

Use the existing H1 environment. From the repository root, with an **absolute,
external, disposable** workspace path:

```bash
.venv/bin/python -B -m foxtrot.execution.worker --workspace /absolute/path/to/fixture-workspace --source-checkout /absolute/path/to/MlOps --fixture
```

Replace the example paths before running. `-B` avoids bytecode caches; `-m`
loads the installed worker module; `--fixture` is mandatory so this infrastructure
runner cannot be mistaken for training. The command creates/reopens its workspace
and leases, processes already registered queued jobs, then waits. Ctrl+C stops
it using the shutdown behavior above. `--once` handles at most one job and exits,
including an empty queue. It does not submit a job, launch an API, or contact
MLflow. Trusted fixture settings can be loaded with `--fixture-settings` from a
JSON file; unknown settings are rejected. Fixture failure is recorded on the job,
while the successfully operating worker can continue to the next job.

For repeatable verification, use the temporary fixtures instead:

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_execution*.py' -v
.venv/bin/python -B scripts/verify_baseline.py
```

Both commands run from the repository root. The first discovers H5 tests and
lists each result. They create and remove temporary workspaces/private leases,
start actual worker/child processes, exercise stop signals, and verify no running
owned session remains. A bounded ASGI health fixture performs ten HTTP dispatches
and database reads during execution; it does not start the product H7 server or
certify browser behavior. The second runs all repository regressions, installed
imports, and protected-file comparisons. Expected success is exit code 0, `OK`,
and unchanged protected files. No production dataset, training, package install,
frontend build, or hosted tracking operation is part of these checks.
