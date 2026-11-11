# H4 job states and durable progress

H4 adds truthful local state and history to H3 storage. Its persistence modules
do not launch training, signal processes, serve HTTP, or upload to MLflow.
H5 now supplies fixture supervision/cancellation callers; see
[execution.md](execution.md). Real training, HTTP, and uploads remain later work.

## Two independent states

Execution follows `queued → preparing → training → evaluating → finished`.
Preparation, training, and evaluation may instead end as failed, cancelled, or
interrupted. Draft/validating states are supported for later callers; today's
submission helper validates before creating a queued record.

Tracking follows `pending → syncing → synced`, with retrying/failed branches.
For example, a job can be `execution_state="finished"` and
`tracking_state="failed"`: training succeeded, but upload did not. Upload retry
does not restart training or change its configuration. Recording these states
does not itself perform an upload.

`config/lifecycle.py` defines enums (named choices) and immutable transition
tables. The validators reject skipped stages, plain unvalidated strings, and
restarting a terminal execution. Repeating the current state is a no-op.
`ErrorDetails` holds a short code and nonblank message. Failed/interrupted
execution and failed tracking require a reason. Callers must pass sanitized
messages, never credentials. Full log redaction belongs to H7's logging paths.

## Atomic history

`app.sqlite` schema 4 stores the authoritative progress history in `job_events`.
There is no independently written `events.jsonl` mirror in H4. SQLite lets a
state change, its error details, its event, and its cursor commit together.
An export can be derived later without becoming a second source of truth.

An event has a job ID, increasing sequence, aware UTC timestamp, type, and JSON
payload. Submission records sequence 1 (`queued`), claim records sequence 2
(`preparing`), and later updates continue from there. Ordering uses sequence,
not wall-clock time. SQLite serializes writers with `BEGIN IMMEDIATE`, so
concurrent updates cannot allocate the same number.

`events._append_event` validates and serializes the payload, inserts the event,
and advances the job cursor inside its caller's transaction. If any write
fails, the transaction rolls back. A mutable dict supplied by a caller cannot
subsequently change the saved event. Invalid JSON values, including NaN and
infinity, are rejected.

`append_job_event` accepts progress/log/error updates from the recorded owner
of an active claim. State events are reserved for state-changing functions.
An optional event key makes an identical update repeatable without duplication;
reusing the key for different contents fails. `system:` keys are reserved.
New execution updates after a terminal state are rejected. An `error` event
alone does not finish a job; a fatal error also needs a failed transition.

`read_job_events(after_sequence=12, limit=100)` returns up to 100 updates after
12. A reconnecting browser will save the last sequence it displayed and ask for
the next page. The function reads only committed events. SSE/HTTP endpoints and
browser cursor storage are H7 integrations, not implemented here.

`transition_execution` and `transition_tracking` validate the transition,
update their own status/reason, append a state event, and return a committed
`JobRecord` snapshot. They never edit submitted/resolved configurations.
Repeating a transition adds no event and cannot replace its saved reason.
Successful tracking clears the current tracking error; the old error remains
in history. Entering preparation must use `claim_next_job`, preserving H3's
one-active-job rule.

## Process identity and restart reconciliation

The supported baseline is Linux. A PID is a reusable process number, so H4
also records the machine boot ID, PID namespace, and process start ticks from
`/proc`. A namespace is the process-number view used by a host or container.
These identifiers distinguish a live owner from another process reusing its
number. A changed boot means the old process stopped; a different namespace or
inaccessible evidence is unknown. Docker and other operating systems remain
unverified.

`claim_next_job` records the calling owner's identity along with its claim
token. Active state updates and progress require both that token and the same
calling process identity. The H5 worker receives child progress and
persist it on the child's behalf. The API does not finalize an active job merely
because the user clicked Cancel; H5 persists a cancellation request, has the
worker stop its children, and only then records the outcome.

H5's implemented fixture launch order is:

1. Claim the queued job.
2. Call `begin_process_launch` to persist launch intent **before spawning**.
3. Launch the execution as a direct child in a dedicated session/process group
   (`subprocess.Popen(..., start_new_session=True)` on Linux).
4. Call `attach_execution_process` to save the child's birth identity and clear
   launch intent. It refuses unrelated processes and shared sessions.
5. Supervise execution, receive its progress, wait for its process session to
   stop, and record the final outcome.

These functions only record ownership; they do not perform steps 3 or 5.
Ordinary terminal transitions refuse an unfinished launch, a recovery hold, or
execution children that may still be running. Children must stay in the owned
session: processes deliberately detaching into another session are outside
this contract. H5 must preserve that boundary when launching adapters.

`reconcile_jobs` runs explicitly after startup, before H5 claims new work:

| Evidence | Result |
| --- | --- |
| Matching live owner | Preserve state; that supervisor remains responsible for child exit and final outcome |
| Owner stopped; no ambiguous launch; execution session stopped or never started | Mark interrupted with a durable reason; release the reservation |
| Unknown/legacy owner, surviving execution children, unreadable evidence, or unfinished launch | Mark interrupted with `recovery_required=True`; block all new claims |
| Previously held job, now with both owner and execution definitely stopped and no ambiguous launch | Clear the hold; leave execution interrupted |

Session checks detect surviving children even after their original group leader
exits. Reconciliation never sends signals, starts a replacement, requeues a job,
or edits its configuration. A job interrupted safely releases capacity for a
*different* queued job; it never retries itself automatically.

Ambiguous launch intent and legacy ownership cannot be cleared by guessing.
They remain blocked for an explicit future recovery workflow (V7). H5 allows
only the still-live original owner to settle its own failed spawn after no child
started or its known child session stopped. There is no public force-clear
operation. A live but unresponsive owner is not declared dead by a timeout;
normal child-exit supervision is implemented in H5. Tracking recovery is V7.

## Schema migration

Migration 4 adds events, independent error details, ownership identities,
launch intent, and recovery holds. Existing records gain one import snapshot
at their previous cursor plus one. It explicitly states that earlier progress
history is unavailable; no history is fabricated. Existing configurations,
artifacts, state, and submission identities remain intact. Legacy active rows
without process evidence become interrupted with a hold when reconciled.
Migration failure rolls back schema and records together.

## Verification

Run from the repository root:

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_*lifecycle*.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_workspace_recovery.py' -v
.venv/bin/python -B scripts/verify_baseline.py
```

The first two commands discover lifecycle and recovery tests; `-v` lists each
check and `-B` avoids bytecode cache writes. They create temporary workspaces,
SQLite databases, and controlled Python fixtures. Tests launch no training.
Fixture termination targets only subprocesses created by those tests; attached
children also exit when the test closes their input pipe. Exit/ownership tests
cover surviving descendants, PID reuse, unknown evidence, and launch crash gaps.

The final command runs all repository tests, installed-package import checks,
and protected-file hashes, including the original issue 3–7 regressions. `OK`,
exit code 0, and unchanged protected files mean the checks passed. No model
quality, browser integration, or MLflow synchronization is proven by these H4
checks. H5's separate worker acceptance is documented in [execution.md](execution.md).
