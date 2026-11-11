# H3 local persistence

H3 supplies storage primitives for the later API and worker. H4 extends these
with states, durable events, and process reconciliation; see
[lifecycle.md](lifecycle.md). Neither launches training, provides HTTP endpoints,
or contacts MLflow.

## Files and database have separate responsibilities

Each job's folder holds its immutable settings and output files. `app.sqlite`
holds small records used to find jobs, reserve work, and identify exact files.
The database stores artifact paths and hashes rather than model or image bytes.

```text
workspace/                       outside the source checkout
├── .foxtrot-workspace            workspace format marker
├── app.sqlite                   schema version 5, including progress/cancellation
├── settings/
├── models/
├── datasets/
├── tracking-outbox/
└── jobs/<job-id>/
    ├── .foxtrot-job              ownership marker for new job folders
    ├── submitted.json           validated choices supplied by the caller
    ├── resolved.json            those choices plus the original defaults
    ├── provenance.json          job ID and configuration hashes
    └── ...                     explicitly assigned outputs
```

The application must finish workspace/database setup before accepting concurrent
requests. Initialize once in the future launcher before starting API/worker
processes. Existing managed directories and SQLite sidecars cannot be symlinks.
No real workspace is created merely by importing these modules.

## Entry points

Functions take `workspace_root` and `source_checkout` as absolute paths.
Workspace paths may use `~`; the workspace and checkout cannot contain one
another. A new/empty folder or recognized workspace is accepted; unrelated
nonempty folders are refused.

| Module / function | Purpose |
| --- | --- |
| `workspace.initialize_workspace` | Validate and initialize the managed workspace directories |
| `database.initialize_database` | Create or migrate `app.sqlite` without discarding history |
| `database.connect_database` | Open a validated database with foreign keys enabled and guaranteed connection cleanup |
| `jobs.create_recorded_job` | Validate, persist settings, and commit a queued row, or return a matching previous submission |
| `jobs.get_job_record` | Read an immutable metadata snapshot using a fresh connection |
| `jobs.load_job_configurations` | Load saved configurations and verify their recorded hashes |
| `jobs.claim_next_job` | Reserve one queued job and return its identity, claim token, directory, and saved configurations |
| `artifacts.record_job_artifact` | Store an exact completed file's relative path, kind, SHA-256, and byte size |
| `artifacts.get_job_artifact` | Retrieve that exact reference and verify the current file |
| `jobs.cleanup_unregistered_job` | Explicitly remove a named abandoned folder that has an ownership marker and no registered job |

`create_job_directory` and `create_configured_job` remain lower-level helpers.
They do not register queued work. Application submissions should go through
`create_recorded_job`.

`files.py` centralizes UUID/path validation, exclusive writes, file hashes, and
Linux directory synchronization. Configuration, artifact, and cleanup operations
share these checks so they cannot independently reinterpret output paths.

## Submission and immutable configurations

```text
validate request/key
    → begin database write transaction
    → find existing submission
        → same submitted choices: verify saved files and return original job
        → different submitted choices: fail without changing the original
    → otherwise create owned job folder
    → exclusively write and flush settings and file provenance
    → insert job row and configuration hashes
    → commit
```

The submission key identifies one intended request. Retrying it, including after
reopening the application, returns the same job. A new key creates a new job
even with identical settings. Object-key order is irrelevant; explicitly adding
a default is a different submitted choice. Saved resolved defaults are never
recomputed on retry. Editing and rerunning means a new key and a new job.

File hashes detect changes after registration, even when modified files still
contain valid JSON. Both files are also validated against `JobSpec`, and explicit
submitted values must agree with the resolved configuration. A missing, invalid,
redirected, or changed configuration raises an error; it does not trigger a
replacement execution.

Completed files are flushed before a queued row becomes visible. Database and
filesystem writes are not one transaction: a preparation error or crash may
leave an unregistered folder. Such a folder is never discovered as queued work.
Database rollback and explicit temporary cleanup handle the two sides separately.

## Claims and the one-active-job rule

A claim performs lookup and `queued → preparing` inside `BEGIN IMMEDIATE`.
A database constraint also permits at most one `preparing`, `training`, or
`evaluating` row per workspace. Ties in creation time use insertion order.
Another claimant returns `None` if the queue is empty or a job is already active.
Upload state does not reserve the training slot.

The claim token and timestamp are durable reservation metadata. H4 also records
the calling owner's PID and birth identity, and commits a preparing event with
the claim. Reopening storage preserves the claim and never automatically
resets/requeues it. H4 explicitly reconciles stopped/uncertain owners; recovery
holds block new claims even when the old job is interrupted. H5's fixture worker
implements process ownership/cancellation; see [execution.md](execution.md).
Real training remains later work.

## Artifact identity

Register only completed files after their writers stop. Paths are canonical
relative paths inside the selected job; traversal, absolute paths, symlinks,
missing files, and managed configuration/ownership files are refused. An artifact
name identifies one immutable path/hash/size combination within a job. Repeating
an identical registration is safe; changing a registered identity fails.

Reading a reference verifies the file's contents. No search for the newest
`best.pt` occurs. A checkpoint can be represented by the existing H2
`CheckpointReference` using the recorded relative path and SHA-256. Actual model
training and evaluation remain later integrations.

## Migration and cleanup rules

- Version 1: initial empty database. Version 2: job records. Version 3:
  configuration hashes, claim metadata, one-active-job constraint, and artifacts.
  Version 4: durable events/errors, process identities, and recovery holds.
  Version 5: cancellation requests, observed exit codes, and final timestamps.
- Each missing migration runs in order inside one transaction. Unsupported newer
  versions and unrelated unversioned databases are rejected. A migration failure
  rolls back schema changes and preserves existing records.
- Old version-2 jobs lack hashes. Their first successful configuration read,
  retry, or claim validates the saved manifests and records their fingerprints.
  Their earlier contents cannot be independently authenticated retroactively.
- A legacy database with multiple active jobs cannot satisfy version 3's
  constraint. Upgrade fails without discarding its records; it does not silently
  select or interrupt a job. No real workers existed in the prior foundation.
- Temporary cleanup requires the exact UUID, an intact `.foxtrot-job` ownership
  marker, and no database record. Stop any standalone preparation writer first.
  Normal registered submissions and cleanup share the same database write lock.
- Cleanup refuses every registered job, including finished jobs with pending
  uploads, and refuses unmarked legacy folders. It never follows a job symlink
  or nested symlink target. Recursive removal requires Python's symlink-resistant
  implementation, verified on Linux. Nothing is cleaned automatically.
- General history/artifact retention and deletion of completed jobs belong to
  V8. Training/runtime/source provenance and tracking associations are expanded
  by their pipeline/tracking workflows; the current provenance file identifies
  only the local configuration files.

## Verification

From the repository root:

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_workspace*.py' -v
.venv/bin/python -B scripts/verify_baseline.py
```

The first command uses the project interpreter to discover workspace tests in
`tests/`; the quoted pattern selects their filenames, `-v` lists results, and
`-B` prevents bytecode caches. Fixtures create and remove temporary directories,
databases, and sample artifact bytes. Threads simulate competing submissions and
claims; fresh Python processes verify restart persistence and an uncommitted
transaction interrupted by process exit. No model is trained.

The second runs the entire repository suite, including issue 3–7 regressions,
with temporary caches and protected-file hash comparisons. It checks installed
package imports outside the checkout too. Nonzero exit means verification failed;
`OK` and unchanged protected hashes are the expected successful outcome.

Verified environment: Ubuntu x86_64 CPU target, Python 3.14 since 11 November 2026 (H1 used 3.12).
Other operating systems, GPU execution, Docker, a real website, and tracking
uploads are outside this evidence.
