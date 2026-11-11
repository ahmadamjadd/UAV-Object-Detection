# H6 pipeline and tracking adapter boundaries

H6 defines the stable inputs and outputs that real YOLO training (V1), the
generator refactor (V2), and MLflow uploads (V1/V7) will plug into. It proves
the boundary with **fixture** adapters. No YOLO model is trained and no
MLflow server is contacted.

## How a pipeline job runs

```text
registered dataset (datasets/<id>/manifest.json + data/)
    → worker started with --pipeline-adapter fixture
    → supervisor verifies the dataset hash BEFORE launch (else failed: dataset_unavailable)
    → execution.json carries the resolved job, dataset metadata and absolute paths
    → child writes jobs/<job>/output/data.yaml from the dataset's own classes
    → Trainer.train(TrainingInput)       → TrainingResult (exact checkpoint + hash)
    → Evaluator.evaluate(EvaluationInput) → H2 EvaluationResult for that checkpoint
    → child writes output/pipeline-result.json and reports Completed
    → supervisor re-verifies job ID, adapter, dataset identity and checkpoint hash
    → final state commits; result.json (H2 JobResult) is written
    → the job's tracking plan is saved in the outbox (tracking_state stays pending)
    → a separate uploader replays the outbox through a TrackingClient
```

## Code responsibilities

| Module | Responsibility |
| --- | --- |
| `pipeline.contracts` | Typed inputs/results and `Protocol`s for generation, splitting, training, evaluation; progress reporter; `PipelineCancelled`/`PipelineFailure` |
| `pipeline.yolo_dataset` | Per-job YOLO `data.yaml` from recorded metadata; never the checkout's root file |
| `pipeline.fixture` | Contract-following trainer/evaluator with placeholder checkpoint bytes and fixed scores |
| `pipeline.registry` | Trusted `PipelineSettings` and the code-owned adapter table |
| `persistence.datasets` | Register/load workspace datasets; content SHA-256 over every file path and byte; symlinks refused |
| `persistence.tracking` | `tracking_runs` association and ordered `tracking_outbox` (schema 6) |
| `config.tracking` | Validated `TrackingOperation`, `MetricPoint`, `ArtifactUpload` |
| `tracking.contracts` | `TrackingClient` protocol and `RunHandle` (run ID + browser URL) |
| `tracking.plan` | Build a job's upload plan from saved local results |
| `tracking.uploader` | `sync_job_tracking`: resume after the last acknowledged operation |
| `tracking.fixture` | In-memory `RecordingTrackingClient` with transient-failure injection |
| `execution.outcomes` | Parent-side result verification, `JobResult` writing, plan enqueueing |

## Decisions

- **Explicit inputs.** Adapters receive the resolved job, the dataset root, the
  per-job `data.yaml`, and the output folder. They never read `params.yaml`,
  write `<checkout>/data.yaml`, or search `runs/` for a checkpoint. The legacy
  `src/train.py`, `src/test.py`, `src/class_names.py`, and DVC stages are unchanged
  and remain the CLI path; issue 7's lightweight class metadata is untouched.
- **Dataset classes come from the dataset.** Hexagon/Triangle remain the default
  generator classes (V2), but a registered dataset's own mapping is used.
- **Generation interface records legacy meanings**: `images_per_class` is `n`;
  `canvas_size` is the square generator `size` and is distinct from training `image_size`.
- **The child is not trusted.** The supervisor rechecks job/adapter/dataset
  identity and fingerprints the checkpoint before recording `finished`.
- **Adapter failures are specific.** `PipelineFailure(code, message)` becomes the
  job's saved error (e.g. `fixture_training_failed`), not a generic exit code.
- **Upload is separate from execution.** A tracking outage moves only
  `tracking_state` to `retrying`; the execution outcome is unchanged. Saved
  errors include only the exception class, never its text (which may contain URLs/tokens).
- **Resumable, ordered outbox.** Operations are acknowledged in order and never
  deleted. Attempts are counted before each call. Run creation first searches for
  a run tagged `foxtrot.job_id`, so an unrecorded earlier create is reused.
  A crash between a successful call and its acknowledgement may repeat that one call.
- **Artifacts are uploaded by registered name** and their hash is rechecked; a
  changed file is never uploaded. Raw stdout/stderr stay local until H7 adds redaction.
- **Status mapping**: finished → FINISHED; failed/interrupted → FAILED; cancelled → KILLED.

## Limits (deferred to their owning slices)

- Real Ultralytics adapter, epoch-level metric streaming, MLflow client: V1.
- Dataset image/label validation and registering external folders: V1; manifests are `pending`.
- Generator and splitter implementations: V2. Only their interfaces exist.
- Automatic upload scheduling, uploader lease, rebuilding a plan lost to a crash
  between the final commit and enqueueing: V7. Run one uploader at a time.
- Setup failures before a child reports (e.g. `supervision_failed`) write no
  `result.json` and queue no upload.
- Dataset hashing reads every file on each load; caching is a later optimisation.

## Development commands

From the repository root, with an absolute external disposable workspace that
already contains a registered dataset:

```bash
.venv/bin/python -B -m foxtrot.execution.worker --workspace /abs/workspace --source-checkout /abs/MlOps --pipeline-adapter fixture --once
.venv/bin/python -B -m unittest discover -s tests -p 'test_pipeline*.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_tracking*.py' -v
```

`--pipeline-adapter` and `--fixture` are mutually exclusive. Trusted options can
be supplied as `PipelineSettings` JSON via `--pipeline-settings`.
