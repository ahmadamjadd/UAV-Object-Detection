# Recovering from failures (V7)

Foxtrot keeps results locally first and uploads separately, so a crash, a closed
laptop or a tracking outage never loses work. The **Recovery** screen lists what
needs attention and the safe way to continue.

## Four different actions

| Action | When | What it does |
| --- | --- | --- |
| **Resume** | training was interrupted, failed or cancelled and saved `last.pt` | a **new job** continues from the last saved epoch (optimizer state included). Model, dataset and settings must be identical; only `workers`, `cache` and `patience` may change. The old job's folder is not touched |
| **Retry from start** | anything that stopped | the same settings again, from the beginning |
| **New experiment…** | you want to change settings | opens New training with the earlier request in the JSON editor |
| **Continue study** | a stopped study, or one that hit its time limit | runs its remaining trials; completed trials are never repeated |

Resume checks, before the job is created: the earlier job stopped early, is not under a
recovery hold, has `output/train/weights/last.pt`, and differs only in allowed settings.
When training starts it copies `last.pt` into the new job, re-checks its SHA-256, and
refuses a checkpoint written by another Ultralytics version or one that already
completed. Ultralytics writes into the new job (`save_dir`), epochs continue with their
real numbers (e.g. 3, 4, …), and MLflow records `initialization=resume`,
`checkpoint_job_id` and `model.start_weights_sha256` (the exact `last.pt`).

## Uploads

- Uploads retry automatically (20 s, doubling to 10 minutes after failures; Retry resets).
- Each job remembers its MLflow run; after a lost response the run is found by its job
  tag instead of creating a second one. Artifacts are re-hashed before every upload.
- Reasons are shown on the job: `tracking_credentials` (check username/token),
  `tracking_quota` (service limits), `tracking_unreachable` (service down or no network),
  `tracking_not_configured`, or a generic failure. Exception text is never shown, since it
  can contain URLs or tokens.
- The Activity screen says **Training finished — upload pending** until the upload is
  done; **View in MLflow** appears only then. Optionally ("Open the MLflow run
  automatically…") the page switches to the run when its upload completes.

## Crash boundaries

| Crash point | What happens on restart |
| --- | --- |
| Preparing / training / evaluating | the job becomes *interrupted* (or held if its processes might still run); Resume or Retry from Recovery |
| Between training finished and the study recording it | the coordinator re-reads the job and records the trial once; Optuna is never told twice |
| Between Optuna ask and saving the trial | the orphan Optuna trial is marked failed; the next trial is asked normally |
| Between creating a trial job and saving its ID | the same submission key returns the same job, never a second one |
| During upload | the queue resumes after the last acknowledged step with the same run |

**Recovery holds**: if Foxtrot cannot prove a job's processes stopped, new training
waits; the worker re-checks when idle and releases the hold once they are gone.

## API

`GET /api/recovery` (interrupted jobs with `resumable`/`resume_reason`, holds, uploads
not yet in MLflow with their error, studies with remaining trials, uploader status);
resume = `POST /api/jobs` with `initialization: "resume"` and `checkpoint_job_id`.
