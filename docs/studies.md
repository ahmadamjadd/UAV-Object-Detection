# Tuning studies (V5 and V6)

A **study** runs several training **trials** with different settings and compares them
fairly: same data, same fixed evaluation (`foxtrot-eval-v1`), same objective. Open
**Studies → New study**, press **Check plan** to see how many trials will run, then
**Start study**. Trials are ordinary training jobs: they appear in Activity, can be
opened, and upload to MLflow.

## Defining a study (`config/study.py`)

| Field | Meaning |
| --- | --- |
| Data | a fixed **training** dataset, or a **dataset recipe** to tune (V6) |
| Models | one or several registered models; several → equal budget per model (recommended) or a separately labelled combined search |
| Search | **grid** (every combination), **random**, or **adaptive TPE** (V6) |
| Search space | training settings (e.g. `lr0`) or recipe settings (`recipe.noise`, `recipe.shape_scale.low` + `.span`); number ranges (stepped or logarithmic) or choices; optional condition ("only when optimizer is SGD") |
| Budget | trials per model, optional time limit, max epochs per trial, search seed |
| Objective | `map50_95` (default), `map50`, `precision` or `recall` on the dataset's **validation** split, maximised |
| Failure policy | continue, or stop the study after a failed trial |
| Pruning | V6: stop trials early whose per-epoch score is below the median (random/TPE, within one model) |

Everything is checked before the first trial: unknown or untunable settings, grids that
are not finite or exceed 500 combinations, epochs above the per-trial limit, and
combinations the training settings refuse (e.g. `optimizer: auto` with `lr0`) are
rejected with the offending combination. A recipe range is tuned as *low + span*, so
every sampled pair is a valid range. Generator ranges (variation inside one dataset) and
tuning ranges (choices between trials) stay separate.

**Planned trials:** grid = combinations × models (equal) or combinations (combined);
random/TPE = trials per model × models (equal) or trials per model in total (combined);
shown before starting, e.g. "12 training trials at most (3 models × 4 trials each)".

## How it runs (`studies/coordinator.py`)

One coordinator per workspace (a lock in `studies/coordinator.lock`), started by the
launcher, is the only writer of each study's Optuna storage (`studies/<id>/optuna.db`,
local SQLite). Each step, per running study (ask-and-tell):

1. ask Optuna for values (grid points are queued in Optuna; repeats are fixed), resolve
   them into complete settings, and **save the trial before creating its job**;
2. recipe studies: build the trial's training data in a separate process;
3. create the trial job with the key `study-<id>-trial-<n>`, so a crash and retry can
   never create a second job;
4. follow the job: with pruning, report each epoch's score and cancel a pruned trial;
   when it ends, tell Optuna *complete*, *failed* or *pruned*.

Trials run one at a time through the normal worker. Equal budgets take turns between
models (fair under a time limit). The budget counts attempts: failed, pruned and
cancelled trials use it too, and out-of-memory or other failures keep their real reason;
settings are never changed silently.

**Stop** ends launching: *let the active trial finish* or *cancel it now*. **Resume**
continues within the remaining budget and time; completed trials are kept and never
repeated. A restart re-reads `trials/N.json` and continues each trial where it was.

## Recipe studies (`studies/recipes.py`, V6)

Before the first trial, the backgrounds are divided once (by the study seed) into
validation and training backgrounds, and validation images are generated with the base
recipe and **frozen** (`studies/<id>/validation/`). Each trial generates training images
only, from training backgrounds only, and uses a copy of the frozen validation set, so a
recipe can never win by making validation easier. Classes and evaluation settings stay
fixed. Trial datasets are identified by recipe + training backgrounds + generator +
frozen validation; an identical recipe reuses its dataset (`dataset_cached`).

## Results

The study page shows progress, outcome counts, trial time, the best trial overall (with
its checkpoint hash and job) and per model, every trial's values, state, score and
reason. **Repeat** (V6) creates a new study that reruns the best trial under several
training seeds and reports each value, the mean and the spread.

**MLflow** (`studies/tracking.py`, in the upload process): one parent run per study
(`foxtrot.kind=study`, the definition as `study.*` params); each trial run is its child
(`mlflow.parentRunId`) with `foxtrot.trial_state` (complete/failed/pruned/cancelled) and
`foxtrot.trial_reason`; pruned and cancelled trials end KILLED, failed ones FAILED. A
trial run is uploaded only after its outcome is recorded. When the study ends and all
trials are uploaded, the parent gets `best/<objective>` (overall and per model),
`study/<state>` counts, `best.*` params, `study/summary.json`, and FINISHED (KILLED if stopped).

Final **test** datasets can never be a study's data or objective
(`DatasetMetadata.can_be_tuning_objective()`); score the chosen model on them afterwards
with **Evaluate**.

## API

`GET /api/studies/options`, `POST /api/studies/plan`, `POST /api/studies`, `GET /api/studies`,
`GET /api/studies/{id}`, `POST /api/studies/{id}/stop?active=finish|cancel`,
`POST /api/studies/{id}/resume`, `POST /api/studies/{id}/repeat` (`top`, `seeds`).

## Limits

- Sequential trials on one machine; the Optuna SQLite file is never shared.
- Repeats are offered for fixed-dataset studies (recipe trials each have their own dataset).
- Pruning uses Optuna's median pruner within one model; cross-model pruning is not offered.
- Resuming an *interrupted training job* is separate (V7).
