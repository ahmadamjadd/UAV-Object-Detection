"""The study coordinator: one per workspace, the only writer of each study's Optuna storage.

Each step it looks at every running study and moves it forward (ask-and-tell):

1. no active trial → ask Optuna for the next values (or take the next grid point
   or repeat), resolve them into complete settings, save the trial *before*
   creating its job;
2. recipe studies → build the trial's training data in a separate process;
3. create the trial's ordinary training job, with a submission key derived from
   study and trial, so a crash and retry can never create a second job;
4. follow the job: report per-epoch scores to the pruner (if enabled) and cancel
   a pruned trial; when the job ends, tell Optuna complete / failed / pruned.

Trials run one at a time through the normal worker. A restart re-reads the trial
files and continues from where each trial was: completed trials are never
repeated. A study stops launching trials when its budget or time limit is used
up, after a failure (if the policy says so), or when the member stops it.
"""

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from foxtrot.config.job import JobSpec
from foxtrot.config.study import OBJECTIVES, StudySpec, active_params, grid_points, resolve_settings
from foxtrot.persistence.commands import request_cancel
from foxtrot.persistence.events import read_job_events
from foxtrot.persistence.files import job_file, job_path
from foxtrot.persistence.jobs import create_recorded_job, get_job_record
from foxtrot.persistence.start_checkpoints import check_dataset_use

from .store import (
    ACTIVE_TRIAL_STATES, list_study_ids, now, read_json, read_study, read_trials,
    study_folder, write_json, write_state, write_trial,
)

TERMINAL_JOB = {"finished", "failed", "cancelled", "interrupted"}


# Optuna's storage URL for a study (local SQLite; never shared between machines).
def _storage(folder: Path) -> str:
    return f"sqlite:///{folder / 'optuna.db'}"


# The Optuna study names: one per model for equal budgets, one shared for combined searches.
def optuna_names(spec: StudySpec) -> list[str]:
    if spec.sampler == "repeat":
        return []
    return list(spec.model_ids) if spec.model_budget == "equal" else ["combined"]


# Create (or open) one Optuna study with the right sampler and pruner.
def _optuna_study(spec: StudySpec, folder: Path, name: str, storage=None):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    offset = optuna_names(spec).index(name)
    if spec.sampler == "tpe":
        startup = max(2, min(5, spec.trials_per_model // 2))
        sampler = optuna.samplers.TPESampler(seed=spec.seed + offset, n_startup_trials=startup)
    else:  # random; grid points are queued explicitly
        sampler = optuna.samplers.RandomSampler(seed=spec.seed + offset)
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1) if spec.pruning
              else optuna.pruners.NopPruner())
    study = optuna.create_study(study_name=name, storage=storage or _storage(folder), direction="maximize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    if spec.sampler == "grid" and not study.trials and not study.user_attrs.get("grid_queued"):
        points = grid_points(spec)
        models = [None] if spec.model_budget == "equal" else list(spec.model_ids)
        for model_id in models:
            for point in points:
                study.enqueue_trial({**({"model_id": model_id} if model_id else {}), **point})
        study.set_user_attr("grid_queued", True)
    return study


# Ask Optuna for one trial's values, sampling conditional parameters only when they apply.
def _ask(spec: StudySpec, study, combined_models: list[str] | None):
    trial = study.ask()
    values = {}
    if combined_models:
        values["model_id"] = trial.suggest_categorical("model_id", combined_models)
    for param in spec.search_space:
        if param.condition is not None:
            current = {**spec.base_training, **values}.get(param.condition.param)
            if current is None:
                from foxtrot.config.training import TrainingSettings
                current = getattr(TrainingSettings(), param.condition.param, None)
            if current not in param.condition.values:
                continue
        if param.type == "categorical":
            values[param.name] = trial.suggest_categorical(param.name, list(param.choices))
        elif param.type == "int":
            values[param.name] = trial.suggest_int(param.name, int(param.low), int(param.high),
                                                   step=int(param.step or 1), log=param.log)
        else:
            values[param.name] = trial.suggest_float(param.name, param.low, param.high, step=param.step, log=param.log)
    return trial, values


# Moves every running study forward; runs as a background thread in the launcher.
class StudyCoordinator:
    def __init__(self, workspace: Path, checkout: Path, *, interval: float = 3.0):
        self.workspace, self.checkout = Path(workspace), Path(checkout)
        self.interval = interval
        self._stop, self._wake = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._lease = None
        self._preparing: dict[tuple[str, int], subprocess.Popen] = {}
        self._storages: dict[str, object] = {}  # one Optuna storage (database connection pool) per study
        self._studies: dict[tuple[str, str], object] = {}

    # The Optuna study for (study, name), opened once and reused; closed by stop().
    def _optuna(self, study_id: str, spec: StudySpec, name: str):
        key = (study_id, name)
        if key not in self._studies:
            import optuna
            folder = study_folder(self.workspace, self.checkout, study_id)
            if study_id not in self._storages:
                self._storages[study_id] = optuna.storages.RDBStorage(_storage(folder))
            self._studies[key] = _optuna_study(spec, folder, name, self._storages[study_id])
        return self._studies[key]

    # Close every Optuna database connection this coordinator opened.
    def close(self) -> None:
        for storage in self._storages.values():
            storage.engine.dispose()
        self._storages.clear()
        self._studies.clear()

    # Take the workspace's coordinator lease (one coordinator per workspace) and start stepping.
    def start(self) -> bool:
        import fcntl
        from .store import studies_root

        path = studies_root(self.workspace, self.checkout) / "coordinator.lock"
        stream = path.open("a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return False  # another Foxtrot process coordinates this workspace
        self._lease = stream
        self._thread = threading.Thread(target=self._loop, name="foxtrot-studies", daemon=True)
        self._thread.start()
        return True

    # Stop stepping; running trial jobs continue in the worker and are picked up next time.
    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        for process in self._preparing.values():
            if process.poll() is None:
                process.terminate()
            process.wait(30)
        self._preparing.clear()
        if self._lease is not None:
            self._lease.close()
        with self._lock:
            self.close()

    # Step soon (after a study is created, stopped or resumed).
    def trigger(self) -> None:
        self._wake.set()

    # The background loop.
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception as error:  # never let one bad study stop the loop
                print(f"study coordinator: {error!r}", file=sys.stderr, flush=True)
            self._wake.wait(self.interval)
            self._wake.clear()

    # Advance every running or stopping study by at most one action each.
    def step(self) -> None:
        with self._lock:
            for study_id in list_study_ids(self.workspace, self.checkout):
                _, _, state = read_study(self.workspace, self.checkout, study_id)
                if state["state"] in ("running", "stopping"):
                    self._advance(study_id)

    # One study: follow its active trial, or start the next one, or finish.
    def _advance(self, study_id: str) -> None:
        spec, record, state = read_study(self.workspace, self.checkout, study_id)
        trials = read_trials(self.workspace, self.checkout, study_id)
        active = next((t for t in trials if t["state"] in ACTIVE_TRIAL_STATES), None)
        if active is not None:
            self._follow(study_id, spec, state, active)
            return
        if state["state"] == "stopping":
            return self._finish(study_id, state, "stopped", state.get("reason") or "Stopped by the member")
        if spec.failure_policy == "stop" and any(t["state"] == "failed" for t in trials):
            return self._finish(study_id, state, "stopped", "A trial failed and the failure policy is 'stop'")
        if spec.time_limit_minutes is not None and self._time_used(state) >= spec.time_limit_minutes * 60:
            return self._finish(study_id, state, "finished", "Time limit reached")
        assignment = self._next_assignment(spec, record, trials)
        if assignment is None:
            return self._finish(study_id, state, "finished", "All planned trials are done")
        self._new_trial(study_id, spec, trials, assignment)

    # Seconds of running time used (across stop/resume).
    @staticmethod
    def _time_used(state: dict) -> float:
        from datetime import datetime
        resumed = datetime.fromisoformat(state["resumed_at"])
        return state.get("time_used_seconds", 0.0) + (datetime.now(resumed.tzinfo) - resumed).total_seconds()

    # Which model / Optuna study the next trial belongs to, or None when the budget is used up.
    def _next_assignment(self, spec: StudySpec, record: dict, trials: list[dict]) -> dict | None:
        if spec.sampler == "repeat":
            return {"repeat": len(trials)} if len(trials) < len(spec.repeats) else None
        if spec.model_budget == "combined" or len(spec.model_ids) == 1:
            name = optuna_names(spec)[0]
            return {"optuna": name} if len(trials) < record["planned_trials"] else None
        per_model = record["planned_trials"] // len(spec.model_ids)
        counts = {m: sum(1 for t in trials if t["model_id"] == m) for m in spec.model_ids}
        # Round-robin: the model with the fewest trials goes next, so a time limit stays fair.
        open_models = [m for m in spec.model_ids if counts[m] < per_model]
        if not open_models:
            return None
        model_id = min(open_models, key=lambda m: (counts[m], spec.model_ids.index(m)))
        return {"optuna": model_id, "model_id": model_id}

    # Ask for values, resolve them, and save the new trial before anything else happens.
    def _new_trial(self, study_id: str, spec: StudySpec, trials: list[dict], assignment: dict) -> None:
        folder = study_folder(self.workspace, self.checkout, study_id)
        number = len(trials)
        trial = {"number": number, "state": "asked", "asked_at": now(), "job_id": None, "dataset_id": spec.dataset_id,
                 "value": None, "error": None, "intermediate": [], "event_cursor": 0, "optuna": None}
        if "repeat" in assignment:
            item = spec.repeats[assignment["repeat"]]
            trial.update(model_id=item.model_id, params={"seed": item.seed}, source_trial=item.source_trial,
                         training={**item.training, "seed": item.seed}, recipe=None)
        else:
            study = self._optuna(study_id, spec, assignment["optuna"])
            self._fail_orphans(study, study_id, assignment["optuna"], trials)
            combined = list(spec.model_ids) if spec.model_budget == "combined" and len(spec.model_ids) > 1 else None
            optuna_trial, values = _ask(spec, study, combined)
            model_id = values.pop("model_id", None) or assignment.get("model_id") or spec.model_ids[0]
            params = active_params(spec, values)
            trial.update(model_id=model_id, params=params,
                         optuna={"study": assignment["optuna"], "number": optuna_trial.number,
                                 "trial_id": optuna_trial._trial_id})
            try:
                settings, recipe = resolve_settings(spec, params)
            except ValueError as error:  # e.g. a sampled combination that settings refuse
                trial.update(state="failed", error=f"Invalid configuration: {error}"[:500], finished_at=now(),
                             training=None, recipe=None)
                write_trial(self.workspace, self.checkout, study_id, trial)
                self._tell(study_id, spec, trial, "fail")
                return
            trial.update(training=settings.model_dump(mode="json", exclude_none=True),
                         recipe=recipe.model_dump(mode="json") if recipe else None)
        write_trial(self.workspace, self.checkout, study_id, trial)

    # Mark Optuna trials left RUNNING without a Foxtrot record (crash between ask and save) as failed.
    def _fail_orphans(self, study, study_id: str, name: str, trials: list[dict]) -> None:
        import optuna

        known = {t["optuna"]["number"] for t in trials if t.get("optuna") and t["optuna"]["study"] == name}
        for frozen in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            if frozen.number not in known:
                study.tell(frozen.number, state=optuna.trial.TrialState.FAIL)

    # Move the active trial along: prepare data, create its job, watch it, record the outcome.
    def _follow(self, study_id: str, spec: StudySpec, state: dict, trial: dict) -> None:
        if trial["state"] == "asked" and state["state"] == "stopping":  # saved but never started
            trial.update(state="cancelled", error="Not started: the study was stopped", finished_at=now())
            write_trial(self.workspace, self.checkout, study_id, trial)
            return self._tell(study_id, spec, trial, "fail")
        if trial["state"] == "asked":
            if trial.get("recipe"):
                trial["state"] = "preparing_data"
                write_trial(self.workspace, self.checkout, study_id, trial)
            else:
                return self._launch(study_id, spec, trial)
        if trial["state"] == "preparing_data":
            return self._prepare(study_id, spec, state, trial)
        record = get_job_record(self.workspace, self.checkout, trial["job_id"])
        if record.execution_state in TERMINAL_JOB:
            return self._settle(study_id, spec, trial, record)
        if state["state"] == "stopping" and state.get("stop_active") == "cancel" and not record.cancel_requested:
            trial["stop_cancelled"] = True
            write_trial(self.workspace, self.checkout, study_id, trial)
            request_cancel(self.workspace, self.checkout, trial["job_id"])
            return
        if spec.pruning and trial["state"] == "running" and trial.get("optuna"):
            self._check_pruning(study_id, spec, trial)

    # Build a recipe trial's training data in a separate process; launch the job once it exists.
    def _prepare(self, study_id: str, spec: StudySpec, state: dict, trial: dict) -> None:
        folder = study_folder(self.workspace, self.checkout, study_id) / "trials"
        outcome_path = folder / f"{trial['number']}.prepare.json"
        key = (study_id, trial["number"])
        if state["state"] == "stopping" and state.get("stop_active") == "cancel":
            process = self._preparing.pop(key, None)
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(10)
            trial.update(state="cancelled", error="Stopped while preparing data", finished_at=now())
            write_trial(self.workspace, self.checkout, study_id, trial)
            return self._tell(study_id, spec, trial, "fail")
        if outcome_path.is_file():
            outcome = read_json(outcome_path)
            finished = self._preparing.pop(key, None)
            if finished is not None:
                finished.wait(60)  # it wrote its outcome and is exiting; reap it
            if not outcome["ok"]:
                trial.update(state="failed", error=f"Data preparation failed: {outcome['message']}", finished_at=now())
                write_trial(self.workspace, self.checkout, study_id, trial)
                return self._tell(study_id, spec, trial, "fail")
            trial.update(dataset_id=outcome["dataset_id"], dataset_cached=outcome["cached"])
            write_trial(self.workspace, self.checkout, study_id, trial)
            return self._launch(study_id, spec, trial)
        process = self._preparing.get(key)
        if process is None or process.poll() is not None:  # first time, or after a restart
            command = [sys.executable, "-B", "-m", "foxtrot.studies.recipes", "--workspace", str(self.workspace),
                       "--source-checkout", str(self.checkout), "--study", study_id, "--trial", str(trial["number"])]
            with (folder / f"{trial['number']}.prepare.log").open("a") as log:
                self._preparing[key] = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                                        stderr=subprocess.STDOUT, start_new_session=True)

    # Create the trial's ordinary training job (idempotent: same study + trial = same job).
    def _launch(self, study_id: str, spec: StudySpec, trial: dict) -> None:
        request = {
            "dataset_id": trial["dataset_id"], "model_id": trial["model_id"], "initialization": spec.initialization,
            "tracking_destination_id": spec.tracking_destination_id, "training": trial["training"],
            "study": {"study_id": study_id, "trial_number": trial["number"]},
        }
        try:
            check_dataset_use(self.workspace, self.checkout, JobSpec.model_validate(request))
            directory = create_recorded_job(self.workspace, self.checkout, request,
                                            f"study-{study_id}-trial-{trial['number']}")
        except ValueError as error:
            trial.update(state="failed", error=f"Could not create the trial job: {error}"[:500], finished_at=now())
            write_trial(self.workspace, self.checkout, study_id, trial)
            return self._tell(study_id, spec, trial, "fail")
        trial.update(state="running", job_id=directory.name, started_at=now())
        write_trial(self.workspace, self.checkout, study_id, trial)

    # Report new per-epoch objective scores to Optuna; cancel the job if the pruner says so.
    def _check_pruning(self, study_id: str, spec: StudySpec, trial: dict) -> None:
        import optuna

        key = OBJECTIVES[spec.objective][1]
        events = read_job_events(self.workspace, self.checkout, trial["job_id"], after_sequence=trial["event_cursor"])
        if not events:
            return
        trial["event_cursor"] = events[-1].sequence
        study = self._optuna(study_id, spec, trial["optuna"]["study"])
        handle = optuna.trial.Trial(study, trial["optuna"]["trial_id"])
        for event in events:
            payload = event.payload
            if event.event_type == "progress" and isinstance(payload.get("epoch"), int) and key in payload:
                handle.report(float(payload[key]), payload["epoch"])
                trial["intermediate"].append([payload["epoch"], payload[key]])
        if trial["intermediate"] and handle.should_prune():
            trial["state"] = "pruning"
            request_cancel(self.workspace, self.checkout, trial["job_id"])
        write_trial(self.workspace, self.checkout, study_id, trial)

    # Record how the trial's job ended and tell Optuna.
    def _settle(self, study_id: str, spec: StudySpec, trial: dict, record) -> None:
        if record.execution_state == "finished":
            from datetime import datetime, UTC
            from foxtrot.config.result import JobResult
            try:  # the worker commits "finished" a moment before it writes result.json
                result = JobResult.model_validate_json(
                    job_file(job_path(self.workspace.resolve(), trial["job_id"]), "result.json").read_bytes())
            except ValueError:
                finished = datetime.fromisoformat(record.finished_at.replace(" ", "T")).replace(tzinfo=UTC)
                if (datetime.now(UTC) - finished).total_seconds() < 300:
                    return  # look again on the next step
                result = None
            trial["finished_at"] = now()
            if result is None:
                trial.update(state="failed", error="The job finished but its result file is missing")
                write_trial(self.workspace, self.checkout, study_id, trial)
                return self._tell(study_id, spec, trial, "fail")
            trial.update(state="complete", value=getattr(result.evaluation, OBJECTIVES[spec.objective][0]),
                         checkpoint_sha256=result.checkpoint.sha256)
            outcome = "complete"
        elif record.execution_state == "cancelled" and trial["state"] == "pruning":
            trial.update(state="pruned", finished_at=now(), error="Stopped early by the pruner (scores below the median)")
            outcome = "pruned"
        elif record.execution_state == "cancelled":
            trial.update(state="cancelled", finished_at=now(), error="Cancelled" + (
                " when the study was stopped" if trial.get("stop_cancelled") else ""))
            outcome = "fail"
        else:  # failed or interrupted: keep the real reason (e.g. out of memory); never retried silently
            details = record.error_details
            trial.update(state="failed", finished_at=now(),
                         error=f"{details.code}: {details.message}" if details else record.execution_state)
            outcome = "fail"
        write_trial(self.workspace, self.checkout, study_id, trial)
        self._tell(study_id, spec, trial, outcome)

    # Tell Optuna a trial's final state (repeat studies have no Optuna storage).
    def _tell(self, study_id: str, spec: StudySpec, trial: dict, outcome: str) -> None:
        if not trial.get("optuna"):
            return
        import optuna

        study = self._optuna(study_id, spec, trial["optuna"]["study"])
        number = trial["optuna"]["number"]
        if study.trials[number].state != optuna.trial.TrialState.RUNNING:
            return  # already told (e.g. a retry after a crash)
        if outcome == "complete":
            study.tell(number, trial["value"])
        else:
            study.tell(number, state=optuna.trial.TrialState.PRUNED if outcome == "pruned"
                       else optuna.trial.TrialState.FAIL)

    # Mark the study stopped or finished, keeping the running time for a later resume.
    def _finish(self, study_id: str, state: dict, final: str, reason: str) -> None:
        state.update(state=final, reason=reason, finished_at=now(), stop_active=None,
                     time_used_seconds=self._time_used(state))
        write_state(self.workspace, self.checkout, study_id, state)


# Member actions (called by the API): stop, resume.
def stop_study(workspace, checkout, study_id: str, active: str) -> dict:
    _, _, state = read_study(workspace, checkout, study_id)
    if state["state"] not in ("running", "stopping"):
        raise ValueError(f"The study is already {state['state']}")
    state.update(state="stopping", stop_active=active,
                 reason="Stopped by the member" + (" (active trial cancelled)" if active == "cancel" else ""))
    write_state(workspace, checkout, study_id, state)
    return state


# Continue a stopped or finished study within its remaining budget; completed trials are kept.
def resume_study(workspace, checkout, study_id: str) -> dict:
    _, _, state = read_study(workspace, checkout, study_id)
    if state["state"] not in ("stopped", "finished"):
        raise ValueError(f"The study is {state['state']}; only stopped or finished studies can continue")
    state.update(state="running", reason=None, resumed_at=now(), finished_at=None)
    write_state(workspace, checkout, study_id, state)
    return state
