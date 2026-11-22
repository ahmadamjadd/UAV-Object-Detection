"""Send pending uploads to MLflow, or test a tracking destination.

Runs as its own short-lived process so MLflow and your credentials are never
loaded into the website server:

    python -m foxtrot.tracking.sync --workspace /abs/ws --source-checkout /abs/MlOps run
    python -m foxtrot.tracking.sync --workspace /abs/ws --source-checkout /abs/MlOps check dagshub
    python -m foxtrot.tracking.sync --workspace /abs/ws --source-checkout /abs/MlOps backfill

Prints one JSON result. Error messages never contain saved credentials.
"""

import argparse
import json
from pathlib import Path
import sys

from foxtrot.api.settings import REDACTED, SettingsStore
from foxtrot.config.job import JobSpec
from foxtrot.execution.locking import WorkerAlreadyRunning, worker_lock
from foxtrot.persistence.jobs import load_job_configurations
from foxtrot.persistence.profile import read_profile
from foxtrot.persistence.tracking import jobs_with_pending_tracking, list_tracking
from foxtrot.persistence.workspace import initialize_workspace

from .mlflow_client import MlflowTrackingClient, configure_environment
from .uploader import record_destination_failure, sync_job_tracking


BUSY = 3  # exit code: another upload is already running for this workspace


# Describe an error briefly, with any saved credential blanked out.
def _safe_message(error: Exception, store: SettingsStore) -> str:
    message = f"{type(error).__name__}: {error}"[:300]
    for secret in store.secret_values():
        message = message.replace(secret, REDACTED)
    return message


# A saved destination's settings, or an error if it is not configured.
def _destination(store: SettingsStore, destination_id: str) -> dict:
    destination = next((d for d in store.destinations() if d["destination_id"] == destination_id), None)
    if destination is None:
        raise LookupError(f"Tracking destination '{destination_id}' is not configured on the Setup page")
    return destination


# The experiment runs go into: named after the member, or the destination's fallback name if none is set.
def experiment_name(destination: dict, member: str | None) -> str:
    return member or destination["experiment_name"]


# The experiment of one job: the member recorded when it finished; a study trial joins its study's experiment.
def job_experiment(workspace: Path, checkout: Path, job_id: str, destination: dict) -> str:
    from foxtrot.studies.tracking import study_experiment
    entries = list_tracking(workspace, checkout, job_id)
    tags = entries[0].operation.payload.get("tags", {}) if entries else {}
    study_id = tags.get("foxtrot.study_id")
    recorded = study_experiment(workspace, checkout, study_id, destination["destination_id"]) if study_id else None
    if recorded is not None:  # the study's parent run already exists: stay with it
        return recorded or destination["experiment_name"]
    return experiment_name(destination, tags.get("foxtrot.member"))


# Build an MLflow client for one destination and experiment, loading only that destination's credentials.
def _client_for(store: SettingsStore, destination_id: str, experiment: str | None = None, *,
                quick_check: bool = False) -> MlflowTrackingClient:
    destination = _destination(store, destination_id)
    credentials = store.credentials(destination_id)
    configure_environment(credentials.get("username"), credentials.get("token"), quick_check=quick_check)
    return MlflowTrackingClient(destination["tracking_uri"], experiment or destination["experiment_name"],
                                destination["browser_url"])


# Try to reach a destination and open its experiment; reports success or a safe error message.
def check_destination(workspace: Path, checkout: Path, destination_id: str) -> dict:
    store = SettingsStore(workspace, checkout)
    try:
        member = read_profile(workspace, checkout)["member"]
        experiment = experiment_name(_destination(store, destination_id), member)
        client = _client_for(store, destination_id, experiment, quick_check=True)
    except Exception as error:
        return {"ok": False, "message": _safe_message(error, store)}
    return {"ok": True, "message": f"Connected; experiment “{experiment}” (ID {client.experiment_id})"}


# Upload every job with pending steps (or just one job), one destination at a time.
def sync_pending(workspace: Path, checkout: Path, *, job_id: str | None = None) -> list[dict]:
    store = SettingsStore(workspace, checkout)
    pending = jobs_with_pending_tracking(workspace, checkout)
    if job_id is not None:
        pending = [job for job in pending if job == job_id]
    by_destination: dict[str, list[str]] = {}
    for job in pending:
        resolved = JobSpec.model_validate_json(load_job_configurations(workspace, checkout, job).resolved_json)
        by_destination.setdefault(resolved.tracking_destination_id, []).append(job)
    from foxtrot.studies.tracking import finalize_studies, trial_run_tags, trial_settled

    results, clients = [], {}

    # One client per destination and experiment, created once (credentials are configured for each).
    def client_for(destination_id: str, experiment: str | None = None):
        destination = _destination(store, destination_id)
        if experiment is None:
            experiment = experiment_name(destination, read_profile(workspace, checkout)["member"])
        experiment = experiment or destination["experiment_name"]
        if (destination_id, experiment) not in clients:
            clients[destination_id, experiment] = _client_for(store, destination_id, experiment)
        return clients[destination_id, experiment]

    for destination_id, jobs in by_destination.items():
        for job in jobs:
            try:
                client = client_for(destination_id, job_experiment(workspace, checkout, job,
                                                                   _destination(store, destination_id)))
            except Exception as error:  # not configured or unreachable: upload stays queued, with the reason shown
                record = record_destination_failure(workspace, checkout, job, error)
                results.append({"job_id": job, "tracking_state": record.tracking_state,
                                "skipped": _safe_message(error, store),
                                "error": record.tracking_error.message if record.tracking_error else None})
                continue
            study = JobSpec.model_validate_json(load_job_configurations(workspace, checkout, job).resolved_json).study
            if study and not trial_settled(workspace, checkout, study.study_id, study.trial_number):
                results.append({"job_id": job, "tracking_state": "pending",
                                "deferred": "waiting for the study to record this trial's outcome"})
                continue
            record = sync_job_tracking(workspace, checkout, job, client, extra_run_tags=lambda tags, client=client: trial_run_tags(
                workspace, checkout, client, destination_id, tags))
            results.append({"job_id": job, "tracking_state": record.tracking_state,
                            "error": record.tracking_error.message if record.tracking_error else None})
    for outcome in finalize_studies(workspace, checkout, client_for):  # V5: close ended studies' parent runs
        results.append({"study_id": outcome["study_id"],
                        "tracking_state": "synced" if outcome["finalized"] else "pending",
                        "error": outcome.get("error")})
    return results


# Add the comparison parameters introduced later (run.name, run.short_id; model_id and dataset_id on
# study parents) to runs uploaded before them. Only missing keys are added; no existing value changes.
def backfill_parameters(workspace: Path, checkout: Path) -> list[dict]:
    from foxtrot.persistence.jobs import list_job_records
    from foxtrot.persistence.tracking import get_run_association
    from foxtrot.studies.store import list_study_ids, read_json, read_study, study_folder

    store = SettingsStore(workspace, checkout)
    targets = []  # (destination, run ID, label, parameters to add)
    for record in list_job_records(workspace, checkout, limit=500):
        association = get_run_association(workspace, checkout, record.job_id)
        if association is not None:
            targets.append((association.destination_id, association.run_id, record.job_id[:8],
                            {"run.short_id": record.job_id[:8]}))
    for study_id in list_study_ids(workspace, checkout):
        spec, _, _ = read_study(workspace, checkout, study_id)
        path = study_folder(workspace, checkout, study_id) / "mlflow.json"
        for destination_id, entry in (read_json(path) if path.is_file() else {}).items():
            targets.append((destination_id, entry["run_id"], f"study {spec.name}", {
                "model_id": ",".join(spec.model_ids), "run.short_id": study_id[:8],
                "dataset_id": spec.dataset_id or "generated per trial (recipe study)"}))
    clients, results = {}, []
    for destination_id, run_id, label, wanted in targets:
        try:
            if destination_id not in clients:
                clients[destination_id] = _client_for(store, destination_id)
            client = clients[destination_id]
            run = client.client.get_run(run_id)
            missing = {key: value for key, value in {"run.name": run.info.run_name, **wanted}.items()
                       if key not in run.data.params}
            if missing:
                client.log_params(run_id, missing)
            results.append({"run": label, "added": sorted(missing)})
        except Exception as error:  # e.g. run deleted in MLflow or destination unreachable
            results.append({"run": label, "error": _safe_message(error, store)})
    return results


# Command line entry point: "run" uploads pending jobs, "check" tests one destination,
# "backfill" adds newer comparison parameters to runs uploaded earlier.
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Upload all pending jobs")
    run.add_argument("--job", help="Upload only this job")
    check = commands.add_parser("check", help="Test a tracking destination")
    check.add_argument("destination_id")
    commands.add_parser("backfill", help="Add newer comparison parameters to earlier runs")
    arguments = parser.parse_args(argv)
    if arguments.command == "check":
        result = check_destination(arguments.workspace, arguments.source_checkout, arguments.destination_id)
        print(json.dumps(result))
        return 0 if result["ok"] else 1
    # Only one uploader at a time per workspace, so no job is ever sent twice at once.
    root = initialize_workspace(arguments.workspace, arguments.source_checkout)
    try:
        with worker_lock(root / "settings" / "uploader" / "uploader.lock"):
            if arguments.command == "backfill":
                results = backfill_parameters(root, arguments.source_checkout)
            else:
                results = sync_pending(root, arguments.source_checkout, job_id=arguments.job)
    except WorkerAlreadyRunning:
        print(json.dumps({"busy": True}))
        return BUSY
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
