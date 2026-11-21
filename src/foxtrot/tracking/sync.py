"""Send pending uploads to MLflow, or test a tracking destination.

Runs as its own short-lived process so MLflow and your credentials are never
loaded into the website server:

    python -m foxtrot.tracking.sync --workspace /abs/ws --source-checkout /abs/MlOps run
    python -m foxtrot.tracking.sync --workspace /abs/ws --source-checkout /abs/MlOps check dagshub

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
from foxtrot.persistence.tracking import jobs_with_pending_tracking
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


# Build an MLflow client for a saved destination, loading only that destination's credentials.
def _client_for(store: SettingsStore, destination_id: str, *, quick_check: bool = False) -> MlflowTrackingClient:
    destination = next((d for d in store.destinations() if d["destination_id"] == destination_id), None)
    if destination is None:
        raise LookupError(f"Tracking destination '{destination_id}' is not configured on the Setup page")
    credentials = store.credentials(destination_id)
    configure_environment(credentials.get("username"), credentials.get("token"), quick_check=quick_check)
    return MlflowTrackingClient(destination["tracking_uri"], destination["experiment_name"],
                                destination["browser_url"])


# Try to reach a destination and open its experiment; reports success or a safe error message.
def check_destination(workspace: Path, checkout: Path, destination_id: str) -> dict:
    store = SettingsStore(workspace, checkout)
    try:
        client = _client_for(store, destination_id, quick_check=True)
    except Exception as error:
        return {"ok": False, "message": _safe_message(error, store)}
    return {"ok": True, "message": f"Connected; experiment ID {client.experiment_id}"}


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

    # One client per destination, created once (credentials are configured for each).
    def client_for(destination_id: str):
        if destination_id not in clients:
            clients[destination_id] = _client_for(store, destination_id)
        return clients[destination_id]

    for destination_id, jobs in by_destination.items():
        try:
            client = client_for(destination_id)
        except Exception as error:  # not configured or unreachable: uploads stay queued, with the reason shown
            reason = _safe_message(error, store)
            for job in jobs:
                record = record_destination_failure(workspace, checkout, job, error)
                results.append({"job_id": job, "tracking_state": record.tracking_state, "skipped": reason,
                                "error": record.tracking_error.message if record.tracking_error else None})
            continue
        for job in jobs:
            study = JobSpec.model_validate_json(load_job_configurations(workspace, checkout, job).resolved_json).study
            if study and not trial_settled(workspace, checkout, study.study_id, study.trial_number):
                results.append({"job_id": job, "tracking_state": "pending",
                                "deferred": "waiting for the study to record this trial's outcome"})
                continue
            record = sync_job_tracking(workspace, checkout, job, client, extra_run_tags=lambda tags: trial_run_tags(
                workspace, checkout, client, destination_id, tags))
            results.append({"job_id": job, "tracking_state": record.tracking_state,
                            "error": record.tracking_error.message if record.tracking_error else None})
    for outcome in finalize_studies(workspace, checkout, client_for):  # V5: close ended studies' parent runs
        results.append({"study_id": outcome["study_id"],
                        "tracking_state": "synced" if outcome["finalized"] else "pending",
                        "error": outcome.get("error")})
    return results


# Command line entry point: "run" uploads pending jobs, "check" tests one destination.
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Upload all pending jobs")
    run.add_argument("--job", help="Upload only this job")
    check = commands.add_parser("check", help="Test a tracking destination")
    check.add_argument("destination_id")
    arguments = parser.parse_args(argv)
    if arguments.command == "check":
        result = check_destination(arguments.workspace, arguments.source_checkout, arguments.destination_id)
        print(json.dumps(result))
        return 0 if result["ok"] else 1
    # Only one uploader at a time per workspace, so no job is ever sent twice at once.
    root = initialize_workspace(arguments.workspace, arguments.source_checkout)
    try:
        with worker_lock(root / "settings" / "uploader" / "uploader.lock"):
            results = sync_pending(root, arguments.source_checkout, job_id=arguments.job)
    except WorkerAlreadyRunning:
        print(json.dumps({"busy": True}))
        return BUSY
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
