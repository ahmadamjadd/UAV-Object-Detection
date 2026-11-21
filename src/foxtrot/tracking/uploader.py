"""Replay a job's pending uploads in order, resuming after the last success.

Run one uploader at a time. Tracking services do not promise exactly-once
delivery: run creation is reconciled by searching for the job tag, and a crash
between a successful call and its acknowledgement may repeat that one call.
"""

from pathlib import Path

from foxtrot.config.job import JobSpec
from foxtrot.config.lifecycle import ErrorDetails, TrackingState
from foxtrot.config.tracking import MetricPoint
from foxtrot.persistence.artifacts import get_job_artifact
from foxtrot.persistence.files import job_path
from foxtrot.persistence.jobs import JobRecord, get_job_record, load_job_configurations
from foxtrot.persistence.lifecycle import transition_tracking
from foxtrot.persistence.tracking import (
    acknowledge_tracking, get_run_association, list_tracking,
    record_run_association, record_tracking_attempt,
)
from foxtrot.persistence.workspace import initialize_workspace

from .contracts import TrackingClient


# V7: a safe, specific reason for a failed upload (exception text may contain URLs or tokens,
# so only its type and a few status words are inspected; nothing from it is shown).
def classify_tracking_error(error: Exception) -> ErrorDetails:
    if isinstance(error, LookupError):  # raised by sync for a destination missing from Setup
        return ErrorDetails(code="tracking_not_configured", message="This job's tracking destination is not "
                            "configured on the Setup page. Local results are kept.")
    text = f"{type(error).__name__} {error}".lower()
    status = getattr(getattr(error, "response", None), "status_code", None) or getattr(error, "status_code", None)
    if status in (401, 403) or any(word in text for word in ("unauthorized", "forbidden", "permission denied",
                                                              "invalid credentials", "authentication")):
        return ErrorDetails(code="tracking_credentials", message="Upload refused: check the username and token "
                            "for this destination on the Setup page. Local results are kept.")
    if status == 429 or any(word in text for word in ("quota", "rate limit", "too many requests",
                                                       "storage limit", "insufficient storage", "507")):
        return ErrorDetails(code="tracking_quota", message="Upload refused by the tracking service's limits "
                            "(quota or rate). Local results are kept; it is retried automatically.")
    if isinstance(error, (ConnectionError, TimeoutError, OSError)) or any(word in text for word in (
            "connection", "timed out", "timeout", "unreachable", "name resolution", "max retries")):
        return ErrorDetails(code="tracking_unreachable", message="The tracking service could not be reached. "
                            "Local results are kept; upload resumes automatically when it is back.")
    return ErrorDetails(code="tracking_upload_failed", message=f"Tracking upload failed ({type(error).__name__}); "
                        "local results are kept and can be retried.")


# V7: the destination could not even be opened (unreachable, refused, not configured):
# show the reason on the waiting job, keeping it queued for the next automatic try.
def record_destination_failure(workspace_root, source_checkout, job_id: str, error: Exception) -> JobRecord:
    root = initialize_workspace(workspace_root, source_checkout)
    state = TrackingState(get_job_record(root, source_checkout, job_id).tracking_state)
    if state == TrackingState.FAILED:
        return transition_tracking(root, source_checkout, job_id, TrackingState.RETRYING,
                                   error=classify_tracking_error(error))
    if state in (TrackingState.PENDING, TrackingState.RETRYING):
        transition_tracking(root, source_checkout, job_id, TrackingState.SYNCING)
        return transition_tracking(root, source_checkout, job_id, TrackingState.RETRYING,
                                   error=classify_tracking_error(error))
    return get_job_record(root, source_checkout, job_id)


def sync_job_tracking(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str, client: TrackingClient,
    *, extra_run_tags=None,
) -> JobRecord:
    """``extra_run_tags(tags) -> dict`` may add tags when the run is created (V5: study parent link)."""
    root = initialize_workspace(workspace_root, source_checkout)
    record = get_job_record(root, source_checkout, job_id)
    pending = list_tracking(root, source_checkout, job_id, pending_only=True)
    if record.tracking_state == TrackingState.SYNCED or not list_tracking(root, source_checkout, job_id):
        return record
    state = TrackingState(record.tracking_state)
    if state == TrackingState.FAILED:
        transition_tracking(root, source_checkout, job_id, TrackingState.RETRYING)
    transition_tracking(root, source_checkout, job_id, TrackingState.SYNCING)
    destination = JobSpec.model_validate_json(
        load_job_configurations(root, source_checkout, job_id).resolved_json,
    ).tracking_destination_id
    association = get_run_association(root, source_checkout, job_id)
    try:
        for entry in pending:
            operation, payload = entry.operation.operation, entry.operation.payload
            record_tracking_attempt(root, source_checkout, job_id, entry.position)
            if operation == "create_run":
                if association is None:
                    tags = dict(payload["tags"])
                    if extra_run_tags is not None:
                        tags.update(extra_run_tags(tags))
                    handle = client.find_run(job_id) or client.create_run(job_id, payload["run_name"], tags)
                    association = record_run_association(
                        root, source_checkout, job_id, destination_id=destination,
                        run_id=handle.run_id, run_url=handle.url,
                    )
            elif association is None:
                raise RuntimeError("Tracking run is not associated")
            elif operation == "log_params":
                client.log_params(association.run_id, dict(payload["params"]))
            elif operation == "log_metrics":
                client.log_metrics(association.run_id, [MetricPoint.model_validate(p) for p in payload["metrics"]])
            elif operation == "log_artifact":
                # Recheck the registered hash so a changed file is never uploaded.
                reference = get_job_artifact(root, source_checkout, job_id, payload["name"])
                path = job_path(root, job_id) / reference.relative_path
                client.log_artifact(association.run_id, path, payload["artifact_path"])
            else:
                client.set_terminated(association.run_id, payload["status"])
            acknowledge_tracking(root, source_checkout, job_id, entry.position)
    except Exception as error:
        return transition_tracking(root, source_checkout, job_id, TrackingState.RETRYING,
                                   error=classify_tracking_error(error))
    return transition_tracking(root, source_checkout, job_id, TrackingState.SYNCED)
