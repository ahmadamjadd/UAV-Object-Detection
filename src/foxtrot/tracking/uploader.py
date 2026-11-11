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


def sync_job_tracking(
    workspace_root: str | Path, source_checkout: str | Path, job_id: str, client: TrackingClient,
) -> JobRecord:
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
                    handle = client.find_run(job_id) or client.create_run(
                        job_id, payload["run_name"], dict(payload["tags"]),
                    )
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
        # Exception text can contain URLs or tokens; save only its class name.
        return transition_tracking(root, source_checkout, job_id, TrackingState.RETRYING, error=ErrorDetails(
            code="tracking_upload_failed",
            message=f"Tracking upload failed ({type(error).__name__}); local results are kept and can be retried.",
        ))
    return transition_tracking(root, source_checkout, job_id, TrackingState.SYNCED)
