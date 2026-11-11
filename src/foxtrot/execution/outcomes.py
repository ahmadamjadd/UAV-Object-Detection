"""Parent-side acceptance of adapter results; the child is never trusted blindly."""

from pathlib import Path, PurePosixPath

from foxtrot.config.result import JobResult
from foxtrot.persistence.artifacts import record_job_artifact
from foxtrot.persistence.files import job_file, sync_directory, write_new_text
from foxtrot.persistence.jobs import JobClaim, JobRecord
from foxtrot.persistence.tracking import enqueue_tracking
from foxtrot.pipeline.contracts import PipelineResult
from foxtrot.pipeline.registry import get_adapters
from foxtrot.tracking.plan import build_tracking_plan

from .protocol import PipelineExecutionRequest


def accept_pipeline_result(
    workspace: Path, checkout: Path, claim: JobClaim, request: PipelineExecutionRequest,
) -> PipelineResult:
    """Validate the report against what this parent assigned, then register files.

    Raising here makes the supervisor record a failure instead of finished.
    """
    path = job_file(claim.directory, "output/pipeline-result.json")
    result = PipelineResult.model_validate_json(path.read_bytes())
    dataset = request.dataset.metadata
    if result.job_id != claim.job_id or result.adapter != request.pipeline.adapter:
        raise ValueError("Pipeline result belongs to a different job or adapter")
    if (result.evaluation.dataset_id, result.evaluation.dataset_sha256) != (dataset.dataset_id, dataset.content_sha256):
        raise ValueError("Pipeline evaluation used a different dataset")
    checkpoint = result.training.checkpoint
    if PurePosixPath(checkpoint.relative_path).parts[0] != "output":
        raise ValueError("Checkpoints must be inside the job's output folder")
    # Registration fingerprints the actual file; it must equal the reported hash.
    reference = record_job_artifact(workspace, checkout, claim.job_id, "checkpoint",
                                    checkpoint.relative_path, kind="checkpoint")
    if reference.sha256 != checkpoint.sha256:
        raise ValueError("Checkpoint hash does not match the pipeline result")
    record_job_artifact(workspace, checkout, claim.job_id, "pipeline-result", "output/pipeline-result.json")
    record_job_artifact(workspace, checkout, claim.job_id, "data-yaml", "output/data.yaml")
    return result


def record_pipeline_outcome(
    workspace: Path, checkout: Path, claim: JobClaim, request: PipelineExecutionRequest,
    record: JobRecord, pipeline: PipelineResult | None,
) -> JobResult:
    """Save the H2 JobResult and the job's pending tracking plan.

    Both happen after the execution outcome commits. A crash in between leaves
    a final job without a queued upload; V7 recovery can rebuild it from these
    saved files. It never changes or reruns the outcome.
    """
    finished = record.execution_state == "finished"
    result = JobResult(
        job_id=claim.job_id, outcome=record.execution_state,
        checkpoint=pipeline.training.checkpoint if finished else None,
        evaluation=pipeline.evaluation if finished else None,
        error_message=(record.error_details.message
                       if record.execution_state in {"failed", "interrupted"} else None),
        tracking_destination_id=request.resolved.tracking_destination_id,
    )
    write_new_text(claim.directory / "result.json", result.model_dump_json(indent=2))
    sync_directory(claim.directory)
    record_job_artifact(workspace, checkout, claim.job_id, "job-result", "result.json")
    names = {"checkpoint", "pipeline-result", "data-yaml"} if finished else set()
    plan = build_tracking_plan(
        result=result, resolved=request.resolved, resolved_sha256=request.resolved_sha256,
        adapter=request.pipeline.adapter, fixture=get_adapters(request.pipeline).fixture,
        dataset=request.dataset, pipeline=pipeline,
        artifact_names=names | {"job-result", "summary"},
    )
    enqueue_tracking(workspace, checkout, claim.job_id, plan)
    return result
