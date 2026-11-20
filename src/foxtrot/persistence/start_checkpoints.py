"""Start a new job from the best checkpoint of an earlier job (V3).

Two checks guard this. When the job is submitted, the earlier job must be
finished, trained the same model, and still have its registered checkpoint
unchanged. When training starts, the child re-reads that job's result and
re-checks the checkpoint's SHA-256, so the exact starting weights are known
and recorded.
"""

from pathlib import Path

from foxtrot.config.job import JobSpec
from foxtrot.config.result import JobResult

from .artifacts import get_job_artifact
from .files import fingerprint, job_file, job_path
from .jobs import get_job_record, load_job_configurations


# Raised when the chosen earlier job cannot be used as a starting point.
class StartCheckpointError(ValueError):
    pass


# Submission check: the earlier job finished, used the same model, and its checkpoint is intact.
def check_start_checkpoint(workspace_root: str | Path, source_checkout: str | Path, spec: JobSpec) -> None:
    if spec.initialization != "checkpoint":
        return
    job_id = spec.checkpoint_job_id
    try:
        record = get_job_record(workspace_root, source_checkout, job_id)
    except (KeyError, ValueError) as error:
        raise StartCheckpointError(f"Unknown job {job_id}") from error
    if record.execution_state != "finished":
        raise StartCheckpointError(f"Job {job_id[:8]} has not finished, so it has no final checkpoint")
    earlier = JobSpec.model_validate_json(load_job_configurations(workspace_root, source_checkout, job_id).resolved_json)
    if earlier.model_id != spec.model_id:
        raise StartCheckpointError(f"Job {job_id[:8]} trained {earlier.model_id}; choose that model to continue from it")
    try:
        get_job_artifact(workspace_root, source_checkout, job_id, "checkpoint")
    except (KeyError, ValueError) as error:
        raise StartCheckpointError(f"Job {job_id[:8]}'s checkpoint is missing or changed") from error


# Training-time lookup: the earlier job's best.pt path and its verified SHA-256.
def locate_start_checkpoint(workspace_root: Path, job_id: str, model_id: str) -> tuple[Path, str]:
    try:
        directory = job_path(workspace_root, job_id)
        result = JobResult.model_validate_json(job_file(directory, "result.json").read_bytes())
        earlier = JobSpec.model_validate_json(job_file(directory, "resolved.json").read_bytes())
    except (ValueError, OSError) as error:
        raise StartCheckpointError(f"Job {job_id[:8]} has no readable result") from error
    if result.outcome != "finished" or result.checkpoint is None:
        raise StartCheckpointError(f"Job {job_id[:8]} did not finish with a checkpoint")
    if earlier.model_id != model_id:
        raise StartCheckpointError(f"Job {job_id[:8]} trained {earlier.model_id}, not {model_id}")
    path = job_file(directory, result.checkpoint.relative_path)
    digest = fingerprint(path)[0]
    if digest != result.checkpoint.sha256:
        raise StartCheckpointError(f"Job {job_id[:8]}'s checkpoint changed after it finished")
    return path, digest


# V4 submission rules for the dataset a job uses:
# - training jobs only use "training" datasets (real validation/test images are never trained on);
# - evaluation jobs only use "validation"/"test" datasets whose classes match the checkpoint's.
def check_dataset_use(workspace_root: str | Path, source_checkout: str | Path, spec: JobSpec) -> None:
    from .datasets import list_datasets

    manifests = {m.dataset_id: m for m in list_datasets(workspace_root, source_checkout)}
    dataset = manifests.get(spec.dataset_id)
    if dataset is None:
        return  # not registered (yet): the worker refuses it visibly when the job starts
    if spec.kind == "train":
        if dataset.role != "training":
            raise StartCheckpointError(f"{spec.dataset_id} is a {dataset.role} dataset: it can only score models")
        return
    if dataset.role == "training":
        raise StartCheckpointError(f"{spec.dataset_id} is a training dataset; evaluate on a validation or test dataset")
    earlier = JobSpec.model_validate_json(
        load_job_configurations(workspace_root, source_checkout, spec.checkpoint_job_id).resolved_json)
    trained_on = manifests.get(earlier.dataset_id)
    if trained_on is not None and trained_on.class_names != dataset.class_names:
        raise StartCheckpointError(
            f"Class mismatch: the model knows {', '.join(trained_on.class_names)} but {spec.dataset_id} has "
            f"{', '.join(dataset.class_names)} (same names in the same order are required)")
