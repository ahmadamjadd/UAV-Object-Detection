"""Fixture trainer/evaluator that honour the real adapter contract.

They read only their explicit inputs, write only inside the assigned output
folder, report progress, honour cancellation, and return H2 checkpoint and
evaluation identities. The checkpoint bytes are NOT a model and the scores are
fixed placeholders; every result is marked ``fixture``.
"""

import hashlib
import json
from pathlib import Path

from foxtrot.config.checkpoint import CheckpointReference
from foxtrot.config.evaluation import EvaluationResult
from foxtrot.persistence.files import fingerprint, job_file, sync_directory, write_new_text

from .contracts import (
    EvaluationInput, PipelineCancelled, PipelineFailure, ProgressReporter,
    TrainingInput, TrainingResult,
)


class FixtureTrainer:
    def __init__(self, *, epoch_seconds: float = 0.05, fail_at_epoch: int | None = None):
        self.epoch_seconds = epoch_seconds
        self.fail_at_epoch = fail_at_epoch

    def train(self, request: TrainingInput, reporter: ProgressReporter) -> TrainingResult:
        epochs = request.resolved.training.epochs
        if not Path(request.data_yaml).is_file():
            raise PipelineFailure("dataset_config_missing", "The job's dataset configuration was not written.")
        for epoch in range(1, epochs + 1):
            if reporter.wait_cancelled(self.epoch_seconds):
                raise PipelineCancelled()
            reporter.progress({"epoch": epoch, "epochs": epochs, "train/loss": round(1.0 / epoch, 6)})
            if self.fail_at_epoch == epoch:
                raise PipelineFailure("fixture_training_failed", f"Fixture training failed at epoch {epoch}.")
        weights = Path(request.output_directory) / "weights"
        weights.mkdir(exist_ok=False)
        # Deterministic placeholder bytes that name exactly what produced them.
        content = json.dumps({
            "fixture": True, "job_id": request.job_id,
            "resolved_sha256": request.resolved_sha256,
            "dataset_sha256": request.dataset.metadata.content_sha256,
            "model_id": request.resolved.model_id,
        }, sort_keys=True)
        path = weights / "best.pt"
        write_new_text(path, content)
        sync_directory(weights)
        relative = path.relative_to(request.job_directory).as_posix()
        return TrainingResult(
            checkpoint=CheckpointReference(
                relative_path=relative, sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            ),
            epochs_completed=epochs,
            metrics={"train/loss": round(1.0 / epochs, 6)},
            fixture=True,
        )


class FixtureEvaluator:
    def evaluate(self, request: EvaluationInput, reporter: ProgressReporter) -> EvaluationResult:
        if reporter.wait_cancelled(0):
            raise PipelineCancelled()
        path = job_file(Path(request.job_directory), request.checkpoint.relative_path)
        digest, _ = fingerprint(path)
        if digest != request.checkpoint.sha256:
            raise PipelineFailure("checkpoint_mismatch", "The checkpoint changed before evaluation.")
        reporter.log({"action": "fixture_evaluation", "checkpoint": request.checkpoint.relative_path})
        return EvaluationResult(
            dataset_id=request.dataset.metadata.dataset_id,
            dataset_sha256=request.dataset.metadata.content_sha256,
            checkpoint=request.checkpoint,
            precision=0.5, recall=0.5, map50=0.5, map50_95=0.25,
        )
