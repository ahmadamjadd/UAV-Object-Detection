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


# A repeatable fake score in [0.1, 0.9] that depends only on the model and training settings (V5 tests).
def config_score(request: TrainingInput) -> float:
    text = json.dumps({"model": request.resolved.model_id,
                       "training": request.resolved.training.model_dump(mode="json")}, sort_keys=True)
    return round(0.1 + 0.8 * int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF, 6)


# Fake trainer for tests: writes a small checkpoint instead of training.
class FixtureTrainer:
    # Delay per fake epoch, optional planned failure, optional scores from settings.
    def __init__(self, *, epoch_seconds: float = 0.05, fail_at_epoch: int | None = None,
                 config_scores: bool = False):
        self.config_scores = config_scores
        self.epoch_seconds = epoch_seconds
        self.fail_at_epoch = fail_at_epoch

    # Pretend to train: report each epoch, then write the checkpoint.
    def train(self, request: TrainingInput, reporter: ProgressReporter) -> TrainingResult:
        epochs = request.resolved.training.epochs
        if not Path(request.data_yaml).is_file():
            raise PipelineFailure("dataset_config_missing", "The job's dataset configuration was not written.")
        for epoch in range(1, epochs + 1):
            if reporter.wait_cancelled(self.epoch_seconds):
                raise PipelineCancelled()
            progress = {"epoch": epoch, "epochs": epochs, "train/loss": round(1.0 / epoch, 6)}
            if self.config_scores:  # rises to the settings' score by the last epoch
                progress["metrics/mAP50-95"] = round(config_score(request) * epoch / epochs, 6)
            reporter.progress(progress)
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
            **({"score": config_score(request)} if self.config_scores else {}),
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


# Fake evaluator for tests: returns fixed scores for the checkpoint.
class FixtureEvaluator:
    # Check the checkpoint exists and return fixed scores.
    def evaluate(self, request: EvaluationInput, reporter: ProgressReporter) -> EvaluationResult:
        if reporter.wait_cancelled(0):
            raise PipelineCancelled()
        path = job_file(Path(request.job_directory), request.checkpoint.relative_path)
        digest, _ = fingerprint(path)
        if digest != request.checkpoint.sha256:
            raise PipelineFailure("checkpoint_mismatch", "The checkpoint changed before evaluation.")
        reporter.log({"action": "fixture_evaluation", "checkpoint": request.checkpoint.relative_path})
        try:  # V5 tests: a checkpoint made with config_scores carries its settings-based score
            score = json.loads(path.read_text(encoding="utf-8")).get("score")
        except (ValueError, OSError, AttributeError):
            score = None
        return EvaluationResult(
            dataset_id=request.dataset.metadata.dataset_id,
            dataset_sha256=request.dataset.metadata.content_sha256,
            checkpoint=request.checkpoint,
            precision=0.5, recall=0.5, map50=0.5, map50_95=0.25 if score is None else score,
        )
