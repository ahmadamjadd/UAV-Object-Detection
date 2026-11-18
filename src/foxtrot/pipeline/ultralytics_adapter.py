"""Real YOLO training through Ultralytics, behind the H6 Trainer contract.

Runs inside the job's child process. Ultralytics is imported only when training
starts, after it is given a private settings folder with its own MLflow/DVC/
cloud logging and usage analytics switched off: Foxtrot does the logging, and
nothing is sent anywhere during training.
"""

import math
import os
from pathlib import Path
import sys
import traceback
from typing import Literal

from foxtrot.config.checkpoint import CheckpointReference
from foxtrot.config.evaluation import EvaluationResult
from foxtrot.persistence.files import fingerprint, job_file
from foxtrot.persistence.models import WeightsError, verify_weights

from .contracts import (
    EvaluationInput, PipelineCancelled, PipelineFailure, ProgressReporter, TrainingInput, TrainingResult,
)

# Ultralytics features that would log elsewhere or contact the internet.
DISABLED_INTEGRATIONS = {
    "sync": False, "mlflow": False, "dvc": False, "clearml": False,
    "comet": False, "raytune": False, "tensorboard": False, "wandb": False,
}


# Give Ultralytics a private, offline settings folder for this job, then import it.
# Calling it again in the same process (e.g. evaluation after training) reuses that setup.
def _load_ultralytics(output: Path):
    config = output / ".ultralytics"
    if "ultralytics" in sys.modules:
        if os.environ.get("YOLO_CONFIG_DIR") != str(config):
            raise RuntimeError("Ultralytics was imported before its settings could be isolated")
        from ultralytics import YOLO
        return YOLO
    config.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(config / "matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from ultralytics import settings
    settings.update(DISABLED_INTEGRATIONS)
    from ultralytics import YOLO
    return YOLO


# Keep only finite numbers and simplify names: "metrics/mAP50(B)" becomes "metrics/mAP50".
def _clean_metrics(values: dict) -> dict[str, float]:
    cleaned = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            cleaned[str(key).replace("(B)", "")] = round(number, 6)
    return cleaned


# Collect one epoch's losses, validation scores and learning rates from the Ultralytics trainer.
def _epoch_metrics(trainer) -> dict[str, float]:
    values = dict(trainer.metrics or {})
    try:
        values.update(trainer.label_loss_items(trainer.tloss))
    except Exception:  # losses are unavailable outside the training loop
        pass
    values.update(getattr(trainer, "lr", None) or {})
    return _clean_metrics(values)


# Trains a real YOLO model with the job's own data.yaml and output folder.
class UltralyticsTrainer:
    # initialization: "pretrained" uses <workspace>/models/<model>.pt; "scratch" starts from random weights.
    def __init__(self, *, initialization: Literal["pretrained", "scratch"] = "pretrained", workers: int = 2):
        self.initialization = initialization
        self.workers = workers

    # Find the starting weights and their fingerprint: the installed, verified pretrained
    # weights, or the bare architecture (no fingerprint) when starting from scratch.
    def _model_source(self, request: TrainingInput) -> tuple[str, str | None]:
        model_id = request.resolved.model_id
        if self.initialization == "scratch":
            return f"{model_id}.yaml", None
        models = Path(request.job_directory).parents[1] / "models"
        try:
            digest = verify_weights(models, model_id)
        except WeightsError as error:
            code = "weights_changed" if "changed" in str(error) else "weights_missing"
            raise PipelineFailure(code, f"{error}. Install {model_id}.pt from the Setup page.") from error
        return str(models / f"{model_id}.pt"), digest

    # Run training, report each epoch, stop on cancel, and return the exact best checkpoint.
    def train(self, request: TrainingInput, reporter: ProgressReporter) -> TrainingResult:
        source, weights_sha256 = self._model_source(request)  # before importing anything heavy
        output = Path(request.output_directory)
        YOLO = _load_ultralytics(output)
        settings = request.resolved.training
        model = YOLO(source)
        reporter.log({"action": "model_loaded", "initialization": self.initialization,
                      "source": Path(source).name, "weights_sha256": weights_sha256})
        state = {"epochs_done": 0, "epoch_pending": False, "metrics": {}}

        # After every batch: stop at once if cancellation was requested.
        def on_batch_end(trainer):
            if reporter.wait_cancelled(0):
                raise PipelineCancelled()

        # Fires only inside the real training loop: marks that an epoch just finished.
        def on_train_epoch_end(trainer):
            state["epoch_pending"] = True

        # Fires after each epoch's validation, and once more for the final best.pt check.
        # Metrics are always kept (the last call holds best.pt's scores); only real epochs are reported.
        def on_fit_epoch_end(trainer):
            state["metrics"] = _epoch_metrics(trainer)
            if state["epoch_pending"]:
                state["epoch_pending"] = False
                state["epochs_done"] += 1
                reporter.progress({"epoch": state["epochs_done"], "epochs": trainer.epochs, **state["metrics"]})

        model.add_callback("on_train_batch_end", on_batch_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        try:
            model.train(
                data=request.data_yaml, epochs=settings.epochs, batch=settings.batch_size,
                imgsz=settings.image_size, seed=settings.seed, deterministic=True, device=settings.device,
                workers=self.workers, project=str(output), name="train", exist_ok=False,
                plots=False, verbose=False,
            )
        except PipelineCancelled:
            raise
        except Exception as error:
            traceback.print_exc()  # full details go to the job's stderr log
            raise PipelineFailure("training_failed",
                                  f"YOLO training failed ({type(error).__name__}); see the job's error log.") from error
        best = output / "train" / "weights" / "best.pt"
        if not best.is_file():
            raise PipelineFailure("checkpoint_missing", "Training finished without saving best.pt.")
        digest, _ = fingerprint(best)
        return TrainingResult(
            checkpoint=CheckpointReference(relative_path=best.relative_to(request.job_directory).as_posix(),
                                           sha256=digest),
            epochs_completed=state["epochs_done"], metrics=state["metrics"], fixture=False,
        )


# Scores the exact checkpoint chosen by training on the dataset's validation images.
class UltralyticsEvaluator:
    # Verify the checkpoint, run YOLO validation once, and return the H2 EvaluationResult.
    def evaluate(self, request: EvaluationInput, reporter: ProgressReporter) -> EvaluationResult:
        if reporter.wait_cancelled(0):
            raise PipelineCancelled()
        job = Path(request.job_directory)
        try:
            checkpoint = job_file(job, request.checkpoint.relative_path)
            data_yaml = job_file(job, "output/data.yaml")
        except ValueError as error:
            raise PipelineFailure("evaluation_input_missing", str(error)) from error
        if fingerprint(checkpoint)[0] != request.checkpoint.sha256:
            raise PipelineFailure("checkpoint_mismatch", "The checkpoint changed after training; it was not scored.")
        YOLO = _load_ultralytics(job / "output")
        model = YOLO(str(checkpoint))  # remembers its training image size, so scoring matches training
        try:
            results = model.val(data=str(data_yaml), split="val", device="cpu", batch=16, plots=False,
                                verbose=False, project=str(job / "output"), name="evaluation", exist_ok=False)
        except Exception as error:
            traceback.print_exc()
            raise PipelineFailure("evaluation_failed",
                                  f"YOLO evaluation failed ({type(error).__name__}); see the job's error log.") from error
        box = results.box
        names = request.dataset.metadata.class_names
        per_class = {names[i]: round(float(value), 6) for i, value in enumerate(box.maps[:len(names)])}
        reporter.log({"action": "evaluation_complete", "checkpoint": request.checkpoint.relative_path,
                      "per_class_map50_95": per_class})
        return EvaluationResult(
            dataset_id=request.dataset.metadata.dataset_id,
            dataset_sha256=request.dataset.metadata.content_sha256,
            checkpoint=request.checkpoint,
            precision=_score(box.mp), recall=_score(box.mr), map50=_score(box.map50), map50_95=_score(box.map),
        )


# Turn a library score into a plain number between 0 and 1 (missing or invalid becomes 0).
def _score(value) -> float:
    number = float(value)
    return min(max(number, 0.0), 1.0) if math.isfinite(number) else 0.0
