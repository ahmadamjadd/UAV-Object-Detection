"""Build a job's upload plan from saved local results only."""

import math
import re

from foxtrot.config.job import JobSpec
from foxtrot.config.result import JobResult
from foxtrot.config.tracking import TrackingOperation
from foxtrot.pipeline.contracts import DatasetInput, PipelineResult


JOB_TAG = "foxtrot.job_id"
STATUS = {"finished": "FINISHED", "failed": "FAILED", "interrupted": "FAILED", "cancelled": "KILLED"}

# Raw stdout/stderr stay local until H7 adds log redaction.
UPLOADED_ARTIFACTS = (
    ("checkpoint", "weights"), ("job-result", "results"),
    ("pipeline-result", "results"), ("data-yaml", "dataset"), ("summary", "results"),
    ("dataset-provenance", "dataset"),
)


# MLflow parameters describing how a generated dataset was made and split (V2).
def generation_params(provenance: dict | None) -> dict[str, str]:
    if not provenance:
        return {}
    recipe, split = provenance.get("recipe", {}), provenance.get("split", {})
    counts, backgrounds = provenance.get("counts", {}), provenance.get("backgrounds", {})
    values = {
        "dataset.generated": True,
        "dataset.generation.identity_sha256": provenance.get("identity_sha256"),
        "dataset.generation.recipe_sha256": provenance.get("recipe_sha256"),
        "dataset.generation.generator_sha256": provenance.get("generator", {}).get("sha256"),
        "dataset.generation.seed": provenance.get("generation_seed"),
        "dataset.generation.images_per_class": recipe.get("images_per_class"),
        "dataset.generation.canvas_size": recipe.get("canvas_size"),
        "dataset.generation.background_count": backgrounds.get("count"),
        "dataset.split.train_ratio": split.get("train_ratio"),
        "dataset.split.seed": split.get("seed"),
        "dataset.split.group_by_background": split.get("group_by_background"),
        "dataset.split.train_images": counts.get("train"),
        "dataset.split.validation_images": counts.get("val"),
    }
    return {key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in values.items() if value is not None}


# Metric names MLflow accepts (letters, digits, _ . / - and spaces).
METRIC_KEY = re.compile(r"[A-Za-z0-9_./ -]{1,250}")


# Turn saved per-epoch progress, e.g. {"epoch": 3, "metrics/mAP50": 0.7}, into metric points
# at step = epoch, so MLflow can draw training curves. Non-numbers and odd names are skipped.
def epoch_metric_points(progress) -> list[dict]:
    points = []
    for payload in progress:
        epoch = payload.get("epoch")
        if type(epoch) is not int or epoch < 1:
            continue
        for key, value in payload.items():
            if key in ("epoch", "epochs") or not METRIC_KEY.fullmatch(str(key)):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            points.append({"key": key, "value": float(value), "step": epoch})
    return points


def _flatten(prefix: str, value, into: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, item, into)
    elif value is not None:  # None = "runtime chooses" (e.g. lr0 with optimizer auto): nothing to log
        into[prefix] = str(value)


def build_tracking_plan(
    *, result: JobResult, resolved: JobSpec, resolved_sha256: str, adapter: str, fixture: bool,
    dataset: DatasetInput, pipeline: PipelineResult | None, artifact_names: set[str],
    epoch_progress=(), dataset_provenance: dict | None = None,
) -> list[TrackingOperation]:
    job_id = result.job_id
    params: dict[str, str] = {}
    _flatten("", resolved.model_dump(mode="json"), params)
    params.update({
        "foxtrot.resolved_sha256": resolved_sha256,
        "dataset.content_sha256": dataset.metadata.content_sha256,
        "dataset.class_names": ",".join(dataset.metadata.class_names),
    })
    params.update(generation_params(dataset_provenance))
    # V3: the model's actual starting weights, traceable to the reported results.
    if pipeline is not None and pipeline.training.initialization is not None:
        params["model.initialization"] = pipeline.training.initialization
        if pipeline.training.start_weights_sha256:
            params["model.start_weights_sha256"] = pipeline.training.start_weights_sha256
    operations = [
        TrackingOperation(operation="create_run", payload={
            "run_name": f"foxtrot-{job_id[:8]}",
            "tags": {JOB_TAG: job_id, "foxtrot.adapter": adapter,
                     "foxtrot.fixture": str(fixture).lower(), "foxtrot.outcome": result.outcome},
        }),
        TrackingOperation(operation="log_params", payload={"params": params}),
    ]
    # Every completed epoch's numbers, for any outcome: a failed run still shows how far it got.
    metrics = epoch_metric_points(epoch_progress)
    if pipeline is not None and result.outcome == "finished":
        step = pipeline.training.epochs_completed
        if not metrics:  # no per-epoch history saved: fall back to training's final numbers
            metrics = [{"key": k, "value": v, "step": step} for k, v in pipeline.training.metrics.items()]
        evaluation = pipeline.evaluation  # the official score of the exact best.pt, logged once
        metrics += [{"key": key, "value": value, "step": step} for key, value in (
            ("eval/precision", evaluation.precision), ("eval/recall", evaluation.recall),
            ("eval/mAP50", evaluation.map50), ("eval/mAP50-95", evaluation.map50_95),
        )]
    if metrics:
        operations.append(TrackingOperation(operation="log_metrics", payload={"metrics": metrics}))
    for name, artifact_path in UPLOADED_ARTIFACTS:
        if name in artifact_names:
            operations.append(TrackingOperation(operation="log_artifact", payload={
                "name": name, "artifact_path": artifact_path,
            }))
    operations.append(TrackingOperation(operation="set_terminated", payload={"status": STATUS[result.outcome]}))
    return operations
