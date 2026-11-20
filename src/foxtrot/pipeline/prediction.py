"""Prediction previews on unlabeled images (V4).

Runs an exact earlier checkpoint on a few photos and returns the boxes it drew.
Without labels nobody can say which boxes are right, so the result is marked
``measured: false`` and never reported as accuracy. Runs in the runner process.
"""

from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from foxtrot.config.job import JobSpec

from .contracts import PipelineCancelled, PipelineFailure, ProgressReporter
from .dataset_validation import IMAGE_EXTENSIONS


# Which checkpoint to run, on which folder of photos, and how sure a box must be to be shown.
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    images: str = Field(min_length=1, max_length=4096)
    confidence: float = Field(default=0.25, ge=0.01, le=0.99)
    max_images: int = Field(default=24, ge=1, le=100)

    # The photos folder must be an absolute path on this computer.
    @field_validator("images")
    @classmethod
    def absolute(cls, value: str) -> str:
        if "\x00" in value or not value.startswith("/"):
            raise ValueError("Images folder must be an absolute path")
        return value


# Run the checkpoint on up to max_images photos; copies them as N.jpg into output/images.
def predict_preview(workspace_root: Path, request: PredictionRequest, output: Path,
                    reporter: ProgressReporter) -> dict:
    from foxtrot.persistence.files import job_file, job_path
    from foxtrot.persistence.start_checkpoints import StartCheckpointError, locate_start_checkpoint

    from .ultralytics_adapter import _load_ultralytics

    try:
        spec = JobSpec.model_validate_json(
            job_file(job_path(workspace_root, request.checkpoint_job_id), "resolved.json").read_bytes())
        checkpoint, digest = locate_start_checkpoint(workspace_root, request.checkpoint_job_id, spec.model_id)
    except (StartCheckpointError, ValueError) as error:
        raise PipelineFailure("checkpoint_unavailable", str(error)) from error
    folder = Path(request.images)
    if not folder.is_dir():
        raise PipelineFailure("images_missing", f"Images folder not found: {folder}")
    photos = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    photos = photos[:request.max_images]
    if not photos:
        raise PipelineFailure("images_missing", f"No images in {folder}")
    images = output / "images"
    images.mkdir(parents=True)
    model = _load_ultralytics(output)(str(checkpoint))
    found = []
    for index, photo in enumerate(photos):
        if reporter.wait_cancelled():
            raise PipelineCancelled()
        try:
            with Image.open(photo) as opened:
                opened.convert("RGB").save(images / f"{index}.jpg", quality=92)
        except OSError:
            found.append({"image": None, "source": photo.name, "boxes": [], "error": "unreadable"})
            continue
        result = model.predict(str(images / f"{index}.jpg"), conf=request.confidence, device="cpu",
                               verbose=False, save=False)[0]
        boxes = []
        for (x, y, w, h), cls, conf in zip(result.boxes.xywhn.tolist(), result.boxes.cls.tolist(),
                                           result.boxes.conf.tolist()):
            boxes.append({"class_name": result.names[int(cls)], "x": x, "y": y, "width": w, "height": h,
                          "confidence": round(conf, 3)})
        found.append({"image": f"{index}.jpg", "source": photo.name, "boxes": boxes})
        reporter.progress({"phase": "predicting", "done": index + 1, "total": len(photos)})
    return {"measured": False, "checkpoint_job_id": request.checkpoint_job_id, "model_id": spec.model_id,
            "checkpoint_sha256": digest, "confidence": request.confidence, "images": found,
            "note": "Preview only: these images have no labels, so no accuracy is measured."}
