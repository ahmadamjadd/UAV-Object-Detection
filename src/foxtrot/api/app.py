"""The local FastAPI application: browser interface plus job API.

It records and reads state only. Heavy work belongs to the separate worker;
no request ever trains, so the website stays responsive during training.
"""

from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import typing
import urllib.request

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from foxtrot.config.generation import GenerationRecipe, GenerationRequest, SplitSettings
from foxtrot.config.job import JobSpec
from foxtrot.config.models import MODELS
from foxtrot.config.study import OBJECTIVES, RECIPE_RANGES, RECIPE_SCALARS, StudySpec, planned_trials
from foxtrot.config.result import JobResult
from foxtrot.config.training import GROUPS, TrainingSettings
from foxtrot.config.snapshot import resolve_job
from foxtrot.persistence.commands import request_cancel
from foxtrot.persistence.database import DATABASE_SCHEMA_VERSION, initialize_database
from foxtrot.persistence.dataset_import import import_dataset, import_evaluation_dataset
from foxtrot.persistence.dvc_datasets import DvcImportRequest
from foxtrot.pipeline.prediction import PredictionRequest
from foxtrot.persistence.datasets import DatasetUnavailable, list_datasets
from foxtrot.persistence.labels import LabelError, read_labels, set_label
from foxtrot.persistence.events import read_job_events
from foxtrot.persistence.models import WeightsError, install_weights, list_weights
from foxtrot.persistence.files import job_file, job_path
from foxtrot.persistence.start_checkpoints import StartCheckpointError, check_dataset_use, check_start_checkpoint
from foxtrot.persistence.jobs import (
    JobRecord, create_recorded_job, get_job_record, list_job_records, load_job_configurations,
)
from foxtrot.persistence.tracking import get_run_association, list_tracking
from foxtrot.pipeline.contracts import PipelineFailure
from foxtrot.pipeline.coverage import coverage_report, ultralytics_default_keys
from foxtrot.pipeline.foxtrot_generator import generator_directory, generator_files
from foxtrot.pipeline.registry import PipelineSettings

from .generation import GenerationBusy, GenerationRuns
from .gpu import cached_gpus, detect_gpus
from .security import LOOPBACK_HOSTS, require_local_session
from .settings import DestinationUpdate, SettingsStore
from .uploads import UploadScheduler

STATIC_DIRECTORY = Path(__file__).with_name("static")


# What the browser sends to start a job: a repeat-safe key, the job request and an optional name.
class JobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_key: str = Field(min_length=8, max_length=200, pattern=r"^\S+$")
    request: dict[str, typing.Any]
    name: str | None = Field(default=None, max_length=200)  # optional run name chosen by the member


# A new name for a dataset or run; null or blank removes it.
class LabelUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=200)


# What the browser sends to import a dataset folder from this computer.
class DatasetImport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=4096)
    dataset_id: str = Field(min_length=1, max_length=128)
    class_names: list[str] = Field(min_length=1, max_length=1000)
    train_images_ref: str = "train/images"
    validation_images_ref: str = "val/images"
    missing_labels: typing.Literal["background", "error"] = "background"
    name: str | None = Field(default=None, max_length=200)  # optional name chosen by the member
    # V4: real labeled images that only score models; optionally divided into validation and test.
    role: typing.Literal["training", "validation", "test"] = "training"
    images_ref: str = "images"
    test_share: float = Field(default=0.0, ge=0.0, le=0.9)


# V8: this installation's member name (shown in shared MLflow as foxtrot.member).
class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    member: str = Field(min_length=1, max_length=64)


# V8: cleanup candidates to remove (IDs from GET /api/storage).
class CleanupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1, max_length=1000)


# V6: which completed trials to repeat (best first) and under which seeds.
class RepeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top: int = Field(default=1, ge=1, le=5)
    seeds: list[int] = Field(min_length=2, max_length=10)


# The allowed values of a Literal field, e.g. the start modes a job may use.
def _choices(model, field: str) -> list[str]:
    return list(typing.get_args(model.model_fields[field].annotation))


# One job as the browser sees it: states, times, errors and its member-chosen name.
def _job_view(record: JobRecord) -> dict:
    return {
        "job_id": record.job_id,
        "execution_state": record.execution_state,
        "tracking_state": record.tracking_state,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "exit_code": record.exit_code,
        "cancel_requested": record.cancel_requested,
        "recovery_required": record.recovery_required,
        "last_event_sequence": record.last_event_sequence,
        "error": record.error_details.model_dump() if record.error_details else None,
        "tracking_error": record.tracking_error.model_dump() if record.tracking_error else None,
    }


def create_app(
    workspace_root: str | Path, source_checkout: str | Path, *, port: int, session_token: str,
    pipeline_adapter: str | None = None, static_directory: Path | None = STATIC_DIRECTORY,
) -> FastAPI:
    """Build the app for one workspace. The database is initialized up front."""
    initialize_database(workspace_root, source_checkout)
    app = FastAPI(title="Foxtrot", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.workspace, app.state.checkout = Path(workspace_root), Path(source_checkout)
    app.state.port, app.state.session_token = port, session_token
    app.state.pipeline_adapter = pipeline_adapter
    app.state.settings = SettingsStore(workspace_root, source_checkout)
    app.state.weights_opener = urllib.request.urlopen  # replaced in tests: no real downloads
    app.state.uploads = UploadScheduler(workspace_root, source_checkout)  # started by the launcher
    app.state.generation = GenerationRuns(workspace_root, source_checkout)  # dataset generation runs
    from foxtrot.studies.coordinator import StudyCoordinator
    app.state.studies = StudyCoordinator(Path(workspace_root), Path(source_checkout))  # started by the launcher
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOOPBACK_HOSTS))
    workspace, checkout = app.state.workspace, app.state.checkout

    # Add the security headers (no sniffing, no referrer, strict content policy) to every response.
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # An unknown ID raised as KeyError becomes a plain 404 answer.
    @app.exception_handler(KeyError)
    async def unknown(_request, error: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(error.args[0]) if error.args else "Not found"})

    # ------------------------------------------------------------ read-only

    # Liveness check: the server answers and reports its database version.
    @app.get("/api/health")
    def health():
        return {"status": "ok", "database_schema_version": DATABASE_SCHEMA_VERSION}

    # What this installation offers: version, models, start modes, devices and settings schema.
    @app.get("/api/capabilities")
    def capabilities():
        try:
            version = importlib.metadata.version("foxtrot-local")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return {
            "version": version,
            "models": _choices(JobSpec, "model_id"),
            "model_details": [{"model_id": m.model_id, "family": m.family, "variant": m.variant,
                               "display_name": m.display_name, "task": m.task} for m in MODELS.values()],
            "devices": ["cpu"] + [d["id"] for d in ((cached_gpus() or {}).get("devices") or [])],
            "pipeline_adapters": _choices(PipelineSettings, "adapter"),
            "active_pipeline_adapter": app.state.pipeline_adapter,
            "real_training": app.state.pipeline_adapter == "ultralytics",
            "tracking_destinations": [d["destination_id"] for d in app.state.settings.destinations()],
            "dvc_default_repository": str(generator_directory().parents[1])
            if (generator_directory().parents[1] / ".dvc").is_dir() else None,
        }

    # The job schema (the form is built from it) plus the order and titles of setting groups.
    @app.get("/api/schemas")
    def schemas():
        return {"job": JobSpec.model_json_schema(), "training_groups": GROUPS}

    # What happens to every runtime and generator setting (exposed, managed, inapplicable, unsupported).
    @app.get("/api/coverage")
    def coverage():
        return coverage_report(ultralytics_default_keys())

    # Finished jobs whose best checkpoint a new job can start from.
    @app.get("/api/checkpoints")
    def checkpoints():
        found, names = [], read_labels(workspace, checkout)["jobs"]
        for record in list_job_records(workspace, checkout, limit=200):
            if record.execution_state != "finished":
                continue
            try:
                spec = JobSpec.model_validate_json(load_job_configurations(workspace, checkout, record.job_id).resolved_json)
                result = JobResult.model_validate_json(
                    job_file(job_path(workspace.resolve(), record.job_id), "result.json").read_bytes())
            except (ValueError, OSError):
                continue
            if result.checkpoint is None or spec.kind != "train":  # evaluations do not make new checkpoints
                continue
            found.append({"job_id": record.job_id, "name": names.get(record.job_id), "model_id": spec.model_id, "dataset_id": spec.dataset_id,
                          "finished_at": record.finished_at,
                          "map50_95": result.evaluation.map50_95 if result.evaluation else None})
        return found

    # Hand the per-launch session token to this website's own pages.
    @app.get("/api/session")
    def session():
        # Same-origin pages only: no CORS headers are sent, and Host is checked.
        return {"token": app.state.session_token}

    # Registered datasets, plus image counts and warning totals from their saved check report.
    @app.get("/api/datasets")
    def datasets():
        found, names = [], read_labels(workspace, checkout)["datasets"]
        for metadata in list_datasets(workspace, checkout):
            view = metadata.model_dump(mode="json")
            view["name"] = names.get(metadata.dataset_id)
            try:
                saved = json.loads((workspace.resolve() / "datasets" / metadata.dataset_id / "validation.json").read_text())
                report = saved.get("report", saved)  # imports wrap the report; generated sets store it directly
                view["images"] = {split: stats.get("images", 0) for split, stats in report.get("splits", {}).items()}
                view["warning_count"] = report.get("warning_count", 0)
            except (OSError, ValueError, AttributeError):
                view["images"], view["warning_count"] = None, None
            found.append(view)
        return found

    # Which pretrained weights are installed (and unchanged) in this workspace.
    @app.get("/api/models")
    def models():
        return list_weights(workspace, checkout)

    # What the Generate screen needs: defaults, limits, and whether the generator is present.
    @app.get("/api/generation/options")
    def generation_options():
        try:
            generator_files()
            available, message = True, None
        except PipelineFailure as error:
            available, message = False, error.message
        backgrounds = generator_directory().parents[1] / "data" / "raw"
        return {"available": available, "message": message,
                "default_backgrounds": str(backgrounds) if backgrounds.is_dir() else None,
                "recipe": GenerationRecipe().model_dump(mode="json"),
                "split": SplitSettings().model_dump(mode="json"),
                "schema": GenerationRequest.model_json_schema()}

    # Progress and outcome of one generation run (preview or dataset).
    @app.get("/api/generation/runs/{run_id}")
    def generation_run(run_id: str):
        return app.state.generation.status(run_id)

    # One sample image from a preview run (the browser draws the boxes on top).
    @app.get("/api/generation/runs/{run_id}/images/{name}")
    def generation_image(run_id: str, name: str):
        return FileResponse(app.state.generation.preview_image(run_id, name), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    # The configured MLflow tracking destinations (never their secrets).
    @app.get("/api/tracking/destinations")
    def destinations():
        return app.state.settings.destinations()

    # What the automatic uploader did last and when it checks next.
    @app.get("/api/uploads")
    def uploads():
        return app.state.uploads.status()

    # Recent jobs with what the runs table shows: kind, model, dataset and study trial.
    @app.get("/api/jobs")
    def jobs(limit: int = 50):
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
        found, names = [], read_labels(workspace, checkout)["jobs"]
        for record in list_job_records(workspace, checkout, limit=limit):
            view = _job_view(record)
            view["name"] = names.get(record.job_id)
            try:
                spec = JobSpec.model_validate_json(load_job_configurations(workspace, checkout, record.job_id).resolved_json)
                view.update(kind="study trial" if spec.study else ("evaluation" if spec.kind == "evaluate" else "training"),
                            model_id=spec.model_id, dataset_id=spec.dataset_id,
                            study=spec.study.model_dump() if spec.study else None)
            except ValueError:
                view.update(kind=None, model_id=None, dataset_id=None, study=None)
            found.append(view)
        return found

    # One job in detail: record, MLflow run link, result and checkpoints.
    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        record = get_job_record(workspace, checkout, job_id)
        association = get_run_association(workspace, checkout, job_id)
        try:
            resolved = JobSpec.model_validate_json(
                load_job_configurations(workspace, checkout, job_id).resolved_json).model_dump(mode="json")
        except ValueError:
            resolved = None  # changed/unsafe configuration; the job itself records why
        try:  # the official outcome and scores, once the job has finished
            result = JobResult.model_validate_json(
                job_file(job_path(workspace.resolve(), job_id), "result.json").read_bytes()).model_dump(mode="json")
        except (ValueError, OSError):
            result = None
        return {**_job_view(record), "name": read_labels(workspace, checkout)["jobs"].get(job_id),
                "resolved": resolved, "result": result,
                "tracking_run": asdict(association) if association else None}

    # The job's progress events after a given position, for live updates in the browser.
    @app.get("/api/jobs/{job_id}/events")
    def events(job_id: str, after: int = 0, limit: int = 200):
        try:
            found = read_job_events(workspace, checkout, job_id, after_sequence=after, limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return [event.model_dump(mode="json") for event in found]

    # ------------------------------------------------------ state-changing

    guarded = [Depends(require_local_session)]

    # A GPU device must exist here; otherwise say so before anything is queued.
    def check_device(device: str) -> None:
        if device == "cpu":
            return
        found = detect_gpus()
        if device not in {d["id"] for d in found["devices"]}:
            raise StartCheckpointError(f"{device} is not available on this computer. {found['message']}")

    # Check and queue a new training or evaluation job (repeating the same key returns the same job).
    @app.post("/api/jobs", dependencies=guarded)
    def submit(submission: JobSubmission):
        try:
            resolved = resolve_job(submission.request)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_input=False)) from error
        try:  # an earlier job's checkpoint must exist, be finished, unchanged, and the same model;
            spec = JobSpec.model_validate_json(resolved.resolved_json)  # and the dataset must suit the job
            check_start_checkpoint(workspace, checkout, spec)
            check_dataset_use(workspace, checkout, spec)
            check_device(spec.training.device)
        except StartCheckpointError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            directory = create_recorded_job(workspace, checkout, submission.request, submission.submission_key)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        if submission.name and submission.name.strip():
            try:
                set_label(workspace, checkout, "jobs", directory.name, submission.name)
            except LabelError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return _job_view(get_job_record(workspace, checkout, directory.name))

    # Name (or rename) a run; the ID stays its identity. Null or blank removes the name.
    @app.put("/api/jobs/{job_id}/name", dependencies=guarded)
    def name_job(job_id: str, body: LabelUpdate):
        get_job_record(workspace, checkout, job_id)  # unknown job -> 404
        try:
            return {"job_id": job_id, "name": set_label(workspace, checkout, "jobs", job_id, body.name)}
        except LabelError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    # Name (or rename) a dataset; its ID and fingerprint stay its identity.
    @app.put("/api/datasets/{dataset_id}/name", dependencies=guarded)
    def name_dataset(dataset_id: str, body: LabelUpdate):
        if dataset_id not in {m.dataset_id for m in list_datasets(workspace, checkout)}:
            raise HTTPException(status_code=404, detail="Unknown dataset")
        try:
            return {"dataset_id": dataset_id, "name": set_label(workspace, checkout, "datasets", dataset_id, body.name)}
        except LabelError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    # Upload this job's unsent results now instead of waiting for the next automatic try.
    @app.post("/api/jobs/{job_id}/retry-sync", dependencies=guarded)
    def retry_sync(job_id: str):
        if not list_tracking(workspace, checkout, job_id, pending_only=True):
            raise HTTPException(status_code=409, detail="Nothing is waiting to be uploaded for this job")
        app.state.uploads.trigger()
        return {"queued": True}

    # Ask the worker to stop a job; a running job stops at the next safe point.
    @app.post("/api/jobs/{job_id}/cancel", dependencies=guarded)
    def cancel(job_id: str):
        return _job_view(request_cancel(workspace, checkout, job_id))

    # Create or change a tracking destination, including its MLflow experiment name.
    @app.put("/api/tracking/destinations/{destination_id}", dependencies=guarded)
    def save_destination(destination_id: str, update: DestinationUpdate):
        if update.destination_id != destination_id:
            raise HTTPException(status_code=422, detail="Destination ID must match the URL")
        return app.state.settings.save_destination(update)

    # Test a saved destination in a separate process, so MLflow and credentials never load here.
    @app.post("/api/tracking/destinations/{destination_id}/check", dependencies=guarded)
    def check_destination(destination_id: str):
        if destination_id not in {d["destination_id"] for d in app.state.settings.destinations()}:
            raise HTTPException(status_code=404, detail="Unknown tracking destination")
        command = [sys.executable, "-B", "-m", "foxtrot.tracking.sync", "--workspace", str(workspace),
                   "--source-checkout", str(checkout), "check", destination_id]
        try:
            result = subprocess.run(command, cwd=workspace, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return {"ok": False, "message": "No answer within 90 seconds."}
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "message": "The check could not run; see the launcher terminal."}

    # Download the official pretrained weights into the workspace and record their fingerprint.
    @app.post("/api/models/{model_id}/install", dependencies=guarded)
    def install_model(model_id: str):
        try:
            record = install_weights(workspace, checkout, model_id, opener=app.state.weights_opener)
        except WeightsError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"model_id": record.model_id, "status": "installed",
                "sha256": record.sha256, "size_bytes": record.size_bytes}

    # Check and import a dataset folder; an invalid dataset returns its report without importing.
    @app.post("/api/datasets", dependencies=guarded)
    def import_local_dataset(body: DatasetImport):
        if body.role != "training":
            if 0 < body.test_share < 0.05:
                raise HTTPException(status_code=422, detail="Test share must be 0 or at least 0.05")
            try:
                found, report = import_evaluation_dataset(
                    workspace, checkout, source=body.source.strip(), dataset_id=body.dataset_id.strip(),
                    class_names=tuple(name.strip() for name in body.class_names), role=body.role,
                    images_ref=body.images_ref, test_share=body.test_share)
            except (DatasetUnavailable, ValueError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            if body.name and found:  # a validation/test pair gets "(validation)" / "(test)" added
                for dataset in found:
                    suffix = f" ({dataset.metadata.role})" if len(found) > 1 else ""
                    set_label(workspace, checkout, "datasets", dataset.metadata.dataset_id, body.name.strip()[:70] + suffix)
            return {"imported": bool(found), "dataset": found[0].metadata.model_dump(mode="json") if found else None,
                    "datasets": [d.metadata.model_dump(mode="json") for d in found],
                    "report": report.model_dump(mode="json")}
        try:
            dataset, report = import_dataset(
                workspace, checkout, source=body.source.strip(), dataset_id=body.dataset_id.strip(),
                class_names=tuple(name.strip() for name in body.class_names),
                train_images_ref=body.train_images_ref, validation_images_ref=body.validation_images_ref,
                missing_labels=body.missing_labels,
            )
        except (DatasetUnavailable, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if dataset is not None and body.name:
            set_label(workspace, checkout, "datasets", dataset.metadata.dataset_id, body.name)
        return {"imported": dataset is not None,
                "dataset": dataset.metadata.model_dump(mode="json") if dataset else None,
                "report": report.model_dump(mode="json")}

    # Start a background run: a few sample images ("preview") or a full dataset ("dataset").
    def start_generation(kind: str, body: GenerationRequest):
        try:
            return app.state.generation.start(kind, body)
        except GenerationBusy as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    # Start a quick preview generation (a few images to look at).
    @app.post("/api/generation/preview", dependencies=guarded)
    def generation_preview(body: GenerationRequest):
        return start_generation("preview", body)

    # Start generating a full dataset with the given recipe.
    @app.post("/api/generation/datasets", dependencies=guarded)
    def generation_dataset(body: GenerationRequest):
        return start_generation("dataset", body)

    # V4: download a dataset version from Git/DVC in the background (same run tracking as generation).
    @app.post("/api/datasets/dvc", dependencies=guarded)
    def dvc_dataset(body: DvcImportRequest):
        try:
            return app.state.generation.start("dvc", body)
        except GenerationBusy as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    # V4: run an exact checkpoint on unlabeled photos and show its boxes (no accuracy).
    @app.post("/api/predictions", dependencies=guarded)
    def prediction_preview(body: PredictionRequest):
        try:
            return app.state.generation.start("predict", body)
        except GenerationBusy as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    # Stop the running generation.
    @app.post("/api/generation/runs/{run_id}/cancel", dependencies=guarded)
    def generation_cancel(run_id: str):
        return app.state.generation.cancel(run_id)

    # ------------------------------------------------------------ V8 setup, profile, storage

    # Which GPUs can train here (detected once in a separate process; refresh=true checks again).
    @app.get("/api/system/gpu")
    def gpu(refresh: bool = False):
        return detect_gpus(refresh=refresh)

    # Is this installation and workspace ready? (read-only)
    @app.get("/api/system/check")
    def system_check_endpoint():
        from .system import system_check
        return system_check(workspace, checkout, static_directory)

    # The member's profile (name used to tag their MLflow runs).
    @app.get("/api/profile")
    def profile():
        from foxtrot.persistence.profile import read_profile
        return read_profile(workspace, checkout)

    # Save the member's name.
    @app.put("/api/profile", dependencies=guarded)
    def save_profile_endpoint(body: ProfileUpdate):
        from foxtrot.persistence.profile import save_profile
        try:
            return save_profile(workspace, checkout, body.member)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    # Disk use per area and what can be removed safely.
    @app.get("/api/storage")
    def storage():
        from foxtrot.persistence.cleanup import plan_cleanup, storage_usage
        return {"usage": storage_usage(workspace, checkout),
                "candidates": plan_cleanup(workspace, checkout, active_run=app.state.generation.active_run())}

    # Remove chosen candidates; the plan is recomputed first, so only still-safe items go.
    @app.post("/api/storage/cleanup", dependencies=guarded)
    def cleanup(body: CleanupRequest):
        from foxtrot.persistence.cleanup import apply_cleanup
        return apply_cleanup(workspace, checkout, body.ids, active_run=app.state.generation.active_run())

    # ------------------------------------------------------------ V7 recovery

    # Everything that needs attention after a failure: what happened and what can be done.
    @app.get("/api/recovery")
    def recovery():
        from foxtrot.persistence.start_checkpoints import RESUMABLE_STATES, StartCheckpointError, check_resume_source
        from foxtrot.studies.store import list_study_ids, read_study, read_trials
        interrupted, holds, uploads = [], [], []
        names = read_labels(workspace, checkout)["jobs"]
        for record in list_job_records(workspace, checkout, limit=500):
            view = _job_view(record)
            view["name"] = names.get(record.job_id)
            try:
                spec = JobSpec.model_validate_json(load_job_configurations(workspace, checkout, record.job_id).resolved_json)
            except ValueError:
                spec = None
            if spec is not None:
                view.update(kind=spec.kind, model_id=spec.model_id, dataset_id=spec.dataset_id,
                            request=spec.model_dump(mode="json", exclude={"schema_version"}, exclude_none=True))
            if record.recovery_required:
                holds.append(view)
            if spec is not None and spec.kind == "train" and record.execution_state in RESUMABLE_STATES:
                try:  # can it continue from its last.pt with the same settings?
                    check_resume_source(workspace, checkout, spec.model_copy(update={
                        "initialization": "resume", "checkpoint_job_id": record.job_id, "study": None}))
                    view.update(resumable=True, resume_reason=None)
                except (StartCheckpointError, ValueError) as error:
                    view.update(resumable=False, resume_reason=str(error))
                interrupted.append(view)
            # Only uploads that actually failed need attention; waiting ones are sent automatically.
            if record.finished_at and record.tracking_state in ("retrying", "failed") \
                    and list_tracking(workspace, checkout, record.job_id):
                uploads.append(view)
        studies = []
        for study_id in list_study_ids(workspace, checkout):
            spec, record, state = read_study(workspace, checkout, study_id)
            started = len(read_trials(workspace, checkout, study_id))
            if state["state"] == "stopped" or (state["state"] == "finished" and started < record["planned_trials"]):
                studies.append({"study_id": study_id, "name": spec.name, "state": state["state"],
                                "reason": state.get("reason"), "trials_started": started,
                                "planned_trials": record["planned_trials"]})
        return {"interrupted": interrupted, "holds": holds, "uploads": uploads, "studies": studies,
                "uploader": app.state.uploads.status()}

    # --------------------------------------------------------- V5/V6 studies

    # What the study form needs: tunable settings, recipe parameters, objectives, samplers.
    @app.get("/api/studies/options")
    def study_options():
        return {"training": [name for name in TrainingSettings.model_fields if name != "device"],
                "recipe": sorted(f"recipe.{n}" for n in RECIPE_SCALARS)
                + sorted(f"recipe.{r}.{end}" for r in RECIPE_RANGES for end in ("low", "span")),
                "objectives": list(OBJECTIVES), "samplers": ["grid", "random", "tpe"],
                "schema": StudySpec.model_json_schema()}

    # Check a study definition and count its trials without starting it.
    @app.post("/api/studies/plan", dependencies=guarded)
    def plan_study(spec: StudySpec):
        return {"planned_trials": planned_trials(spec)}

    # All tuning studies with a short summary each.
    @app.get("/api/studies")
    def studies():
        from foxtrot.studies.store import list_study_ids, read_study, read_trials
        from foxtrot.studies.summary import summarize
        found = []
        for study_id in list_study_ids(workspace, checkout):
            spec, record, state = read_study(workspace, checkout, study_id)
            summary = summarize(spec, record, state, read_trials(workspace, checkout, study_id))
            found.append({"study_id": study_id, "name": spec.name, "created_at": record["created_at"],
                          "state": state["state"], "planned_trials": record["planned_trials"],
                          "trials_started": summary["trials_started"], "best": summary["best"]})
        return found

    # One tuning study: definition, trials, best result and state.
    @app.get("/api/studies/{study_id}")
    def study(study_id: str):
        from foxtrot.studies.store import read_study, read_trials, study_folder
        from foxtrot.studies.summary import summarize
        spec, record, state = read_study(workspace, checkout, study_id)
        trials = read_trials(workspace, checkout, study_id)
        mlflow = study_folder(workspace, checkout, study_id) / "mlflow.json"
        return {"study_id": study_id, "spec": spec.model_dump(mode="json"), "record": record, "state": state,
                "trials": trials, "summary": summarize(spec, record, state, trials),
                "mlflow": json.loads(mlflow.read_text()) if mlflow.is_file() else {}}

    # Check and create a new tuning study; the coordinator then starts its trials.
    @app.post("/api/studies", dependencies=guarded)
    def create_study_endpoint(spec: StudySpec):
        from foxtrot.studies.store import StudyError, create_study
        try:
            check_device(str(spec.base_training.get("device", "cpu")))
            study_id = create_study(workspace, checkout, spec)
        except (StudyError, StartCheckpointError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        app.state.studies.trigger()
        return {"study_id": study_id, "planned_trials": planned_trials(spec)}

    # Stop launching trials; the active trial either finishes or is cancelled.
    @app.post("/api/studies/{study_id}/stop", dependencies=guarded)
    def stop_study_endpoint(study_id: str, active: typing.Literal["finish", "cancel"] = "finish"):
        from foxtrot.studies.coordinator import stop_study
        try:
            state = stop_study(workspace, checkout, study_id, active)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        app.state.studies.trigger()
        return state

    # Continue a stopped/finished study within its remaining budget (completed trials are kept).
    @app.post("/api/studies/{study_id}/resume", dependencies=guarded)
    def resume_study_endpoint(study_id: str):
        from foxtrot.studies.coordinator import resume_study
        try:
            state = resume_study(workspace, checkout, study_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        app.state.studies.trigger()
        return state

    # V6: repeat the best configurations of a study under several training seeds (a new repeat study).
    @app.post("/api/studies/{study_id}/repeat", dependencies=guarded)
    def repeat_study_endpoint(study_id: str, body: RepeatRequest):
        from foxtrot.studies.store import StudyError, create_study, read_study, read_trials
        spec, _, _ = read_study(workspace, checkout, study_id)
        if spec.recipe is not None:
            raise HTTPException(status_code=409, detail="Repeating recipe-study trials is not supported yet: "
                                "each trial has its own generated dataset")
        complete = sorted((t for t in read_trials(workspace, checkout, study_id) if t["state"] == "complete"),
                          key=lambda t: -t["value"])[:body.top]
        if not complete:
            raise HTTPException(status_code=409, detail="The study has no completed trial to repeat")
        repeats = [{"model_id": t["model_id"], "seed": seed, "source_trial": t["number"],
                    "training": {k: v for k, v in t["training"].items() if k != "seed"}}
                   for t in complete for seed in body.seeds]
        try:
            repeat = StudySpec.model_validate({**spec.model_dump(mode="json"), "name": f"{spec.name} (repeats)",
                                               "sampler": "repeat", "search_space": [], "repeats": repeats,
                                               "base_training": {}, "pruning": False, "recipe": None,
                                               "dataset_id": spec.dataset_id or complete[0]["dataset_id"],
                                               "model_ids": sorted({t["model_id"] for t in complete})})
            new_id = create_study(workspace, checkout, repeat)
        except (ValidationError, StudyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        app.state.studies.trigger()
        return {"study_id": new_id, "planned_trials": len(repeats)}

    # ------------------------------------------------------------ interface

    # Any other /api path is an unknown endpoint (never the web page).
    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unknown_api(rest: str):
        raise HTTPException(status_code=404, detail="Unknown API endpoint")

    if static_directory is not None and (static_directory / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_directory, html=True), name="interface")
    else:
        # Shown instead of the website when the interface was not built.
        @app.get("/", response_class=HTMLResponse)
        def interface_missing():
            return ("<!doctype html><title>Foxtrot</title><p>The browser interface has not been built. "
                    "Run <code>npm run build</code> in <code>frontend/</code>; the API is available under /api.</p>")

    return app
