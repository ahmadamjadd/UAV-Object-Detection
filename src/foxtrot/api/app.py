"""The local FastAPI application: browser interface plus job API.

It records and reads state only. Heavy work belongs to the separate worker;
no request ever trains, so the website stays responsive during training.
"""

from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import typing

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from foxtrot.config.job import JobSpec
from foxtrot.config.training import TrainingSettings
from foxtrot.config.snapshot import resolve_job
from foxtrot.persistence.commands import request_cancel
from foxtrot.persistence.database import DATABASE_SCHEMA_VERSION, initialize_database
from foxtrot.persistence.datasets import list_datasets
from foxtrot.persistence.events import read_job_events
from foxtrot.persistence.jobs import (
    JobRecord, create_recorded_job, get_job_record, list_job_records, load_job_configurations,
)
from foxtrot.persistence.tracking import get_run_association
from foxtrot.pipeline.registry import PipelineSettings

from .security import LOOPBACK_HOSTS, require_local_session
from .settings import DestinationUpdate, SettingsStore

STATIC_DIRECTORY = Path(__file__).with_name("static")


class JobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_key: str = Field(min_length=8, max_length=200, pattern=r"^\S+$")
    request: dict[str, typing.Any]


def _choices(model, field: str) -> list[str]:
    return list(typing.get_args(model.model_fields[field].annotation))


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
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOOPBACK_HOSTS))
    workspace, checkout = app.state.workspace, app.state.checkout

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

    @app.exception_handler(KeyError)
    async def unknown(_request, error: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(error.args[0]) if error.args else "Not found"})

    # ------------------------------------------------------------ read-only

    @app.get("/api/health")
    def health():
        return {"status": "ok", "database_schema_version": DATABASE_SCHEMA_VERSION}

    @app.get("/api/capabilities")
    def capabilities():
        try:
            version = importlib.metadata.version("foxtrot-local")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return {
            "version": version,
            "models": _choices(JobSpec, "model_id"),
            "devices": _choices(TrainingSettings, "device"),
            "pipeline_adapters": _choices(PipelineSettings, "adapter"),
            "active_pipeline_adapter": app.state.pipeline_adapter,
            "real_training": False,  # becomes True with the V1 Ultralytics adapter
            "tracking_destinations": [d["destination_id"] for d in app.state.settings.destinations()],
        }

    @app.get("/api/schemas")
    def schemas():
        return {"job": JobSpec.model_json_schema()}

    @app.get("/api/session")
    def session():
        # Same-origin pages only: no CORS headers are sent, and Host is checked.
        return {"token": app.state.session_token}

    @app.get("/api/datasets")
    def datasets():
        return [m.model_dump(mode="json") for m in list_datasets(workspace, checkout)]

    @app.get("/api/tracking/destinations")
    def destinations():
        return app.state.settings.destinations()

    @app.get("/api/jobs")
    def jobs(limit: int = 50):
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
        return [_job_view(record) for record in list_job_records(workspace, checkout, limit=limit)]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        record = get_job_record(workspace, checkout, job_id)
        association = get_run_association(workspace, checkout, job_id)
        try:
            resolved = JobSpec.model_validate_json(
                load_job_configurations(workspace, checkout, job_id).resolved_json).model_dump(mode="json")
        except ValueError:
            resolved = None  # changed/unsafe configuration; the job itself records why
        return {**_job_view(record), "resolved": resolved,
                "tracking_run": asdict(association) if association else None}

    @app.get("/api/jobs/{job_id}/events")
    def events(job_id: str, after: int = 0, limit: int = 200):
        try:
            found = read_job_events(workspace, checkout, job_id, after_sequence=after, limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return [event.model_dump(mode="json") for event in found]

    # ------------------------------------------------------ state-changing

    guarded = [Depends(require_local_session)]

    @app.post("/api/jobs", dependencies=guarded)
    def submit(submission: JobSubmission):
        try:
            resolve_job(submission.request)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_input=False)) from error
        try:
            directory = create_recorded_job(workspace, checkout, submission.request, submission.submission_key)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return _job_view(get_job_record(workspace, checkout, directory.name))

    @app.post("/api/jobs/{job_id}/cancel", dependencies=guarded)
    def cancel(job_id: str):
        return _job_view(request_cancel(workspace, checkout, job_id))

    @app.put("/api/tracking/destinations/{destination_id}", dependencies=guarded)
    def save_destination(destination_id: str, update: DestinationUpdate):
        if update.destination_id != destination_id:
            raise HTTPException(status_code=422, detail="Destination ID must match the URL")
        return app.state.settings.save_destination(update)

    # ------------------------------------------------------------ interface

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unknown_api(rest: str):
        raise HTTPException(status_code=404, detail="Unknown API endpoint")

    if static_directory is not None and (static_directory / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_directory, html=True), name="interface")
    else:
        @app.get("/", response_class=HTMLResponse)
        def interface_missing():
            return ("<!doctype html><title>Foxtrot</title><p>The browser interface has not been built. "
                    "Run <code>npm run build</code> in <code>frontend/</code>; the API is available under /api.</p>")

    return app
