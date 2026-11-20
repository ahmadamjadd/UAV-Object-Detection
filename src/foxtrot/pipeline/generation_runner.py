"""Separate process that runs one background request for the website.

Kinds: "preview"/"dataset" (V2 generation), "dvc" (V4 dataset download) and
"predict" (V4 prediction preview on unlabeled images).

    python -m foxtrot.pipeline.generation_runner --workspace W --source-checkout C --run RUN_FOLDER

The website starts it so heavy image work never slows the API down. It reads
``request.json`` from the run folder, keeps ``status.json`` up to date
(replaced atomically, so the website never reads half a file), and stops early
when a ``cancel`` file appears.
"""

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict

from foxtrot.config.generation import GenerationRequest
from foxtrot.persistence.dvc_datasets import DvcImportRequest
from foxtrot.pipeline.contracts import PipelineCancelled, PipelineFailure
from foxtrot.pipeline.prediction import PredictionRequest

REQUEST, STATUS, CANCEL, PREVIEW = "request.json", "status.json", "cancel", "preview"
FINAL_STATES = {"finished", "failed", "cancelled", "interrupted"}


# What the website asked this run to do; exactly the field for its kind is set.
class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["preview", "dataset", "dvc", "predict"]
    request: GenerationRequest | None = None
    dvc: DvcImportRequest | None = None
    predict: PredictionRequest | None = None


# Current UTC time for status timestamps.
def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# Replace status.json in one step: write a temporary file, then rename it over the old one.
def write_status(run: Path, status: dict) -> None:
    temporary = run / f".{STATUS}.{os.getpid()}"
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary, run / STATUS)


# Progress reporter that keeps status.json current and watches for the cancel file.
class _StatusReporter:
    def __init__(self, run: Path, status: dict):
        self.run, self.status = run, status

    # Merge new progress (phase, done, total) into the status and save it.
    def progress(self, payload: dict) -> None:
        self.status.update({key: payload[key] for key in ("phase", "done", "total") if key in payload})
        write_status(self.run, self.status)

    # Free-text messages go to the run's log (stderr).
    def log(self, payload: dict) -> None:
        print(json.dumps(payload), file=sys.stderr, flush=True)

    # True once the website asked to cancel (it creates the "cancel" file).
    def wait_cancelled(self, seconds: float = 0.0) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            if (self.run / CANCEL).exists():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


# Run the request in the run folder and record how it ended; returns the exit code.
def run(workspace: Path, checkout: Path, folder: Path) -> int:
    from foxtrot.persistence.dataset_generation import build_generated_dataset, generate_preview

    status = json.loads((folder / STATUS).read_text(encoding="utf-8"))
    status.update({"state": "running", "phase": "starting", "started_at": now()})
    write_status(folder, status)
    reporter = _StatusReporter(folder, status)
    try:
        order = RunRequest.model_validate_json((folder / REQUEST).read_bytes())
        if order.kind == "preview":
            result = generate_preview(workspace, checkout, order.request, folder / PREVIEW, reporter)
        elif order.kind == "dvc":
            from foxtrot.persistence.dvc_datasets import import_dvc_dataset
            dataset, provenance = import_dvc_dataset(workspace, checkout, order.dvc, reporter)
            result = {"dataset_id": dataset.metadata.dataset_id, "role": dataset.metadata.role,
                      "commit": provenance["commit"], "dvc_md5": provenance["dvc_md5"],
                      "class_names": list(dataset.metadata.class_names)}
        elif order.kind == "predict":
            from foxtrot.persistence.workspace import initialize_workspace
            from foxtrot.pipeline.prediction import predict_preview
            result = predict_preview(initialize_workspace(workspace, checkout), order.predict,
                                     folder / PREVIEW, reporter)
        else:
            built = build_generated_dataset(workspace, checkout, order.request, reporter)
            provenance = built.provenance
            result = {"dataset_id": built.dataset.metadata.dataset_id, "cached": built.cached,
                      "class_names": list(built.dataset.metadata.class_names),
                      "counts": provenance.get("counts"), "groups": {
                          split: len(groups) for split, groups in provenance.get("groups", {}).items()},
                      "report": built.report.model_dump(mode="json") if built.report else None}
        status.update({"state": "finished", "phase": "done", "result": result})
        code = 0
    except PipelineCancelled:
        status.update({"state": "cancelled", "phase": "cancelled"})
        code = 130
    except PipelineFailure as failure:
        status.update({"state": "failed", "error": {"code": failure.code, "message": failure.message}})
        code = 1
    except Exception as error:  # unexpected: keep the message, never a stack trace in the UI
        print(f"generation failed: {error!r}", file=sys.stderr, flush=True)
        status.update({"state": "failed", "error": {"code": "internal", "message": str(error)[:500]}})
        code = 1
    status["finished_at"] = now()
    write_status(folder, status)
    return code


# Command-line entry point used by the website.
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    arguments = parser.parse_args(argv)
    return run(arguments.workspace, arguments.source_checkout, arguments.run)


if __name__ == "__main__":
    raise SystemExit(main())
