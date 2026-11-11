"""Fixed execution entry point: the H5 fixture or an H6 code-registered adapter."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from foxtrot.config.job import JobSpec
from foxtrot.config.lifecycle import ErrorDetails
from foxtrot.persistence.files import job_file, job_path, sync_directory, write_new_text
from foxtrot.pipeline.contracts import (
    EvaluationInput, PipelineCancelled, PipelineFailure, PipelineResult, TrainingInput,
)
from foxtrot.pipeline.registry import get_adapters
from foxtrot.pipeline.yolo_dataset import write_yolo_data_yaml

from .protocol import REQUEST, Completed, ExecutionRequest, PipelineExecutionRequest, Phase, Ready, Update


def load_request(path: Path) -> ExecutionRequest | PipelineExecutionRequest:
    """Check managed paths and exact submitted bytes before running anything."""
    if not path.is_absolute() or path.name != "execution.json":
        raise ValueError("Use an absolute managed execution request")
    request = REQUEST.validate_json(job_file(path.parent, path.name).read_bytes())
    workspace = path.parents[2]
    directory = job_path(workspace, request.job_id)
    if (directory != path.parent or str(directory) != request.job_directory
            or request.output_directory != str(directory / "output")):
        raise ValueError("Execution paths do not belong to this job")
    output = directory / "output"
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Unsafe execution output directory")
    resolved = job_file(directory, "resolved.json").read_bytes()
    if (hashlib.sha256(resolved).hexdigest() != request.resolved_sha256
            or JobSpec.model_validate_json(resolved) != request.resolved):
        raise ValueError("Execution configuration identity mismatch")
    if isinstance(request, PipelineExecutionRequest):
        # The dataset must be the registered one for this job, inside this workspace.
        expected = workspace / "datasets" / request.resolved.dataset_id / "data"
        root = Path(request.dataset.root)
        if (request.dataset.metadata.dataset_id != request.resolved.dataset_id
                or root != expected or root.is_symlink() or not root.is_dir()):
            raise ValueError("Execution dataset does not belong to this job")
    return request


class _Reporter:
    """Adapter progress becomes pipe messages; the parent saves them."""

    def __init__(self, emit, cancelled: threading.Event):
        self.emit, self.cancelled = emit, cancelled

    def progress(self, payload: dict) -> None:
        self.emit(Update(event_type="progress", payload=payload))

    def log(self, payload: dict) -> None:
        self.emit(Update(event_type="log", payload=payload))

    def wait_cancelled(self, seconds: float = 0.0) -> bool:
        return self.cancelled.wait(seconds) if seconds > 0 else self.cancelled.is_set()


def run_pipeline(request: PipelineExecutionRequest, emit, cancelled: threading.Event) -> int:
    """Generate the job's data.yaml, train, evaluate the exact checkpoint, report."""
    adapters = get_adapters(request.pipeline)
    reporter = _Reporter(emit, cancelled)
    directory, output = Path(request.job_directory), Path(request.output_directory)
    try:
        emit(Phase(phase="training"))
        data_yaml = write_yolo_data_yaml(request.dataset, output / "data.yaml")
        training = adapters.trainer.train(TrainingInput(
            job_id=request.job_id, resolved=request.resolved, resolved_sha256=request.resolved_sha256,
            dataset=request.dataset, data_yaml=str(data_yaml),
            job_directory=str(directory), output_directory=str(output),
        ), reporter)
        if cancelled.is_set():
            raise PipelineCancelled()
        emit(Phase(phase="evaluating"))
        evaluation = adapters.evaluator.evaluate(EvaluationInput(
            job_id=request.job_id, checkpoint=training.checkpoint,
            job_directory=str(directory), dataset=request.dataset,
        ), reporter)
        result = PipelineResult(job_id=request.job_id, adapter=request.pipeline.adapter,
                                training=training, evaluation=evaluation)
        write_new_text(output / "pipeline-result.json", result.model_dump_json(indent=2))
        sync_directory(output)
        emit(Completed(outcome="finished"))
        return 0
    except PipelineCancelled:
        emit(Completed(outcome="cancelled"))
        return 130
    except PipelineFailure as failure:
        print(f"pipeline failure: {failure.code}", file=sys.stderr, flush=True)
        emit(Completed(outcome="failed", error=ErrorDetails(code=failure.code, message=failure.message)))
        return 1


def run_fixture(request: ExecutionRequest, emit, cancelled: threading.Event) -> int:
    """Emit phases/progress, exercise output capture, and optionally own a child."""
    settings = request.fixture
    child = None
    try:
        emit(Phase(phase="training"))
        if settings.ignore_cancel:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if settings.spawn_child:
            program = """
import os, signal, sys, time
if sys.argv[1] == '1':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sys.argv[2] == '1':
    os.setpgrp()
print('ready', flush=True)
while True:
    time.sleep(1)
"""
            child = subprocess.Popen(
                [sys.executable, "-B", "-c", program, str(int(settings.child_ignores_term)), str(int(settings.child_new_group))],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            )
            if child.stdout.readline() != b"ready\n":
                raise RuntimeError("Fixture child did not start")
            child.stdout.close()
            emit(Update(event_type="log", payload={"fixture_child_pid": child.pid}))
        for index in range(settings.output_lines):
            print(f"fixture stdout {index}: " + "x" * 200)
            print(f"fixture stderr {index}: " + "y" * 200, file=sys.stderr)
        for step in range(1, settings.steps + 1):
            if cancelled.wait(settings.step_seconds):
                emit(Completed(outcome="cancelled"))
                return 130
            print(f"fixture step {step}", flush=True)
            emit(Update(payload={"step": step, "total": settings.steps}))
            if settings.fail_at_step == step:
                print("controlled fixture failure", file=sys.stderr, flush=True)
                emit(Completed(outcome="failed"))
                return 17
        emit(Phase(phase="evaluating"))
        result = {"fixture": True, "job_id": request.job_id,
                  "resolved": request.resolved.model_dump(mode="json"), "steps": settings.steps}
        output = Path(request.output_directory)
        write_new_text(output / "fixture-result.json", json.dumps(result, indent=2))
        sync_directory(output)
        emit(Completed(outcome="finished"))
        return 0
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--events-fd", required=True, type=int)
    arguments = parser.parse_args()
    request = load_request(arguments.request)
    cancelled = threading.Event()
    with os.fdopen(arguments.events_fd, "w", encoding="utf-8", buffering=1) as events:
        def emit(message):
            events.write(message.model_dump_json() + "\n")
            events.flush()

        emit(Ready())
        if sys.stdin.readline() != '{"command":"start"}\n':
            return 2

        # Only the H5 fixture can be told to ignore cancel; pipeline requests have no fixture.
        ignore_cancel = request.fixture is not None and request.fixture.ignore_cancel

        def commands():
            for line in sys.stdin:
                if line == '{"command":"cancel"}\n' and not ignore_cancel:
                    cancelled.set()
            # Losing the worker's pipe is also a graceful stop request.
            cancelled.set()

        threading.Thread(target=commands, daemon=True).start()
        if isinstance(request, PipelineExecutionRequest):
            return run_pipeline(request, emit, cancelled)
        return run_fixture(request, emit, cancelled)


if __name__ == "__main__":
    raise SystemExit(main())
