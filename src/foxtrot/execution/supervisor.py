"""Bounded execution supervision, independent of HTTP requests and browsers."""

from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time

from foxtrot.config.job import JobSpec
from foxtrot.config.lifecycle import TERMINAL_STATES, ErrorDetails, ExecutionState
from foxtrot.persistence.artifacts import record_job_artifact
from foxtrot.persistence.datasets import DatasetUnavailable, load_dataset
from foxtrot.persistence.events import append_job_event
from foxtrot.persistence.files import job_file, sync_directory, write_new_text
from foxtrot.persistence.jobs import JobClaim, JobRecord, get_job_record
from foxtrot.persistence.lifecycle import transition_execution
from foxtrot.persistence.processes import (
    Liveness, capture_execution_process, probe_execution_group,
)
from foxtrot.persistence.recovery import (
    attach_execution_process, begin_process_launch, hold_owned_execution, settle_failed_launch,
)

from foxtrot.pipeline.registry import PipelineSettings

from .outcomes import accept_pipeline_result, record_pipeline_outcome
from .process_control import stop_session, signal_session
from .protocol import (
    Completed, ExecutionRequest, FixtureSettings, MAX_MESSAGE_BYTES, MESSAGE,
    Phase, PipelineExecutionRequest, Ready, Update,
)


@dataclass(frozen=True)
class SupervisorLimits:
    poll_seconds: float = 0.05
    startup_seconds: float = 10.0
    grace_seconds: float = 1.0
    terminate_seconds: float = 1.0
    kill_seconds: float = 2.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in self.__dict__.values()):
            raise ValueError("Supervisor time limits must be positive and finite")


class Supervisor:
    def __init__(self, workspace: Path, checkout: Path, runner: FixtureSettings | PipelineSettings,
                 stop: threading.Event, limits: SupervisorLimits):
        if not isinstance(runner, (FixtureSettings, PipelineSettings)):
            raise TypeError("Supervisor needs trusted FixtureSettings or PipelineSettings")
        self.workspace, self.checkout = workspace, checkout
        self.runner, self.stop, self.limits = runner, stop, limits

    def record(self, claim):
        return get_job_record(self.workspace, self.checkout, claim.job_id)

    def update(self, claim, event_type, payload):
        return append_job_event(self.workspace, self.checkout, claim.job_id, event_type, payload,
                                claim_token=claim.claim_token)

    def phase(self, claim, target, **options):
        return transition_execution(self.workspace, self.checkout, claim.job_id, target,
                                    claim_token=claim.claim_token, **options)

    def run(self, claim: JobClaim) -> JobRecord:
        """Prepare explicit files, launch a fixed child, and always supervise cleanup."""
        process = identity = None
        read_fd = write_fd = None
        launch_pending = False
        try:
            if self.record(claim).cancel_requested:
                return self.phase(claim, ExecutionState.CANCELLED)
            if self.stop.is_set():
                return self.phase(claim, ExecutionState.INTERRUPTED,
                                  error=ErrorDetails(code="worker_shutdown", message="The worker stopped before launching execution."))
            directory = claim.directory
            resolved = JobSpec.model_validate_json(claim.configurations.resolved_json)
            dataset = None
            if isinstance(self.runner, PipelineSettings):
                try:
                    # Verified in the parent before launch: manifest, paths, content hash.
                    dataset = load_dataset(self.workspace, self.checkout, resolved.dataset_id)
                except DatasetUnavailable:
                    return self.phase(claim, ExecutionState.FAILED, error=ErrorDetails(
                        code="dataset_unavailable",
                        message=f"Dataset {resolved.dataset_id} is not registered, is unsafe, or changed.",
                    ))
            for name in ("logs", "output"):
                (directory / name).mkdir(exist_ok=False)
            common = dict(
                job_id=claim.job_id, job_directory=str(directory), output_directory=str(directory / "output"),
                resolved_sha256=self.record(claim).resolved_sha256, resolved=resolved,
            )
            request = (PipelineExecutionRequest(**common, pipeline=self.runner, dataset=dataset)
                       if dataset is not None else ExecutionRequest(**common, fixture=self.runner))
            request_path = directory / "execution.json"
            write_new_text(request_path, request.model_dump_json(indent=2))
            sync_directory(directory)
            read_fd, write_fd = os.pipe()
            begin_process_launch(self.workspace, self.checkout, claim.job_id, claim.claim_token)
            launch_pending = True
            process = subprocess.Popen(
                [sys.executable, "-B", "-m", "foxtrot.execution.child", "--request", str(request_path),
                 "--events-fd", str(write_fd)],
                cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=(write_fd,), start_new_session=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            )
            os.close(write_fd)
            write_fd = None
            # The child waits for START. Even a very short fixture cannot exit
            # normally between successful spawn and durable ownership recording.
            identity = capture_execution_process(process.pid)
            attach_execution_process(self.workspace, self.checkout, claim.job_id, claim.claim_token, process.pid)
            launch_pending = False
            owned_events_fd, read_fd = read_fd, None
            result = self._monitor(claim, process, identity, owned_events_fd, request)
            return result
        except Exception as error:
            # Includes invalid protocol, filesystem/DB errors, and launch errors.
            # Do not release capacity while any owned execution might survive.
            stopped = process is None
            if process is not None:
                if identity is not None:
                    try:
                        stopped = stop_session(identity, terminate_seconds=self.limits.terminate_seconds,
                                               kill_seconds=self.limits.kill_seconds)
                    except Exception:
                        stopped = False
                else:
                    # A returned child that died before identity capture cannot
                    # have received START or run the adapter.
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=self.limits.kill_seconds)
                    stopped = True
                if stopped:
                    process.wait(timeout=self.limits.kill_seconds)
                else:
                    process.poll()  # Reap an exited leader even if child cleanup is uncertain.
            if not stopped:
                hold_owned_execution(self.workspace, self.checkout, claim.job_id, claim.claim_token)
                raise RuntimeError("Execution cleanup is uncertain; claim remains blocked")
            if self.record(claim).execution_state in TERMINAL_STATES:
                raise
            if launch_pending:
                settle_failed_launch(self.workspace, self.checkout, claim.job_id, claim.claim_token,
                                     stopped_child=identity)
            target = ExecutionState.INTERRUPTED if self.stop.is_set() else ExecutionState.FAILED
            return self.phase(claim, target,
                              error=ErrorDetails(code="supervision_failed", message=f"Execution setup/supervision failed ({type(error).__name__}); inspect local logs."),
                              exit_code=process.returncode if process is not None else None)
        finally:
            for descriptor in (read_fd, write_fd):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    def _monitor(self, claim, process, identity, events_fd, request):
        """Drain all pipes without blocking cancellation or growing memory unbounded."""
        ready = False
        completed = None
        buffered = b""
        stopping = None
        stage = 0
        deadline = time.monotonic() + self.limits.startup_seconds
        root_exited = None
        with ExitStack() as stack:
            selector = stack.enter_context(selectors.DefaultSelector())
            stack.callback(os.close, events_fd)
            logs = {}
            for channel, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                logs[channel] = stack.enter_context((claim.directory / "logs" / f"{channel}.log").open("xb"))
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, channel)
            os.set_blocking(events_fd, False)
            selector.register(events_fd, selectors.EVENT_READ, "events")
            while True:
                now = time.monotonic()
                record = self.record(claim)
                if stopping is None and (record.cancel_requested or self.stop.is_set()):
                    stopping = "cancel" if record.cancel_requested else "shutdown"
                    self.update(claim, "log", {"action": "graceful_stop", "reason": stopping})
                    try:
                        process.stdin.write(b'{"command":"cancel"}\n')
                        process.stdin.flush()
                    except BrokenPipeError:
                        pass
                    stage, deadline = 1, now + self.limits.grace_seconds
                if not ready and stopping is None and now >= deadline:
                    stopping, stage, deadline = "startup_timeout", 1, now
                if stage and now >= deadline and probe_execution_group(identity) != Liveness.DEAD:
                    number = signal.SIGTERM if stage == 1 else signal.SIGKILL
                    signal_session(identity, number)
                    self.update(claim, "log", {"action": "termination_signal", "signal": int(number)})
                    if stage >= 3:
                        raise RuntimeError("Owned execution did not stop within the kill deadline")
                    stage += 1
                    deadline = now + (self.limits.terminate_seconds if stage == 2 else self.limits.kill_seconds)
                for key, _ in selector.select(self.limits.poll_seconds):
                    data = os.read(key.fd, 65536)
                    channel = key.data
                    if not data:
                        selector.unregister(key.fileobj)
                        if channel == "events" and buffered:
                            raise ValueError("Truncated execution message")
                        continue
                    if channel != "events":
                        logs[channel].write(data)
                        logs[channel].flush()
                        self.update(claim, "log", {"channel": channel, "text": data[:4096].decode("utf-8", "replace"),
                                                   "bytes": len(data), "truncated": len(data) > 4096})
                        continue
                    buffered += data
                    if len(buffered) > MAX_MESSAGE_BYTES and b"\n" not in buffered:
                        raise ValueError("Execution message exceeded size limit")
                    while b"\n" in buffered:
                        line, buffered = buffered.split(b"\n", 1)
                        if len(line) > MAX_MESSAGE_BYTES:
                            raise ValueError("Execution message exceeded size limit")
                        message = MESSAGE.validate_json(line)
                        if not ready:
                            if not isinstance(message, Ready):
                                raise ValueError("Execution did not begin with READY")
                            ready = True
                            if stopping is None:
                                process.stdin.write(b'{"command":"start"}\n')
                                process.stdin.flush()
                        elif completed is not None:
                            raise ValueError("Unexpected event after completion")
                        elif isinstance(message, Phase):
                            self.phase(claim, ExecutionState(message.phase))
                        elif isinstance(message, Update):
                            self.update(claim, message.event_type, message.payload)
                        elif isinstance(message, Completed):
                            completed = message
                        else:
                            raise ValueError("Unexpected execution handshake")
                if process.poll() is not None:
                    if root_exited is None:
                        root_exited = time.monotonic()
                    if probe_execution_group(identity) == Liveness.DEAD:
                        if not selector.get_map():
                            break
                    elif stopping is None:
                        stopping, stage, deadline = "orphaned_children", 1, time.monotonic()
                    # Do not wait forever for leaked pipes after the session dies.
                    if time.monotonic() - root_exited > (self.limits.grace_seconds + self.limits.terminate_seconds + self.limits.kill_seconds + 2):
                        raise RuntimeError("Execution pipes or children remained open after exit")
            process.wait(timeout=self.limits.kill_seconds)
            for stream in logs.values():
                stream.flush()
                os.fsync(stream.fileno())
        for channel in ("stdout", "stderr"):
            record_job_artifact(self.workspace, self.checkout, claim.job_id, f"{channel}-log", f"logs/{channel}.log")
        pipeline_result = None
        if stopping == "shutdown":
            target = ExecutionState.INTERRUPTED
            error = ErrorDetails(code="worker_shutdown", message="The worker shut down during execution.")
        elif self.record(claim).cancel_requested or (completed and completed.outcome == "cancelled"):
            target, error = ExecutionState.CANCELLED, None
        elif stopping is None and completed and completed.outcome == "finished" and process.returncode == 0:
            if isinstance(request, PipelineExecutionRequest):
                # Re-verify the child's report: identities, dataset, checkpoint hash.
                pipeline_result = accept_pipeline_result(self.workspace, self.checkout, claim, request)
            else:
                result = job_file(claim.directory, "output/fixture-result.json")
                value = json.loads(result.read_bytes())
                if value.get("job_id") != claim.job_id or value.get("fixture") is not True:
                    raise ValueError("Fixture result identity mismatch")
                record_job_artifact(self.workspace, self.checkout, claim.job_id, "fixture-result", "output/fixture-result.json")
            target, error = ExecutionState.FINISHED, None
        else:
            target = ExecutionState.FAILED
            error = (completed.error if completed is not None and completed.error is not None
                     else ErrorDetails(code="execution_failed", message=f"Execution exited with code {process.returncode}; inspect local logs."))
        record = self.phase(claim, target, error=error, exit_code=process.returncode)
        fixture = not isinstance(request, PipelineExecutionRequest) or request.pipeline.adapter == "fixture"
        summary = {"schema_version": 1, "fixture": fixture, "runner": request.runner, "job_id": claim.job_id,
                   "execution_state": record.execution_state, "tracking_state": record.tracking_state,
                   "exit_code": record.exit_code, "error": record.error_details.model_dump() if record.error_details else None}
        write_new_text(claim.directory / "summary.json", json.dumps(summary, indent=2))
        sync_directory(claim.directory)
        record_job_artifact(self.workspace, self.checkout, claim.job_id, "summary", "summary.json")
        if isinstance(request, PipelineExecutionRequest):
            # After the outcome commits: write the H2 result and queue its upload.
            record_pipeline_outcome(self.workspace, self.checkout, claim, request, record, pipeline_result)
        return record
