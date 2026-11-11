"""One local worker and fixed subprocess supervisor (H5 fixture or H6 pipeline adapter)."""

import argparse
import os
from pathlib import Path
import signal
import sys
import threading

from foxtrot.config.lifecycle import ErrorDetails, ExecutionState
from foxtrot.persistence.jobs import InvalidJobConfiguration, claim_next_job
from foxtrot.persistence.lifecycle import transition_execution
from foxtrot.persistence.recovery import reconcile_jobs
from foxtrot.persistence.workspace import initialize_workspace
from foxtrot.pipeline.registry import PipelineSettings

from .locking import WorkerAlreadyRunning, default_runtime_directory, worker_lock
from .protocol import FixtureSettings
from .supervisor import Supervisor, SupervisorLimits


class Worker:
    """Run outside the API process; only trusted code selects the runner/settings."""

    def __init__(self, workspace_root: str | Path, source_checkout: str | Path, *,
                 fixture: FixtureSettings | None = None, pipeline: PipelineSettings | None = None,
                 stop: threading.Event | None = None,
                 limits: SupervisorLimits | None = None, runtime_directory: Path | None = None):
        if (fixture is None) == (pipeline is None):
            raise TypeError("Choose exactly one trusted runner: FixtureSettings or PipelineSettings")
        if fixture is not None and not isinstance(fixture, FixtureSettings):
            raise TypeError("fixture must be FixtureSettings")
        if pipeline is not None and not isinstance(pipeline, PipelineSettings):
            raise TypeError("pipeline must be PipelineSettings")
        self.workspace = Path(workspace_root)
        self.checkout = Path(source_checkout)
        self.runner = fixture if fixture is not None else pipeline
        self.stop = stop if stop is not None else threading.Event()
        self.limits = limits if limits is not None else SupervisorLimits()
        self.runtime_directory = runtime_directory or default_runtime_directory()

    def run(self, *, once: bool = False) -> int:
        """Hold leases for the whole loop, reconciling before new claims."""
        if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("The H5 supervisor requires Linux with pidfd support")
        processed = 0
        # Host lease comes first, also serializing first workspace initialization.
        with worker_lock(self.runtime_directory / "worker.lock"):
            root = initialize_workspace(self.workspace, self.checkout)
            with worker_lock(root / "settings" / "worker" / "worker.lock"):
                reconcile_jobs(root, self.checkout)
                supervisor = Supervisor(root, self.checkout, self.runner, self.stop, self.limits)
                while not self.stop.is_set():
                    try:
                        claim = claim_next_job(root, self.checkout)
                    except InvalidJobConfiguration as error:
                        transition_execution(root, self.checkout, error.job_id, ExecutionState.FAILED,
                                             error=ErrorDetails(code="invalid_configuration", message="Saved job configuration is missing, unsafe, or changed."))
                        processed += 1
                        if once:
                            break
                        continue
                    if claim is not None:
                        supervisor.run(claim)
                        processed += 1
                    if once:
                        break
                    if claim is None:
                        reconcile_jobs(root, self.checkout)
                        self.stop.wait(self.limits.poll_seconds)
        return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true", help="Explicitly run H5 infrastructure fixtures; no YOLO training")
    mode.add_argument("--pipeline-adapter", choices=("fixture",), help="Run a code-registered H6 pipeline adapter on registered datasets")
    parser.add_argument("--fixture-settings", type=Path, help="Trusted local fixture settings JSON; never job-supplied commands")
    parser.add_argument("--pipeline-settings", type=Path, help="Trusted local PipelineSettings JSON for --pipeline-adapter")
    parser.add_argument("--runtime-directory", type=Path, help="Private lease directory; use a separate path only for isolated tests")
    parser.add_argument("--once", action="store_true", help="Handle at most one queued job and exit")
    arguments = parser.parse_args()
    stop = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda *_: stop.set())
    if arguments.fixture_settings and not arguments.fixture:
        parser.error("--fixture-settings requires --fixture")
    if arguments.pipeline_settings and not arguments.pipeline_adapter:
        parser.error("--pipeline-settings requires --pipeline-adapter")
    try:
        if arguments.fixture:
            runner = {"fixture": (FixtureSettings.model_validate_json(arguments.fixture_settings.read_bytes())
                                  if arguments.fixture_settings else FixtureSettings())}
        else:
            pipeline = (PipelineSettings.model_validate_json(arguments.pipeline_settings.read_bytes())
                        if arguments.pipeline_settings else PipelineSettings(adapter=arguments.pipeline_adapter))
            if pipeline.adapter != arguments.pipeline_adapter:
                raise ValueError("Pipeline settings name a different adapter")
            runner = {"pipeline": pipeline}
        Worker(arguments.workspace, arguments.source_checkout, **runner, stop=stop,
               runtime_directory=arguments.runtime_directory).run(once=arguments.once)
    except WorkerAlreadyRunning:
        print("A Foxtrot worker is already running.", file=sys.stderr)
        return 2
    except Exception as error:
        # Do not echo arbitrary settings, environment values, or exception text.
        print(f"Worker stopped after {type(error).__name__}; inspect job records/local logs.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
