"""In-memory tracking service for tests and fixtures; contacts no network.

It mimics MLflow behaviour Foxtrot must tolerate: parameters cannot change
once logged, and calls can fail transiently (``fail_once``).
"""

import hashlib
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from foxtrot.config.tracking import MetricPoint

from .contracts import RunHandle
from .plan import JOB_TAG


class RecordingTrackingClient:
    def __init__(self, *, fail_once: set[str] | None = None, base_url: str = "http://tracking.invalid"):
        self.fail_once = set(fail_once or ())
        self.base_url = base_url
        self.runs: dict[str, dict] = {}
        self.calls: list[str] = []

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_once:
            self.fail_once.discard(name)
            raise ConnectionError("fixture tracking outage")

    def find_run(self, job_id: str) -> RunHandle | None:
        self._call("find_run")
        for run_id, run in self.runs.items():
            if run["tags"].get(JOB_TAG) == job_id:
                return RunHandle(run_id, f"{self.base_url}/runs/{run_id}")
        return None

    def create_run(self, job_id: str, run_name: str, tags: dict[str, str]) -> RunHandle:
        self._call("create_run")
        run_id = uuid4().hex
        self.runs[run_id] = {"name": run_name, "tags": dict(tags), "params": {},
                             "metrics": [], "artifacts": [], "status": "RUNNING"}
        return RunHandle(run_id, f"{self.base_url}/runs/{run_id}")

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        self._call("log_params")
        saved = self.runs[run_id]["params"]
        for key, value in params.items():
            if key in saved and saved[key] != value:
                raise ValueError("Logged parameters are immutable")
            saved[key] = value

    def log_metrics(self, run_id: str, metrics: Sequence[MetricPoint]) -> None:
        self._call("log_metrics")
        self.runs[run_id]["metrics"].extend((m.key, m.value, m.step) for m in metrics)

    def log_artifact(self, run_id: str, local_path: Path, artifact_path: str) -> None:
        self._call("log_artifact")
        digest = hashlib.sha256(Path(local_path).read_bytes()).hexdigest()
        self.runs[run_id]["artifacts"].append((artifact_path, Path(local_path).name, digest))

    def set_terminated(self, run_id: str, status: str) -> None:
        self._call("set_terminated")
        self.runs[run_id]["status"] = status
