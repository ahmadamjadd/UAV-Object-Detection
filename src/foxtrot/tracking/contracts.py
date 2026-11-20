"""The operations Foxtrot needs from a tracking service such as MLflow.

A real MLflow client implements this in V1. Foxtrot owns run identity: it
stores the run ID it created instead of using ``mlflow.last_active_run()``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from foxtrot.config.tracking import MetricPoint, TerminalStatus


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    url: str | None  # browser-facing link, separate from the tracking API address


class TrackingClient(Protocol):
    def find_run(self, job_id: str) -> RunHandle | None:
        """Find a run already tagged with this job, after an ambiguous create."""
        ...

    def create_run(self, job_id: str, run_name: str, tags: dict[str, str]) -> RunHandle: ...

    def log_params(self, run_id: str, params: dict[str, str]) -> None: ...

    def log_metrics(self, run_id: str, metrics: Sequence[MetricPoint]) -> None: ...

    def log_artifact(self, run_id: str, local_path: Path, artifact_path: str) -> None: ...

    def set_terminated(self, run_id: str, status: TerminalStatus) -> None: ...
