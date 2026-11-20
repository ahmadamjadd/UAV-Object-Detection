"""The real MLflow implementation of Foxtrot's TrackingClient.

Used only inside the sync process (foxtrot.tracking.sync), never by the website,
because MLflow reads credentials from process-wide environment variables.
Foxtrot creates and names every run itself and stores the run ID; it never
relies on MLflow guessing the "active" run.
"""

import os
from pathlib import Path
import time
from typing import Sequence

from foxtrot.config.tracking import MetricPoint, TerminalStatus

from .contracts import RunHandle
from .plan import JOB_TAG

PARAM_BATCH = 100       # MLflow accepts at most 100 parameters per request
METRIC_BATCH = 1000     # and at most 1000 metric points per request
PARAM_VALUE_LIMIT = 6000


# Set MLflow's environment for one destination: its credentials, no telemetry, no hints.
# Must run before MLflow is imported or used in this process.
def configure_environment(username: str | None, token: str | None, *, quick_check: bool = False) -> None:
    for name, value in (("MLFLOW_TRACKING_USERNAME", username), ("MLFLOW_TRACKING_PASSWORD", token)):
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    if quick_check:  # a connection test should answer fast instead of retrying for minutes
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "15"


# Where a local MLflow database keeps its files (next to the database), or None for servers.
def _local_artifact_root(tracking_uri: str) -> str | None:
    if tracking_uri.startswith("sqlite:///"):
        return (Path(tracking_uri.removeprefix("sqlite:///")).parent / "mlartifacts").as_uri()
    return None


# Talks to one MLflow experiment on one tracking server (or local database).
class MlflowTrackingClient:
    # Connect, and find (or create, or restore) the experiment that runs go into.
    def __init__(self, tracking_uri: str, experiment_name: str, browser_url: str | None = None):
        from mlflow.tracking import MlflowClient
        self.client = MlflowClient(tracking_uri=tracking_uri)
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(
                experiment_name, artifact_location=_local_artifact_root(tracking_uri))
        else:
            if experiment.lifecycle_stage == "deleted":
                self.client.restore_experiment(experiment.experiment_id)
            self.experiment_id = experiment.experiment_id
        # The page people open in a browser; a hosted tracking URI usually serves the UI too.
        if browser_url is None and tracking_uri.startswith(("http://", "https://")):
            browser_url = tracking_uri
        self.browser_url = browser_url.rstrip("/") if browser_url else None

    # The browser link to one run, when there is a browser-facing MLflow page.
    def _url(self, run_id: str) -> str | None:
        if not self.browser_url:
            return None
        return f"{self.browser_url}/#/experiments/{self.experiment_id}/runs/{run_id}"

    # Find a run already created for this job (e.g. if an earlier attempt was cut off).
    def find_run(self, job_id: str) -> RunHandle | None:
        runs = self.client.search_runs([self.experiment_id], filter_string=f"tags.`{JOB_TAG}` = '{job_id}'",
                                       max_results=1)
        return RunHandle(runs[0].info.run_id, self._url(runs[0].info.run_id)) if runs else None

    # Create a new run with Foxtrot's name and tags.
    def create_run(self, job_id: str, run_name: str, tags: dict[str, str]) -> RunHandle:
        run = self.client.create_run(self.experiment_id, tags=dict(tags), run_name=run_name)
        return RunHandle(run.info.run_id, self._url(run.info.run_id))

    # Log settings as parameters, in batches MLflow accepts.
    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        from mlflow.entities import Param
        items = [Param(key, str(value)[:PARAM_VALUE_LIMIT]) for key, value in params.items()]
        for start in range(0, len(items), PARAM_BATCH):
            self.client.log_batch(run_id, params=items[start:start + PARAM_BATCH])

    # Log metric points (each with its epoch as the step), in batches MLflow accepts.
    def log_metrics(self, run_id: str, metrics: Sequence[MetricPoint]) -> None:
        from mlflow.entities import Metric
        now = int(time.time() * 1000)
        items = [Metric(point.key, point.value, now, point.step) for point in metrics]
        for start in range(0, len(items), METRIC_BATCH):
            self.client.log_batch(run_id, metrics=items[start:start + METRIC_BATCH])

    # Upload one file into a folder of the run's artifacts.
    def log_artifact(self, run_id: str, local_path: Path, artifact_path: str) -> None:
        self.client.log_artifact(run_id, str(local_path), artifact_path)

    # Mark the run finished, failed or killed.
    def set_terminated(self, run_id: str, status: TerminalStatus) -> None:
        self.client.set_terminated(run_id, status)
