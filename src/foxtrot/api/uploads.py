"""Automatic uploads while Foxtrot runs.

A background thread in the launcher checks for jobs with unsent results and runs
the sync command (a separate process, so MLflow and credentials never load into
the website). When uploads fail it waits longer between tries, up to 10 minutes;
the Retry button (or a successful upload) brings it back to the normal pace.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import threading

from foxtrot.persistence.tracking import jobs_with_pending_tracking


# Runs uploads in the background, one at a time, with growing waits after failures.
class UploadScheduler:
    # interval: normal wait between checks; max_interval: longest wait after repeated failures.
    def __init__(self, workspace: Path, checkout: Path, *, interval: float = 20.0,
                 max_interval: float = 600.0, command_timeout: float = 1800.0):
        self.workspace, self.checkout = Path(workspace), Path(checkout)
        self.interval, self.max_interval, self.command_timeout = interval, max_interval, command_timeout
        self.delay = interval
        self.last_run: str | None = None
        self.last_results: list | dict | None = None
        self._wake, self._stop = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    # Start checking in the background (called by the launcher).
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="foxtrot-uploads", daemon=True)
            self._thread.start()

    # Stop the background thread; an upload in progress is stopped and resumes next time.
    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout)

    # Upload as soon as possible (the Retry button): resets the wait to normal.
    def trigger(self) -> None:
        self.delay = self.interval
        if self._thread is not None:
            self._wake.set()
        else:  # not started (e.g. website-only mode): do one upload pass in the background
            threading.Thread(target=self.run_once, name="foxtrot-upload-once", daemon=True).start()

    # The background loop: wait, and if anything is pending, upload it.
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.delay)
            self._wake.clear()
            if self._stop.is_set():
                break
            if not jobs_with_pending_tracking(self.workspace, self.checkout) and not self._studies_waiting():
                self.delay = self.interval
                continue
            succeeded = self.run_once()
            self.delay = self.interval if succeeded else min(self.delay * 2, self.max_interval)

    # One upload pass in a separate process; True when every pending job reached MLflow.
    def run_once(self) -> bool:
        with self._lock:
            command = [sys.executable, "-B", "-m", "foxtrot.tracking.sync", "--workspace", str(self.workspace),
                       "--source-checkout", str(self.checkout), "run"]
            try:
                self._process = subprocess.Popen(command, cwd=self.workspace, stdin=subprocess.DEVNULL,
                                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, _ = self._process.communicate(timeout=self.command_timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.communicate()
                stdout = ""
            finally:
                self._process = None
            self.last_run = datetime.now(UTC).isoformat(timespec="seconds")
            try:
                self.last_results = json.loads(stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                self.last_results = {"error": "The upload process did not finish normally."}
                return False
            if isinstance(self.last_results, dict):  # {"busy": true}: someone else is uploading
                return True
            # A deferred study trial is not a failure: it uploads once its outcome is recorded.
            return all(result.get("tracking_state") == "synced" or result.get("deferred")
                       for result in self.last_results)

    # True when an ended study still needs its MLflow parent run closed (V5).
    def _studies_waiting(self) -> bool:
        from foxtrot.studies.tracking import studies_awaiting_finalize
        try:
            return studies_awaiting_finalize(self.workspace, self.checkout)
        except (OSError, ValueError):
            return False

    # A short summary for the website.
    def status(self) -> dict:
        return {"running": self._thread is not None and not self._stop.is_set(),
                "next_check_seconds": self.delay, "last_run": self.last_run, "last_results": self.last_results}
