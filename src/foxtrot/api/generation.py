"""Start, watch and cancel background runs for the website (V2 generation; V4 DVC and previews).

Each run gets a folder ``generation/runs/<run-id>/`` holding its request,
status, log and (for previews) sample images. The run itself is a separate
process (``foxtrot.pipeline.generation_runner``); one run at a time.
"""

import json
from pathlib import Path
import re
import subprocess
import sys
import threading
from uuid import uuid4

from foxtrot.persistence.dataset_generation import generation_root
from foxtrot.persistence.files import write_new_text
from foxtrot.persistence.workspace import initialize_workspace
from foxtrot.pipeline.generation_runner import CANCEL, FINAL_STATES, PREVIEW, REQUEST, STATUS, RunRequest, now

RUN_ID = re.compile(r"[0-9a-f]{32}")
PREVIEW_IMAGE = re.compile(r"[0-9]{1,7}\.jpg")


# Raised when a run cannot start because another one is still going.
class GenerationBusy(RuntimeError):
    pass


# Keeps track of generation runs and the one process currently running.
class GenerationRuns:
    def __init__(self, workspace: Path, checkout: Path):
        self.workspace, self.checkout = Path(workspace), Path(checkout)
        self._lock = threading.Lock()
        self._active: tuple[str, subprocess.Popen] | None = None

    # The folder that holds every run's subfolder.
    def _runs(self) -> Path:
        folder = generation_root(initialize_workspace(self.workspace, self.checkout)) / "runs"
        folder.mkdir(exist_ok=True)
        return folder

    # The folder of one run; unknown or malformed IDs raise KeyError (HTTP 404).
    def _folder(self, run_id: str) -> Path:
        folder = self._runs() / run_id
        if not RUN_ID.fullmatch(run_id) or folder.is_symlink() or not folder.is_dir():
            raise KeyError("Unknown generation run")
        return folder

    # True while a run's process is still alive.
    def _running(self) -> bool:
        return self._active is not None and self._active[1].poll() is None

    # Start a run in its own process; returns its first status. `request` matches the kind:
    # GenerationRequest (preview/dataset), DvcImportRequest (dvc) or PredictionRequest (predict).
    def start(self, kind: str, request) -> dict:
        with self._lock:
            if self._running():
                raise GenerationBusy("Another generation is still running; wait for it or cancel it")
            run_id = uuid4().hex
            folder = self._runs() / run_id
            folder.mkdir()
            field = {"preview": "request", "dataset": "request", "dvc": "dvc", "predict": "predict"}[kind]
            write_new_text(folder / REQUEST, RunRequest(kind=kind, **{field: request}).model_dump_json(indent=2))
            status = {"run_id": run_id, "kind": kind, "state": "queued", "phase": "queued", "done": 0, "total": 0,
                      "created_at": now(), "error": None, "result": None}
            write_new_text(folder / STATUS, json.dumps(status, indent=2))
            command = [sys.executable, "-B", "-m", "foxtrot.pipeline.generation_runner",
                       "--workspace", str(self.workspace), "--source-checkout", str(self.checkout),
                       "--run", str(folder)]
            with (folder / "log.txt").open("w") as log:
                process = subprocess.Popen(command, cwd=folder, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            self._active = (run_id, process)
            return status

    # The run's current status; a run whose process vanished is reported as interrupted.
    def status(self, run_id: str) -> dict:
        folder = self._folder(run_id)
        status = json.loads((folder / STATUS).read_text(encoding="utf-8"))
        alive = self._active is not None and self._active[0] == run_id and self._active[1].poll() is None
        if status.get("state") not in FINAL_STATES and not alive:
            if self._active is not None and self._active[0] == run_id:
                self._active[1].poll()
            # Re-read once: the process may have written its final status just now.
            status = json.loads((folder / STATUS).read_text(encoding="utf-8"))
            if status.get("state") not in FINAL_STATES:
                status.update({"state": "interrupted",
                               "error": {"code": "stopped", "message": "The generation process stopped unexpectedly"}})
        return status

    # Ask a run to stop; it notices within one image and cleans up its temporary files.
    def cancel(self, run_id: str) -> dict:
        folder = self._folder(run_id)
        (folder / CANCEL).touch()
        return self.status(run_id)

    # Path of one preview image, only for well-formed names inside the run's preview folder.
    def preview_image(self, run_id: str, name: str) -> Path:
        if not PREVIEW_IMAGE.fullmatch(name):
            raise KeyError("Unknown preview image")
        path = self._folder(run_id) / PREVIEW / "images" / name
        if path.is_symlink() or not path.is_file():
            raise KeyError("Unknown preview image")
        return path

    # On shutdown: ask the active run to cancel, then stop it if it does not finish quickly.
    def stop(self, timeout: float = 10.0) -> None:
        with self._lock:
            if not self._running():
                return
            run_id, process = self._active
            (self._runs() / run_id / CANCEL).touch()
            try:
                process.wait(timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
