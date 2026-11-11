"""Start Foxtrot: the loopback API/website plus a separate worker process.

    python -m foxtrot.api.launcher --workspace /abs/ws --source-checkout /abs/MlOps --pipeline-adapter fixture

Closing the browser changes nothing. Ctrl+C (or SIGTERM) stops the website,
then asks the worker to stop and waits for its bounded cleanup, so training
children are never silently orphaned. Queued jobs stay queued.
"""

import argparse
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import webbrowser

import uvicorn

from foxtrot.execution.supervisor import SupervisorLimits
from foxtrot.persistence.database import initialize_database

from .app import create_app
from .security import new_session_token
from .settings import RedactingFilter

HOST = "127.0.0.1"  # loopback only; deliberately not configurable
WORKER_ALREADY_RUNNING = 2


def worker_stop_seconds(limits: SupervisorLimits = SupervisorLimits()) -> float:
    """Graceful + TERM + KILL deadlines, plus margin for finalization."""
    return limits.grace_seconds + limits.terminate_seconds + limits.kill_seconds + 10


def start_worker(arguments) -> subprocess.Popen:
    command = [sys.executable, "-B", "-m", "foxtrot.execution.worker",
               "--workspace", str(arguments.workspace), "--source-checkout", str(arguments.source_checkout)]
    command += (["--pipeline-adapter", arguments.pipeline_adapter] if arguments.pipeline_adapter
                else ["--fixture"])
    if arguments.runtime_directory:
        command += ["--runtime-directory", str(arguments.runtime_directory)]
    # Own session: a terminal Ctrl+C reaches the launcher, which then stops the
    # worker deliberately and waits for it, rather than racing it.
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)


def stop_worker(worker: subprocess.Popen) -> int | None:
    if worker.poll() is None:
        worker.send_signal(signal.SIGTERM)
        try:
            worker.wait(timeout=worker_stop_seconds())
        except subprocess.TimeoutExpired:
            print("Worker did not stop in time; inspect job records before restarting.", file=sys.stderr)
            return None
    return worker.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pipeline-adapter", choices=("fixture",),
                        help="Worker runs this adapter; omit to run H5 infrastructure fixtures")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument("--no-worker", action="store_true", help="Serve the website/API only")
    parser.add_argument("--runtime-directory", type=Path, help=argparse.SUPPRESS)  # isolated tests only
    arguments = parser.parse_args(argv)
    if not 1024 <= arguments.port <= 65535:
        parser.error("--port must be between 1024 and 65535")

    # Initialize storage before the API and worker can race to create it.
    initialize_database(arguments.workspace, arguments.source_checkout)
    app = create_app(arguments.workspace, arguments.source_checkout, port=arguments.port,
                     session_token=new_session_token(), pipeline_adapter=arguments.pipeline_adapter)
    redactor = RedactingFilter(app.state.settings)
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).addFilter(redactor)

    worker = None if arguments.no_worker else start_worker(arguments)
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=arguments.port, log_level="warning",
                                           access_log=False, lifespan="off"))
    url = f"http://{HOST}:{arguments.port}/"

    def announce():
        while not server.started and not server.should_exit:
            time.sleep(0.05)
        if not server.started:
            return
        print(f"Foxtrot is running at {url} (Ctrl+C to stop)", flush=True)
        if worker is not None:
            time.sleep(0.5)
            if worker.poll() == WORKER_ALREADY_RUNNING:
                print("Another Foxtrot worker is already running; it will process queued jobs.", flush=True)
        if not arguments.no_browser:
            webbrowser.open(url)

    threading.Thread(target=announce, daemon=True).start()
    try:
        # Handles SIGINT/SIGTERM by shutting the website down gracefully. uvicorn
        # then re-raises Ctrl+C, and exits by itself if it cannot bind the port.
        server.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        code = stop_worker(worker) if worker is not None else 0
    if not server.started:
        print(f"Could not serve on {url}; is the port already in use?", file=sys.stderr)
        return 1
    return 0 if code in (0, WORKER_ALREADY_RUNNING) else 1


if __name__ == "__main__":
    raise SystemExit(main())
