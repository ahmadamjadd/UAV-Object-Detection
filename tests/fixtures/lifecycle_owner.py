"""Controlled H4 test owner; never imports or launches training."""

import json
import subprocess
import sys

from foxtrot.config.lifecycle import ExecutionState
from foxtrot.persistence.events import append_job_event
from foxtrot.persistence.jobs import claim_next_job
from foxtrot.persistence.lifecycle import transition_execution
from foxtrot.persistence.recovery import begin_process_launch, attach_execution_process


workspace, checkout, mode = sys.argv[1:]
claim = claim_next_job(workspace, checkout)
transition_execution(workspace, checkout, claim.job_id, ExecutionState.TRAINING,
                     claim_token=claim.claim_token)
for epoch in range(3):
    append_job_event(workspace, checkout, claim.job_id, "progress", {"epoch": epoch},
                     claim_token=claim.claim_token, event_key=f"epoch:{epoch}")

child = None
try:
    if mode in {"pending", "attached"}:
        begin_process_launch(workspace, checkout, claim.job_id, claim.claim_token)
    if mode == "attached":
        # A dedicated session matches the future H5 execution contract. It
        # waits for EOF so a failed test controller cannot leave it sleeping.
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", "import sys; sys.stdin.buffer.read()"],
            stdin=sys.stdin, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(json.dumps({"child_pid": child.pid}), flush=True)
        attach_execution_process(workspace, checkout, claim.job_id, claim.claim_token, child.pid)
    print(json.dumps({"ready": True, "job_id": claim.job_id}), flush=True)
    if child is None:
        sys.stdin.buffer.read()
    else:
        child.wait()
finally:
    if child is not None and child.poll() is None:
        child.terminate()
        child.wait(timeout=5)
