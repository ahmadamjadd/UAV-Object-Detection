"""Where studies live: ``<workspace>/studies/<study-id>/``.

* ``study.json``  – the validated StudySpec and planned trial count (written once);
* ``state.json``  – running / stopping / stopped / finished, plus why;
* ``optuna.db``   – Optuna's own storage, written only by the one coordinator;
* ``trials/N.json`` – each trial: its sampled values, resolved settings, job, outcome;
* ``validation/`` – recipe studies: the frozen validation images every trial shares.

Small JSON files are replaced atomically (write a temporary file, then rename),
so a crash never leaves half a record.
"""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from foxtrot.config.study import StudySpec, planned_trials
from foxtrot.persistence.files import sync_directory, write_new_text
from foxtrot.persistence.workspace import initialize_workspace

STUDY_ID = re.compile(r"[0-9a-f]{32}")
ACTIVE_TRIAL_STATES = {"asked", "preparing_data", "running", "pruning"}
FINAL_TRIAL_STATES = {"complete", "failed", "pruned", "cancelled"}


# Raised for unknown, unsafe or unusable studies.
class StudyError(ValueError):
    pass


# Current UTC time for records.
def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# The studies folder of a workspace (created on first use, never a symbolic link).
def studies_root(workspace_root, source_checkout) -> Path:
    folder = initialize_workspace(workspace_root, source_checkout) / "studies"
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise StudyError("Workspace studies folder is unsafe")
    folder.mkdir(exist_ok=True)
    return folder


# One study's folder; malformed or unknown IDs raise KeyError (HTTP 404).
def study_folder(workspace_root, source_checkout, study_id: str) -> Path:
    folder = studies_root(workspace_root, source_checkout) / study_id
    if not STUDY_ID.fullmatch(study_id) or folder.is_symlink() or not folder.is_dir():
        raise KeyError("Unknown study")
    return folder


# Replace a small JSON file in one step.
def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


# Read a JSON file written by write_json / write_new_text.
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# Record a new study and mark it running; the coordinator starts its first trial.
def create_study(workspace_root, source_checkout, spec: StudySpec) -> str:
    from foxtrot.persistence.datasets import list_datasets

    if spec.dataset_id is not None:
        found = {m.dataset_id: m for m in list_datasets(workspace_root, source_checkout)}.get(spec.dataset_id)
        if found is None:
            raise StudyError(f"Unknown dataset {spec.dataset_id}")
        if found.role != "training":
            raise StudyError(f"{spec.dataset_id} is a {found.role} dataset; studies train on training datasets")
        if not found.can_be_tuning_objective():
            raise StudyError("Final test data can never be a tuning objective")
    study_id = uuid4().hex
    folder = studies_root(workspace_root, source_checkout) / study_id
    folder.mkdir()
    (folder / "trials").mkdir()
    write_new_text(folder / "study.json", json.dumps({
        "study_id": study_id, "created_at": now(), "planned_trials": planned_trials(spec),
        "spec": spec.model_dump(mode="json"),
    }, indent=2))
    write_json(folder / "state.json", {"state": "running", "reason": None, "stop_active": None,
                                       "time_used_seconds": 0.0, "resumed_at": now(), "finished_at": None})
    sync_directory(folder)
    return study_id


# The study's spec, record and current state.
def read_study(workspace_root, source_checkout, study_id: str) -> tuple[StudySpec, dict, dict]:
    folder = study_folder(workspace_root, source_checkout, study_id)
    record = read_json(folder / "study.json")
    return StudySpec.model_validate(record["spec"]), record, read_json(folder / "state.json")


# Update the study's state record.
def write_state(workspace_root, source_checkout, study_id: str, state: dict) -> None:
    write_json(study_folder(workspace_root, source_checkout, study_id) / "state.json", state)


# All trials of a study, in order.
def read_trials(workspace_root, source_checkout, study_id: str) -> list[dict]:
    folder = study_folder(workspace_root, source_checkout, study_id) / "trials"
    trials = [read_json(path) for path in folder.glob("*.json") if path.stem.isdigit()]
    return sorted(trials, key=lambda trial: trial["number"])


# Save one trial record.
def write_trial(workspace_root, source_checkout, study_id: str, trial: dict) -> None:
    folder = study_folder(workspace_root, source_checkout, study_id) / "trials"
    write_json(folder / f"{trial['number']}.json", trial)


# Every study ID, newest first.
def list_study_ids(workspace_root, source_checkout) -> list[str]:
    root = studies_root(workspace_root, source_checkout)
    found = [p for p in root.iterdir() if STUDY_ID.fullmatch(p.name) and p.is_dir() and not p.is_symlink()]
    return [p.name for p in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)]
