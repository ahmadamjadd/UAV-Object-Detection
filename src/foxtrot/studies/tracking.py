"""Studies in MLflow: one parent run per study, each trial's job run as its child.

Runs inside the separate upload process (it has the credentials). A trial's run
is uploaded only once the coordinator has recorded the trial's outcome, so its
``foxtrot.trial_state`` tag (complete / failed / pruned / cancelled) and reason
are final. When the study ends, the parent gets the best scores, outcome
counts, a summary file, and FINISHED (or KILLED if the member stopped it).
"""

import json

from foxtrot.config.tracking import MetricPoint
from foxtrot.tracking.plan import JOB_TAG

from .store import FINAL_TRIAL_STATES, list_study_ids, read_json, read_study, read_trials, study_folder, write_json
from .summary import summarize


# The study and trial of a job's run tags, if the job belongs to a study.
def _study_of(tags: dict) -> tuple[str, int] | None:
    study_id = tags.get("foxtrot.study_id")
    return (study_id, int(tags["foxtrot.trial_number"])) if study_id else None


# True when a study job may be uploaded: its trial outcome is already recorded.
def trial_settled(workspace, checkout, study_id: str, number: int) -> bool:
    trials = {t["number"]: t for t in read_trials(workspace, checkout, study_id)}
    return trials.get(number, {}).get("state") in FINAL_TRIAL_STATES


# Find or create the study's parent run in this destination, logging the study definition once.
def ensure_parent(workspace, checkout, client, destination_id: str, study_id: str) -> str:
    folder = study_folder(workspace, checkout, study_id)
    path = folder / "mlflow.json"
    records = read_json(path) if path.is_file() else {}
    entry = records.get(destination_id)
    if entry is None:
        spec, record, _ = read_study(workspace, checkout, study_id)
        key = f"study-{study_id}"
        handle = client.find_run(key) or client.create_run(key, f"study-{spec.name}", {
            JOB_TAG: key, "foxtrot.kind": "study", "foxtrot.study_id": study_id})
        definition = spec.model_dump(mode="json")
        params = {f"study.{k}": json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                  for k, v in definition.items() if v not in (None, [], {})}
        params["study.planned_trials"] = str(record["planned_trials"])
        client.log_params(handle.run_id, params)
        entry = {"run_id": handle.run_id, "url": handle.url, "finalized": False}
        records[destination_id] = entry
        write_json(path, records)
    return entry["run_id"]


# Extra tags for a trial's run: link to the parent, and the trial's recorded outcome.
def trial_run_tags(workspace, checkout, client, destination_id: str, tags: dict) -> dict:
    found = _study_of(tags)
    if found is None:
        return {}
    study_id, number = found
    trial = next(t for t in read_trials(workspace, checkout, study_id) if t["number"] == number)
    return {"mlflow.parentRunId": ensure_parent(workspace, checkout, client, destination_id, study_id),
            "foxtrot.trial_state": trial["state"], "foxtrot.trial_reason": (trial.get("error") or "")[:250]}


# Close the parent run of every ended study whose trials are all uploaded.
def finalize_studies(workspace, checkout, client_for) -> list[dict]:
    from foxtrot.persistence.jobs import get_job_record

    results = []
    for study_id in list_study_ids(workspace, checkout):
        spec, record, state = read_study(workspace, checkout, study_id)
        if state["state"] not in ("finished", "stopped"):
            continue
        folder = study_folder(workspace, checkout, study_id)
        path = folder / "mlflow.json"
        records = read_json(path) if path.is_file() else {}
        if records.get(spec.tracking_destination_id, {}).get("finalized"):
            continue
        trials = read_trials(workspace, checkout, study_id)
        jobs = [t["job_id"] for t in trials if t.get("job_id")]
        if any(get_job_record(workspace, checkout, job).tracking_state != "synced" for job in jobs):
            continue  # wait until every trial run is uploaded
        try:
            client = client_for(spec.tracking_destination_id)
            run_id = ensure_parent(workspace, checkout, client, spec.tracking_destination_id, study_id)
            summary = summarize(spec, record, state, trials)
            write_json(folder / "summary.json", summary)
            metrics = [MetricPoint(key=f"study/{name}", value=float(count), step=0)
                       for name, count in summary["counts"].items()]
            if summary["best"]:
                metrics.append(MetricPoint(key=f"best/{spec.objective}", value=summary["best"]["value"], step=0))
                client.log_params(run_id, {"best.trial": str(summary["best"]["number"]),
                                           "best.job_id": summary["best"]["job_id"],
                                           "best.model_id": summary["best"]["model_id"],
                                           "best.checkpoint_sha256": str(summary["best"]["checkpoint_sha256"])})
            for model, best in summary["best_per_model"].items():
                if best:
                    metrics.append(MetricPoint(key=f"best/{model}/{spec.objective}", value=best["value"], step=0))
            client.log_metrics(run_id, metrics)
            client.log_artifact(run_id, folder / "summary.json", "study")
            client.set_terminated(run_id, "FINISHED" if state["state"] == "finished" else "KILLED")
            records = read_json(path)
            records[spec.tracking_destination_id]["finalized"] = True
            write_json(path, records)
            results.append({"study_id": study_id, "finalized": True})
        except Exception as error:  # retried on the next upload pass
            results.append({"study_id": study_id, "finalized": False, "error": type(error).__name__})
    return results


# True if any ended study's parent run is not closed yet (cheap file check for the upload scheduler).
def studies_awaiting_finalize(workspace, checkout) -> bool:
    for study_id in list_study_ids(workspace, checkout):
        spec, _, state = read_study(workspace, checkout, study_id)
        if state["state"] in ("finished", "stopped"):
            path = study_folder(workspace, checkout, study_id) / "mlflow.json"
            if not (read_json(path) if path.is_file() else {}).get(spec.tracking_destination_id, {}).get("finalized"):
                return True
    return False
