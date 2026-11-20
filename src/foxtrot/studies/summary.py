"""What a study found: best trial overall and per model, outcomes, time used, repeat statistics."""

from datetime import datetime
import statistics

from foxtrot.config.study import StudySpec

from .store import FINAL_TRIAL_STATES


# Seconds between two ISO timestamps (None if either is missing).
def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


# The trial with the highest objective value among complete trials.
def _best(trials: list[dict]) -> dict | None:
    complete = [t for t in trials if t["state"] == "complete" and t["value"] is not None]
    if not complete:
        return None
    best = max(complete, key=lambda t: (t["value"], -t["number"]))  # ties: the earlier trial
    return {"number": best["number"], "value": best["value"], "model_id": best["model_id"], "job_id": best["job_id"],
            "checkpoint_sha256": best.get("checkpoint_sha256"), "params": best["params"]}


# A plain summary for the website and the MLflow study run.
def summarize(spec: StudySpec, record: dict, state: dict, trials: list[dict]) -> dict:
    counts = {name: sum(1 for t in trials if t["state"] == name)
              for name in ("complete", "failed", "pruned", "cancelled")}
    counts["active"] = sum(1 for t in trials if t["state"] not in FINAL_TRIAL_STATES)
    durations = [d for t in trials if (d := _seconds(t.get("started_at"), t.get("finished_at"))) is not None]
    summary = {
        "objective": spec.objective, "direction": "maximize", "sampler": spec.sampler,
        "planned_trials": record["planned_trials"], "trials_started": len(trials), "counts": counts,
        "best": _best(trials),
        "best_per_model": {model: _best([t for t in trials if t["model_id"] == model]) for model in spec.model_ids},
        "trial_seconds_total": round(sum(durations), 1),
        "trial_seconds_mean": round(statistics.fmean(durations), 1) if durations else None,
        "comparison": ("equal budget per model" if spec.model_budget == "equal" else "combined model search")
        if len(spec.model_ids) > 1 else "single model",
        "state": state["state"], "reason": state.get("reason"),
    }
    if spec.sampler == "repeat":  # group the repeats of each configuration across seeds
        groups: dict[str, list[dict]] = {}
        for trial in trials:
            item = spec.repeats[trial["number"]]
            key = f"{item.model_id} {sorted(item.training.items())}"
            groups.setdefault(key, []).append(trial)
        summary["repeat_groups"] = [{
            "configuration": key, "source_trial": group[0].get("source_trial"),
            "values": [{"seed": t["params"]["seed"], "value": t["value"], "state": t["state"]} for t in group],
            "mean": round(statistics.fmean(v), 6) if (v := [t["value"] for t in group if t["value"] is not None]) else None,
            "stdev": round(statistics.stdev(v), 6) if len(v) > 1 else None,
        } for key, group in groups.items()]
    return summary
