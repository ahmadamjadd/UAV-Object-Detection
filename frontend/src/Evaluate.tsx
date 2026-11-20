import { FormEvent, useEffect, useRef, useState } from "react";
import {
  api, Capabilities, CheckpointOption, Dataset, describeError, GenerationRun, PredictionResult,
} from "./api";
import { BoxGallery } from "./Generate";

const FINAL = new Set(["finished", "failed", "cancelled", "interrupted"]);

// Follow one background run (DVC download or prediction preview) until it ends.
export function useRun() {
  const [run, setRun] = useState<GenerationRun | null>(null);
  const timer = useRef<number | null>(null);
  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current); }, []);
  // Poll the run's status every 0.7 s while it is still going.
  function follow(started: GenerationRun) {
    setRun(started);
    api.generationRun(started.run_id).then((status) => {
      setRun(status);
      if (!FINAL.has(status.state)) timer.current = window.setTimeout(() => follow(status), 700);
    }, () => undefined);
  }
  return { run, follow, active: run !== null && !FINAL.has(run.state) };
}

// The Evaluate section: score an exact checkpoint on labeled real images, or preview it on unlabeled ones.
export function Evaluate({ capabilities, onSubmitted }: {
  capabilities: Capabilities | null; onSubmitted: (jobId: string) => void;
}) {
  const [checkpoints, setCheckpoints] = useState<CheckpointOption[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [error, setError] = useState("");

  // Load finished training jobs and every registered dataset.
  useEffect(() => {
    api.checkpoints().then(setCheckpoints, (e) => setError(describeError(e)));
    api.datasets().then(setDatasets, (e) => setError(describeError(e)));
  }, []);

  if (checkpoints.length === 0) {
    return <section className="card narrow"><h2>Evaluate</h2>
      <p className="muted">{error || "No finished training job yet: train a model first."}</p></section>;
  }
  return (
    <section className="grid">
      <EvaluationForm checkpoints={checkpoints} datasets={datasets} capabilities={capabilities} onSubmitted={onSubmitted} />
      <PredictionPreview checkpoints={checkpoints} />
    </section>
  );
}

// Score one checkpoint on one labeled validation/test dataset with the fixed evaluation settings.
function EvaluationForm({ checkpoints, datasets, capabilities, onSubmitted }: {
  checkpoints: CheckpointOption[]; datasets: Dataset[]; capabilities: Capabilities | null;
  onSubmitted: (jobId: string) => void;
}) {
  const [jobId, setJobId] = useState(checkpoints[0].job_id);
  const [datasetId, setDatasetId] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submissionKey = useRef<string | null>(null);
  const checkpoint = checkpoints.find((c) => c.job_id === jobId) ?? checkpoints[0];
  const classes = datasets.find((d) => d.dataset_id === checkpoint.dataset_id)?.class_names.join(",");
  const sets = datasets.filter((d) => d.role !== "training");
  const chosen = sets.find((d) => d.dataset_id === datasetId) ?? sets[0];

  // Submit an evaluation job; it appears in Activity like a training job.
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    submissionKey.current ??= crypto.randomUUID();
    setBusy(true);
    setError("");
    try {
      const job = await api.submit(submissionKey.current, {
        kind: "evaluate", dataset_id: chosen?.dataset_id, model_id: checkpoint.model_id,
        initialization: "checkpoint", checkpoint_job_id: checkpoint.job_id,
        tracking_destination_id: form.get("tracking_destination_id"),
      });
      submissionKey.current = null;
      onSubmitted(job.job_id);
    } catch (e) { setError(describeError(e)); } finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>Evaluate on labeled images</h2>
      <p className="muted">Scores an exact trained model on real labeled images it never trained on, with the fixed
        settings "foxtrot-eval-v1". Results, per-class scores and plots go to MLflow.</p>
      <form className="form" onSubmit={submit}>
        <label>Model (finished training job)<select value={jobId} onChange={(e) => setJobId(e.target.value)}>
          {checkpoints.map((c) => <option key={c.job_id} value={c.job_id}>
            {c.job_id.slice(0, 8)} · {c.model_id} · {c.dataset_id}{c.map50_95 !== null ? ` · mAP50-95 ${c.map50_95.toFixed(3)}` : ""}
          </option>)}
        </select></label>
        {sets.length === 0 ? <p className="error small">No validation or test dataset yet: register one on the Datasets page.</p> : (
          <label>Labeled evaluation dataset<select value={chosen?.dataset_id} onChange={(e) => setDatasetId(e.target.value)}>
            {sets.map((d) => <option key={d.dataset_id} value={d.dataset_id}>
              {d.dataset_id} · {d.role}{d.class_names.join(",") !== classes ? " · different classes" : ""}</option>)}
          </select></label>
        )}
        {chosen?.role === "test" && <p className="badge small">Final test data: use it for the last check, not to
          choose between settings.</p>}
        <label>Tracking destination<input name="tracking_destination_id" required
          defaultValue={capabilities?.tracking_destinations[0] ?? "local"} /></label>
        <button type="submit" disabled={busy || !chosen}>{busy ? "Submitting…" : "Start evaluation"}</button>
      </form>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

// Run a checkpoint on unlabeled photos to see its boxes. Clearly not an accuracy measurement.
function PredictionPreview({ checkpoints }: { checkpoints: CheckpointOption[] }) {
  const { run, follow, active } = useRun();
  const [error, setError] = useState("");

  // Start a preview run with the chosen checkpoint, folder and confidence.
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setError("");
    try {
      follow(await api.startPrediction({
        checkpoint_job_id: form.get("checkpoint_job_id"), images: String(form.get("images")).trim(),
        confidence: Number(form.get("confidence")),
      }));
    } catch (e) { setError(describeError(e)); }
  }

  const result = run?.state === "finished" ? run.result as PredictionResult : null;
  return (
    <div className="card">
      <h2>Preview on unlabeled photos</h2>
      <p className="muted">Shows what a model detects on photos without labels. Without labels nothing can be
        measured, so no score is given.</p>
      <form className="form" onSubmit={submit}>
        <label>Model<select name="checkpoint_job_id">
          {checkpoints.map((c) => <option key={c.job_id} value={c.job_id}>{c.job_id.slice(0, 8)} · {c.model_id} · {c.dataset_id}</option>)}
        </select></label>
        <label>Photos folder (absolute path)<input name="images" required placeholder="/home/you/flight-photos" /></label>
        <label>Minimum confidence<input name="confidence" type="number" step="0.01" min="0.01" max="0.99" defaultValue="0.25" /></label>
        <button type="submit" disabled={active}>{active ? "Running…" : "Preview"}</button>
      </form>
      {error && <p className="error">{error}</p>}
      {run && active && <p className="muted">{run.phase}{run.total ? ` · ${run.done} / ${run.total}` : ""}</p>}
      {run?.error && <p className="error">{run.error.message}</p>}
      {result && (
        <>
          <p className="badge">Preview only: not an accuracy measurement</p>
          <BoxGallery runId={run!.run_id} items={result.images.filter((item) => item.image).map((item) => ({
            image: item.image!, caption: `${item.source} · ${item.boxes.length} box(es)`, boxes: item.boxes }))} />
        </>
      )}
    </div>
  );
}
