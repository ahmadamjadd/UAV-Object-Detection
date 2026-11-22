import { useEffect, useRef, useState } from "react";
import { useShared } from "../App";
import { api, CheckpointOption, Dataset, describeError, PredictionResult } from "../api";
import { Banner, BoxGallery, Empty, ErrorText, go, modelName, optionLabel, Progress, score, Tabs } from "../ui";
import { useRun } from "../ui";

// Evaluate: how good is a trained model on real images?
export function EvaluatePage({ params }: { params: URLSearchParams }) {
  const tab = params.get("tab") === "preview" ? "preview" : "score";
  const [checkpoints, setCheckpoints] = useState<CheckpointOption[] | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  useEffect(() => {
    api.checkpoints().then(setCheckpoints, () => setCheckpoints([]));
    api.datasets().then(setDatasets, () => undefined);
  }, []);
  return (
    <>
      <h2>Evaluate</h2>
      <p className="sub">How good is a trained model on real images?</p>
      <Tabs value={tab} onChange={(t) => go("evaluate", t === "preview" ? { tab: "preview" } : {})}
        options={[["score", "Score on labeled images"], ["preview", "Preview on unlabeled photos"]]} />
      {checkpoints && checkpoints.length === 0 ? (
        <div className="card"><Empty title="No trained model yet">Finish a training run first; its best checkpoint appears here.
          <div className="row" style={{ justifyContent: "center", marginTop: 12 }}><button className="btn acc" onClick={() => go("train")}>Train a model</button></div></Empty></div>
      ) : !checkpoints ? <p className="mute">Loading…</p>
        : tab === "score" ? <ScoreForm checkpoints={checkpoints} datasets={datasets} initialDataset={params.get("dataset")} />
          : <PreviewForm checkpoints={checkpoints} />}
    </>
  );
}

// Score one exact checkpoint on one labeled validation/test dataset (an evaluation job).
function ScoreForm({ checkpoints, datasets, initialDataset }: { checkpoints: CheckpointOption[]; datasets: Dataset[]; initialDataset: string | null }) {
  const { capabilities } = useShared();
  const [jobId, setJobId] = useState(checkpoints[0].job_id);
  const [datasetId, setDatasetId] = useState(initialDataset ?? "");
  const [destination, setDestination] = useState(capabilities?.tracking_destinations[0] ?? "local");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submissionKey = useRef<string | null>(null);
  const checkpoint = checkpoints.find((c) => c.job_id === jobId) ?? checkpoints[0];
  const trainedClasses = datasets.find((d) => d.dataset_id === checkpoint.dataset_id)?.class_names.join(",");
  // Validation sets first: final test data is for the last check, so it is never the default.
  const sets = datasets.filter((d) => d.role !== "training").sort((a, b) => (a.role === "test" ? 1 : 0) - (b.role === "test" ? 1 : 0));
  const chosen = sets.find((d) => d.dataset_id === datasetId) ?? sets[0];
  const differentClasses = chosen && trainedClasses !== undefined && chosen.class_names.join(",") !== trainedClasses;

  // Submit the evaluation job and follow it in Runs.
  async function submit() {
    submissionKey.current ??= crypto.randomUUID();
    setBusy(true); setError("");
    try {
      const job = await api.submit(submissionKey.current, {
        kind: "evaluate", dataset_id: chosen?.dataset_id, model_id: checkpoint.model_id, initialization: "checkpoint",
        checkpoint_job_id: checkpoint.job_id, tracking_destination_id: destination,
      }, `Evaluation of ${checkpoint.name ?? checkpoint.job_id.slice(0, 8)} on ${chosen?.name ?? chosen?.dataset_id}`.slice(0, 80));
      submissionKey.current = null;
      go("runs", { job: job.job_id });
    } catch (e) { setError(describeError(e)); } finally { setBusy(false); }
  }

  return (
    <div className="card" style={{ maxWidth: 720 }}>
      <label className="f">Trained model (finished job)<select value={jobId} onChange={(e) => setJobId(e.target.value)}>
        {checkpoints.map((c) => <option key={c.job_id} value={c.job_id}>{c.name ?? c.job_id.slice(0, 8)} · {modelName(c.model_id, capabilities?.model_details)} · {c.dataset_id}{c.map50_95 !== null ? ` · mAP50-95 ${score(c.map50_95)}` : ""}</option>)}
      </select></label>
      {sets.length === 0 ? <Banner kind="info">No validation or test dataset yet. Import real labeled photos in Datasets › Add dataset › Import a folder.
        <button className="btn sm" style={{ marginLeft: 8 }} onClick={() => go("datasets", { add: "import" })}>Add dataset</button></Banner> : (
        <label className="f">Dataset to score on<select value={chosen?.dataset_id} onChange={(e) => setDatasetId(e.target.value)}>
          {sets.map((d) => <option key={d.dataset_id} value={d.dataset_id}>{optionLabel(d.name, d.dataset_id)} ({d.role === "test" ? "test · protected" : d.role})
            {trainedClasses !== undefined && d.class_names.join(",") !== trainedClasses ? " — different classes" : ""}</option>)}
        </select></label>)}
      {chosen?.role === "test" && <Banner kind="warn">Final test data: use it for the last check, not to choose between settings.</Banner>}
      {differentClasses && <Banner kind="bad">This dataset's classes ({chosen.class_names.join(", ")}) differ from the model's. Evaluation needs the same class names in the same order.</Banner>}
      <label className="f">Tracking destination<select value={destination} onChange={(e) => setDestination(e.target.value)}>
        {[...new Set([destination, ...(capabilities?.tracking_destinations ?? [])])].filter(Boolean).map((d) => <option key={d}>{d}</option>)}</select></label>
      <p className="mute small">Uses the fixed evaluation settings <span className="mono">foxtrot-eval-v1</span>, so scores are comparable between models. Plots and per-class scores go to MLflow.</p>
      <button className="btn acc" onClick={submit} disabled={busy || !chosen || differentClasses}>{busy ? "Starting…" : "Start evaluation"}</button>
      <ErrorText message={error} />
    </div>
  );
}

// Run a checkpoint on unlabeled photos and show its boxes — never as a score.
function PreviewForm({ checkpoints }: { checkpoints: CheckpointOption[] }) {
  const { capabilities } = useShared();
  const { run, follow, active } = useRun();
  const [jobId, setJobId] = useState(checkpoints[0].job_id);
  const [folder, setFolder] = useState("");
  const [confidence, setConfidence] = useState(0.25);
  const [error, setError] = useState("");

  // Start the preview run.
  async function start() {
    setError("");
    try { follow(await api.startPrediction({ checkpoint_job_id: jobId, images: folder.trim(), confidence })); }
    catch (e) { setError(describeError(e)); }
  }

  const result = run?.state === "finished" ? run.result as PredictionResult : null;
  return (
    <>
      <Banner kind="warn"><b>Preview only — not an accuracy measurement.</b> These photos have no labels, so nothing is scored.</Banner>
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="fg">
          <label className="f">Model<select value={jobId} onChange={(e) => setJobId(e.target.value)}>
            {checkpoints.map((c) => <option key={c.job_id} value={c.job_id}>{c.name ?? c.job_id.slice(0, 8)} · {modelName(c.model_id, capabilities?.model_details)}</option>)}</select></label>
          <label className="f">Photos folder (absolute path)<input type="text" value={folder} placeholder="/home/you/flights/2026-11-18" onChange={(e) => setFolder(e.target.value)} /></label>
        </div>
        <label className="f">Minimum confidence: <b>{confidence.toFixed(2)}</b>
          <input type="range" min={0.01} max={0.99} step={0.01} value={confidence} style={{ width: "100%" }} onChange={(e) => setConfidence(Number(e.target.value))} /></label>
        <button className="btn acc" onClick={start} disabled={active || !folder.trim()}>{active ? "Running…" : "Run preview"}</button>
        <ErrorText message={error} />
        {run && active && <div style={{ marginTop: 12 }}><p className="mute small">{run.phase}{run.total ? ` · ${run.done} / ${run.total}` : ""}</p><Progress done={run.done} total={run.total || 1} /></div>}
        {run?.error && <Banner kind="bad">{run.error.message}</Banner>}
      </div>
      {result && <BoxGallery runId={run!.run_id} items={result.images.filter((i) => i.image).map((i) => ({
        image: i.image!, caption: `${i.source} · ${i.boxes.length} box${i.boxes.length === 1 ? "" : "es"}`, boxes: i.boxes }))} />}
    </>
  );
}
