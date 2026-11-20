import { useEffect, useRef, useState } from "react";
import {
  api, DatasetBuildResult, describeError, GenerationRequest, GenerationRun, PreviewResult, Range,
} from "./api";
import { ReportView } from "./Datasets";

const FINAL = new Set(["finished", "failed", "cancelled", "interrupted"]);

// The Generate section: choose settings, preview samples, build a dataset, then train on it.
export function Generate({ onTrain }: { onTrain: (datasetId: string) => void }) {
  const [form, setForm] = useState<GenerationRequest | null>(null);
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const [run, setRun] = useState<GenerationRun | null>(null);
  const [message, setMessage] = useState("");
  const timer = useRef<number | null>(null);

  // Load the backend's defaults (the same values as params.yaml) into the form.
  useEffect(() => {
    api.generationOptions().then((options) => {
      if (!options.available) setUnavailable(options.message);
      setForm({ recipe: options.recipe, split: options.split, backgrounds: options.default_backgrounds ?? "" });
    }, (e) => setMessage(describeError(e)));
    return () => { if (timer.current) window.clearTimeout(timer.current); };
  }, []);

  // Ask for the run's status until it ends.
  function follow(runId: string) {
    api.generationRun(runId).then((status) => {
      setRun(status);
      if (!FINAL.has(status.state)) timer.current = window.setTimeout(() => follow(runId), 700);
    }, (e) => setMessage(describeError(e)));
  }

  // Start a preview or a full dataset run with the current settings.
  async function start(kind: "preview" | "dataset") {
    if (!form) return;
    setMessage("");
    try {
      const started = kind === "preview" ? await api.startPreview(form) : await api.startGeneration(form);
      setRun(started);
      follow(started.run_id);
    } catch (e) { setMessage(describeError(e)); }
  }

  if (!form) return <section className="card"><p className="muted">{message || "Loading…"}</p></section>;
  const running = run !== null && !FINAL.has(run.state);
  return (
    <section className="grid wide">
      <div className="card">
        <h2>Generate a dataset</h2>
        {unavailable && <p className="error">{unavailable}</p>}
        <Settings form={form} onChange={setForm} />
        <div className="actions">
          <button onClick={() => start("preview")} disabled={running || !!unavailable}>Preview samples</button>
          <button className="primary" onClick={() => start("dataset")} disabled={running || !!unavailable}>
            Generate dataset</button>
        </div>
        {message && <p className="error">{message}</p>}
      </div>
      <div className="card">
        {run ? <RunView run={run} onTrain={onTrain} /> :
          <p className="muted">Preview a few labeled samples first, then generate the full dataset.
            Same settings and backgrounds reuse the dataset already made.</p>}
      </div>
    </section>
  );
}

// All generator and split settings, with the important ones first and the rest under "More settings".
function Settings({ form, onChange }: { form: GenerationRequest; onChange: (value: GenerationRequest) => void }) {
  const r = form.recipe;
  const recipe = (change: Partial<GenerationRequest["recipe"]>) => onChange({ ...form, recipe: { ...r, ...change } });
  const split = (change: Partial<GenerationRequest["split"]>) => onChange({ ...form, split: { ...form.split, ...change } });
  const num = (label: string, value: number, set: (n: number) => void, step = 1, hint?: string) => (
    <label>{label}<input type="number" step={step} value={value} title={hint}
      onChange={(e) => set(e.target.valueAsNumber)} /></label>
  );
  const range = (label: string, value: Range, set: (v: Range) => void, step = 1) => (
    <label>{label}<span className="pair">
      <input type="number" step={step} value={value.low} onChange={(e) => set({ ...value, low: e.target.valueAsNumber })} />
      <span className="muted">to</span>
      <input type="number" step={step} value={value.high} onChange={(e) => set({ ...value, high: e.target.valueAsNumber })} />
    </span></label>
  );
  const check = (label: string, value: boolean, set: (b: boolean) => void) => (
    <label className="check"><input type="checkbox" checked={value} onChange={(e) => set(e.target.checked)} />{label}</label>
  );
  return (
    <div className="form">
      <label>Backgrounds folder (absolute path)<input value={form.backgrounds}
        onChange={(e) => onChange({ ...form, backgrounds: e.target.value })} placeholder="/path/to/backgrounds" /></label>
      <div className="row">
        {num("Images per class", r.images_per_class, (n) => recipe({ images_per_class: n }))}
        {num("Canvas size (px)", r.canvas_size, (n) => recipe({ canvas_size: n }))}
        {num("Generation seed", r.seed, (n) => recipe({ seed: n }))}
      </div>
      <p className="muted small">Canvas size is the generated picture size. The training image size is chosen
        separately in New training.</p>
      <div className="row">
        {num("Train share (%)", Math.round(form.split.train_ratio * 100), (n) => split({ train_ratio: n / 100 }))}
        {num("Split seed", form.split.seed, (n) => split({ seed: n }))}
      </div>
      {check("Keep images from one background in the same split", form.split.group_by_background,
        (b) => split({ group_by_background: b }))}
      <details>
        <summary>More settings</summary>
        <div className="form">
          {range("Shape scale (share of canvas)", r.shape_scale, (v) => recipe({ shape_scale: v }), 0.05)}
          <div className="row">
            {num("Noise", r.noise, (n) => recipe({ noise: n }))}
            {num("Shape/letter brightness difference", r.brightness_difference, (n) => recipe({ brightness_difference: n }))}
          </div>
          {range("Blur kernel", r.blur, (v) => recipe({ blur: v }))}
          {range("Contrast", r.contrast, (v) => recipe({ contrast: v }), 0.05)}
          {range("Brightness shift", r.brightness, (v) => recipe({ brightness: v }))}
          {range("Saturation shift", r.saturation, (v) => recipe({ saturation: v }))}
          {check("Desaturate (grey images)", r.desaturate, (b) => recipe({ desaturate: b }))}
          {check("Draw letters on shapes", r.add_letter, (b) => recipe({ add_letter: b }))}
          {num("Letter probability (0–1)", r.letter_probability, (n) => recipe({ letter_probability: n }), 0.05)}
          {check("Add light gradient", r.add_light_gradient, (b) => recipe({ add_light_gradient: b }))}
          <div className="row">
            {num("Light probability (0–1)", r.light_probability, (n) => recipe({ light_probability: n }), 0.05)}
            {num("Light opacity (0–1)", r.light_opacity, (n) => recipe({ light_opacity: n }), 0.05)}
          </div>
        </div>
      </details>
    </div>
  );
}

// Progress of the current run, then its preview images or dataset summary.
function RunView({ run, onTrain }: { run: GenerationRun; onTrain: (datasetId: string) => void }) {
  const active = !FINAL.has(run.state);
  return (
    <>
      <h2>{run.kind === "preview" ? "Preview" : "Dataset"} <span className={`state ${run.state}`}>{run.state}</span></h2>
      {active && (
        <>
          <p className="muted">{run.phase}{run.total > 0 ? ` · ${run.done} / ${run.total} images` : ""}</p>
          <progress max={run.total || 1} value={run.total ? run.done : 0} />
          <p><button onClick={() => api.cancelGeneration(run.run_id)}>Cancel</button></p>
        </>
      )}
      {run.error && <p className="error">{run.error.message}</p>}
      {run.state === "cancelled" && <p className="muted">Stopped. Nothing was published.</p>}
      {run.state === "finished" && run.kind === "preview" && run.result &&
        <PreviewGallery runId={run.run_id} preview={run.result as PreviewResult} />}
      {run.state === "finished" && run.kind === "dataset" && run.result &&
        <DatasetSummary result={run.result as DatasetBuildResult} onTrain={onTrain} />}
    </>
  );
}

// Sample images with their label boxes drawn on top, exactly as the label files say.
function PreviewGallery({ runId, preview }: { runId: string; preview: PreviewResult }) {
  return (
    <>
      <p className="muted">{preview.images.length} samples · canvas {preview.canvas_size}px · seed {preview.seed} ·
        {" "}{preview.background_count} backgrounds</p>
      <div className="gallery">
        {preview.images.map((item) => (
          <figure key={item.image}>
            <div className="sample">
              <img src={api.previewImageUrl(runId, item.image)} alt={item.image} />
              {item.boxes.map((box, index) => (
                <div key={index} className="box" style={{
                  left: `${(box.x - box.width / 2) * 100}%`, top: `${(box.y - box.height / 2) * 100}%`,
                  width: `${box.width * 100}%`, height: `${box.height * 100}%`,
                }}><span>{box.class_name}</span></div>
              ))}
            </div>
            <figcaption className="muted">{item.image} · {item.background}</figcaption>
          </figure>
        ))}
      </div>
    </>
  );
}

// What the full run produced (or reused), with a button to train on it.
function DatasetSummary({ result, onTrain }: { result: DatasetBuildResult; onTrain: (datasetId: string) => void }) {
  return (
    <>
      <p><strong>{result.dataset_id}</strong>{" "}
        <span className="badge small">{result.cached ? "reused: same settings and backgrounds" : "new"}</span></p>
      {result.counts && (
        <dl>
          <dt>Images</dt><dd>{result.counts.images}</dd>
          <dt>Train / validation</dt><dd>{result.counts.train} / {result.counts.val}</dd>
          <dt>Backgrounds per split</dt><dd>{result.groups.train ?? "?"} / {result.groups.val ?? "?"}</dd>
          <dt>Per class</dt><dd>{Object.entries(result.counts.objects_per_class).map(([k, v]) => `${k}: ${v}`).join(", ")}</dd>
        </dl>
      )}
      <p><button className="primary" onClick={() => onTrain(result.dataset_id)}>Train on this dataset</button></p>
      {result.report && <ReportView report={result.report} />}
    </>
  );
}
