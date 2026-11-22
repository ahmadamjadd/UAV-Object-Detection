import { FormEvent, useEffect, useState } from "react";
import { useShared } from "../App";
import {
  api, Dataset, DatasetBuildResult, DatasetReport, describeError, GenerationRequest, PreviewResult, Range,
} from "../api";
import {
  Banner, BoxGallery, Empty, ErrorText, go, Named, Progress, RenameButton, RolePill, SwitchField, Tabs, usePolling, useRun,
} from "../ui";

// The three ways to add a dataset.
type AddTab = "generate" | "import" | "dvc";

// Datasets: what models learn from and are tested on; "Add dataset" offers three ways.
export function DatasetsPage({ params }: { params: URLSearchParams }) {
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [error, setError] = useState("");
  const adding = params.get("add") as AddTab | null;
  const load = () => api.datasets().then((d) => { setDatasets(d); setError(""); }, (e) => setError(describeError(e)));
  usePolling(load, 5000);

  return (
    <>
      <div className="row between" style={{ alignItems: "flex-start" }}>
        <div><h2>Datasets</h2><p className="sub">Data the models learn from and are tested on.</p></div>
        <button className="btn acc" onClick={() => go("datasets", { add: "generate" })}>+ Add dataset</button>
      </div>
      <div className="legend">
        <div><span className="pill p-blue">training</span><p>Used to train models.</p></div>
        <div><span className="pill p-ok">validation</span><p>Real labeled photos for comparing models. Never trained on.</p></div>
        <div style={{ borderColor: "var(--warn)" }}><span className="pill p-warn">test · protected</span>
          <p>Real labeled photos for the final honest score only. Never trained on or tuned with.</p></div>
      </div>
      <ErrorText message={error} />
      <div className="card scroll">
        {datasets && datasets.length === 0 ? (
          <Empty title="No datasets yet">Generate synthetic shapes, import a folder, or bring a saved DVC version.
            <div className="row" style={{ justifyContent: "center", marginTop: 12 }}>
              <button className="btn acc" onClick={() => go("datasets", { add: "generate" })}>Add dataset</button></div></Empty>
        ) : (
          <table>
            <thead><tr><th>Name</th><th>Classes</th><th>Role</th><th className="num">Images (train / val)</th><th>Status</th><th></th></tr></thead>
            <tbody>{(datasets ?? []).map((d) => (
              <tr key={d.dataset_id}>
                <td className="wrapc"><Named name={d.name} id={d.dataset_id} />
                  <div style={{ marginTop: 4 }}><RenameButton name={d.name} onSave={(name) => api.renameDataset(d.dataset_id, name).then(load)} /></div></td>
                <td className="wrapc">{d.class_names.join(", ")}</td>
                <td><RolePill role={d.role} /></td>
                <td className="num">{imageCounts(d)}</td>
                <td>{d.validation_status !== "valid" ? <span className="pill p-mute">{d.validation_status}</span>
                  : d.warning_count ? <span className="pill p-warn">{d.warning_count} warning{d.warning_count > 1 ? "s" : ""}</span>
                    : <span className="pill p-ok">ready</span>}</td>
                <td>{d.role === "training"
                  ? <button className="btn ghost sm" onClick={() => go("train", { dataset: d.dataset_id })}>Train on it</button>
                  : <button className="btn ghost sm" onClick={() => go("evaluate", { dataset: d.dataset_id })}>Evaluate on it</button>}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </div>
      {adding && <AddDataset tab={adding} onDone={load} />}
    </>
  );
}

// "12,000 / 3,000" for training sets, "412" for evaluation sets.
function imageCounts(d: Dataset): string {
  if (!d.images) return "—";
  const n = (v: number | undefined) => (v ?? 0).toLocaleString();
  return d.role === "training" ? `${n(d.images.train)} / ${n(d.images.val)}` : n(Object.values(d.images)[0]);
}

// The "Add dataset" window with its three tabs.
function AddDataset({ tab, onDone }: { tab: AddTab; onDone: () => void }) {
  const close = () => go("datasets");
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Add dataset" onClick={(e) => { if (e.target === e.currentTarget) close(); }}>
      <div className="card">
        <h3 style={{ fontSize: 18 }}>Add dataset<button className="r btn ghost" onClick={close}>Close</button></h3>
        <Tabs value={tab} onChange={(t) => go("datasets", { add: t })} options={[
          ["generate", "Generate synthetic"], ["import", "Import a folder"], ["dvc", "From Git / DVC"]]} />
        {tab === "generate" && <GenerateTab onDone={onDone} />}
        {tab === "import" && <ImportTab onDone={onDone} />}
        {tab === "dvc" && <DvcTab onDone={onDone} />}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ A) generate synthetic

function GenerateTab({ onDone }: { onDone: () => void }) {
  const [form, setForm] = useState<GenerationRequest | null>(null);
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [name, setName] = useState("");
  const { run, follow, active } = useRun();
  useEffect(() => {
    api.generationOptions().then((o) => {
      if (!o.available) setUnavailable(o.message);
      setForm({ recipe: o.recipe, split: o.split, backgrounds: o.default_backgrounds ?? "" });
    }, (e) => setError(describeError(e)));
  }, []);
  useEffect(() => {  // name the new (or reused) dataset as soon as it exists
    if (run?.kind !== "dataset" || run.state !== "finished") return;
    const id = (run.result as DatasetBuildResult | null)?.dataset_id;
    (name.trim() && id ? api.renameDataset(id, name.trim()) : Promise.resolve()).then(onDone, onDone);
  }, [run?.state]);

  // Start a preview or a full dataset with the current settings.
  async function start(kind: "preview" | "dataset") {
    if (!form) return;
    setError("");
    try { follow(kind === "preview" ? await api.startPreview(form) : await api.startGeneration(form)); }
    catch (e) { setError(describeError(e)); }
  }

  if (!form) return <p className="mute">{error || "Loading…"}</p>;
  const r = form.recipe;
  const recipe = (change: Partial<GenerationRequest["recipe"]>) => setForm({ ...form, recipe: { ...r, ...change } });
  const split = (change: Partial<GenerationRequest["split"]>) => setForm({ ...form, split: { ...form.split, ...change } });
  const num = (label: string, value: number, set: (n: number) => void, step: number | "any" = 1, help?: string) => (
    <label className="f">{label}<input type="number" step={step} value={Number.isFinite(value) ? value : ""}
      onChange={(e) => set(e.target.valueAsNumber)} />{help && <span className="h">{help}</span>}</label>);
  const range = (label: string, value: Range, set: (v: Range) => void, step: number | "any" = 1) => (
    <label className="f">{label}<span className="pair">
      <input type="number" step={step} value={value.low} aria-label={`${label} low`} onChange={(e) => set({ ...value, low: e.target.valueAsNumber })} />
      <span className="mute">to</span>
      <input type="number" step={step} value={value.high} aria-label={`${label} high`} onChange={(e) => set({ ...value, high: e.target.valueAsNumber })} />
    </span></label>);
  return (
    <>
      {unavailable && <Banner kind="warn">{unavailable}</Banner>}
      <label className="f">Name (optional)<input type="text" value={name} maxLength={80} placeholder="e.g. airfield · noisy · 100 per class"
        onChange={(e) => setName(e.target.value)} /><span className="h">How you will recognise this dataset later. The ID (a fingerprint of the settings) is kept too.</span></label>
      <div className="fg">
        <label className="f">Backgrounds folder<input type="text" value={form.backgrounds} placeholder="/path/to/backgrounds"
          onChange={(e) => setForm({ ...form, backgrounds: e.target.value })} /><span className="h">Real photos the shapes are drawn on (absolute path).</span></label>
        {num("Images per class", r.images_per_class, (n) => recipe({ images_per_class: n }))}
        {num("Canvas size", r.canvas_size, (n) => recipe({ canvas_size: n }), 1, "Size of each generated image. Training image size is set in Train.")}
        {num("Train share %", Math.round(form.split.train_ratio * 100), (n) => split({ train_ratio: n / 100 }))}
        {num("Generation seed", r.seed, (n) => recipe({ seed: n }))}
        {num("Split seed", form.split.seed, (n) => split({ seed: n }))}
      </div>
      <SwitchField label="Keep images from one background in the same split" help="Validation never shows a background the model trained on."
        checked={form.split.group_by_background} onChange={(b) => split({ group_by_background: b })} />
      <details className="grp"><summary>More settings</summary><div className="in fg">
        {range("Shape scale range (share of canvas)", r.shape_scale, (v) => recipe({ shape_scale: v }), 0.05)}
        {range("Blur range", r.blur, (v) => recipe({ blur: v }))}
        {num("Noise", r.noise, (n) => recipe({ noise: n }))}
        {num("Shape/letter brightness difference", r.brightness_difference, (n) => recipe({ brightness_difference: n }))}
        {range("Contrast range", r.contrast, (v) => recipe({ contrast: v }), 0.05)}
        {range("Brightness shift range", r.brightness, (v) => recipe({ brightness: v }))}
        {range("Saturation shift range", r.saturation, (v) => recipe({ saturation: v }))}
        {num("Letters probability (0–1)", r.letter_probability, (n) => recipe({ letter_probability: n }), 0.05)}
        <SwitchField label="Draw letters on shapes" checked={r.add_letter} onChange={(b) => recipe({ add_letter: b })} />
        <SwitchField label="Desaturate (grey images)" checked={r.desaturate} onChange={(b) => recipe({ desaturate: b })} />
        <SwitchField label="Light gradient" checked={r.add_light_gradient} onChange={(b) => recipe({ add_light_gradient: b })} />
        {num("Light probability (0–1)", r.light_probability, (n) => recipe({ light_probability: n }), 0.05)}
        {num("Light gradient opacity (0–1)", r.light_opacity, (n) => recipe({ light_opacity: n }), 0.05)}
      </div></details>
      <div className="row" style={{ margin: "6px 0 16px" }}>
        <button className="btn" disabled={active || !!unavailable} onClick={() => start("preview")}>
          {active && run?.kind === "preview" ? "Generating preview…" : run?.kind === "preview" ? "Preview again" : "Preview samples"}</button>
        <button className="btn acc" disabled={active || !!unavailable} onClick={() => start("dataset")}>Generate dataset</button>
      </div>
      <ErrorText message={error} />
      {run && <GenerationRunView run={run} />}
    </>
  );
}

// Progress, then preview images or the generated dataset's summary.
function GenerationRunView({ run }: { run: NonNullable<ReturnType<typeof useRun>["run"]> }) {
  const finalState = ["finished", "failed", "cancelled", "interrupted"].includes(run.state);
  if (!finalState) return (
    <div className="card" style={{ background: "var(--card2)" }}>
      <div className="row between small"><b>{run.kind === "preview" ? "Preview" : "Generating dataset"} · {run.phase}</b>
        <span className="mute">{run.total ? `${run.done} / ${run.total} images` : ""}</span></div>
      <div style={{ margin: "8px 0" }}><Progress done={run.done} total={run.total || 1} /></div>
      <button className="btn sm" onClick={() => api.cancelGeneration(run.run_id)}>Cancel</button>
    </div>
  );
  if (run.error) return <Banner kind="bad">{run.error.message}</Banner>;
  if (run.state === "cancelled") return <Banner kind="info">Stopped. Nothing was published.</Banner>;
  if (run.kind === "preview" && run.result) {
    const preview = run.result as PreviewResult;
    return (<>
      <p className="mute small" style={{ margin: "0 0 8px" }}>{preview.images.length} sample images with label boxes · canvas {preview.canvas_size}px · seed {preview.seed}</p>
      <BoxGallery runId={run.run_id} items={preview.images.map((i) => ({ image: i.image, caption: `${i.image} · ${i.background}`, boxes: i.boxes }))} />
    </>);
  }
  const result = run.result as DatasetBuildResult;
  return (
    <div className="card" style={{ background: "var(--card2)" }}>
      <h3><span className="mono">{result.dataset_id}</span>
        <span className={`pill ${result.cached ? "p-blue" : "p-ok"}`}>{result.cached ? "reused (same settings and backgrounds)" : "new"}</span></h3>
      {result.counts && <div className="kv">
        <div><b>{result.counts.images}</b><span>images</span></div>
        <div><b>{result.counts.train} / {result.counts.val}</b><span>train / val</span></div>
        <div><b>{result.groups.train ?? "?"} / {result.groups.val ?? "?"}</b><span>backgrounds per split</span></div>
        <div><b>{Object.values(result.counts.objects_per_class).join(" / ")}</b><span>{Object.keys(result.counts.objects_per_class).join(" / ")}</span></div>
      </div>}
      {result.report && <ReportView report={result.report} />}
      <div className="row" style={{ marginTop: 14 }}><button className="btn acc" onClick={() => go("train", { dataset: result.dataset_id })}>Train on this dataset</button></div>
    </div>
  );
}

// ------------------------------------------------------------------ B) import a folder

function ImportTab({ onDone }: { onDone: () => void }) {
  const [role, setRole] = useState<Dataset["role"]>("training");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [error, setError] = useState("");

  // Check and import; an invalid folder returns its report without importing.
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const text = (name: string) => String(form.get(name) ?? "").trim();
    setBusy(true); setError(""); setMessage(""); setReport(null);
    try {
      const body: Record<string, unknown> = { source: text("source"), dataset_id: text("dataset_id"), role, name: text("name") || null,
        class_names: text("class_names").split(",").map((n) => n.trim()).filter(Boolean) };
      if (role === "training") Object.assign(body, { train_images_ref: text("train_images_ref"), validation_images_ref: text("validation_images_ref") });
      else Object.assign(body, { images_ref: text("images_ref"), test_share: Number(form.get("test_share") ?? 0) / 100 });
      const result = await api.importDataset(body);
      setReport(result.report);
      const names = (result.datasets ?? (result.dataset ? [result.dataset] : [])).map((d) => `${d.dataset_id} (${d.role})`);
      setMessage(result.imported ? `Imported ${names.join(" and ")}.` : "Not imported: fix the problems below and try again.");
      if (result.imported) onDone();
    } catch (e) { setError(describeError(e)); } finally { setBusy(false); }
  }

  return (
    <form onSubmit={submit}>
      <p className="mute small" style={{ marginTop: 0 }}>A YOLO detection folder on this computer. It is checked, then copied into the workspace; the original is not changed.</p>
      <label className="f">Use<select value={role} onChange={(e) => setRole(e.target.value as Dataset["role"])}>
        <option value="training">Training (train/ and val/ folders)</option>
        <option value="validation">Real labeled images: validation</option>
        <option value="test">Real labeled images: final test</option></select></label>
      <div className="fg">
        <label className="f">Folder (absolute path)<input type="text" name="source" required placeholder="/home/you/datasets/shapes" /></label>
        <label className="f">Dataset ID<input type="text" name="dataset_id" required pattern="[A-Za-z0-9][A-Za-z0-9_\-]*" placeholder="shapes-v1" /><span className="h">Short, no spaces; cannot change later.</span></label>
        <label className="f">Name (optional)<input type="text" name="name" maxLength={80} placeholder="e.g. Field photos, October" /><span className="h">Free text; you can rename it later.</span></label>
        <label className="f">Class names, in label order (0, 1, …)<input type="text" name="class_names" required placeholder="Hexagon, Triangle" /></label>
        {role === "training" ? (<>
          <label className="f">Train images folder<input type="text" name="train_images_ref" required defaultValue="train/images" /></label>
          <label className="f">Validation images folder<input type="text" name="validation_images_ref" required defaultValue="val/images" /></label>
        </>) : (<>
          <label className="f">Images folder<input type="text" name="images_ref" required defaultValue="images" /><span className="h">Labels go in the matching "labels" folder; every image needs one.</span></label>
          <label className="f">Also set aside a final test set (%)<input type="number" name="test_share" min={0} max={90} step={5} defaultValue={0} />
            <span className="h">Optional groups.json ({"{"}"photo.jpg": "flight-3"{"}"}) keeps each flight or scene in only one of validation and test.</span></label>
        </>)}
      </div>
      <button className="btn acc" type="submit" disabled={busy}>{busy ? "Checking…" : "Check and import"}</button>
      <ErrorText message={error} />
      {message && <p className={report && !report.valid ? "err small" : "okc small"}>{message}</p>}
      {report && <ReportView report={report} />}
    </form>
  );
}

// ------------------------------------------------------------------ C) from Git / DVC

function DvcTab({ onDone }: { onDone: () => void }) {
  const { capabilities } = useShared();
  const { run, follow, active } = useRun();
  const [role, setRole] = useState<Dataset["role"]>("training");
  const [error, setError] = useState("");
  const [name, setName] = useState("");
  useEffect(() => {  // name the downloaded dataset once it is registered
    if (run?.state !== "finished") return;
    const id = (run.result as Record<string, string> | null)?.dataset_id;
    (name.trim() && id ? api.renameDataset(id, name.trim()) : Promise.resolve()).then(onDone, onDone);
  }, [run?.state]);

  // Start the download; the run reports its phase and the exact commit when done.
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const text = (name: string) => String(form.get(name) ?? "").trim();
    setError("");
    try {
      follow(await api.startDvcImport({
        repository: text("repository"), revision: text("revision"), path: text("path"), dataset_id: text("dataset_id"), role,
        class_names: text("class_names").split(",").map((n) => n.trim()).filter(Boolean),
        train_images_ref: text("train_images_ref") || "images",
        ...(role === "training" ? { validation_images_ref: text("validation_images_ref") } : {}),
        ...(text("credentials") ? { credentials_destination_id: text("credentials") } : {}),
      }));
    } catch (e) { setError(describeError(e)); }
  }

  const done = run?.state === "finished" ? run.result as Record<string, string> : null;
  return (
    <form onSubmit={submit}>
      <p className="mute small" style={{ marginTop: 0 }}>Downloads a DVC-tracked folder exactly as it was at a commit, tag or branch. The repository is not changed; the exact commit is recorded and logged to MLflow.</p>
      <div className="fg">
        <label className="f">Repository (folder or Git URL)<input type="text" name="repository" required defaultValue={capabilities?.dvc_default_repository ?? ""} /></label>
        <label className="f">Revision<input type="text" name="revision" required defaultValue="HEAD" /><span className="h">Commit, tag or branch.</span></label>
        <label className="f">DVC folder<input type="text" name="path" required placeholder="data/split" /></label>
        <label className="f">Use<select value={role} onChange={(e) => setRole(e.target.value as Dataset["role"])}>
          <option value="training">Training</option><option value="validation">Validation (real)</option><option value="test">Final test (real)</option></select></label>
        <label className="f">Dataset ID<input type="text" name="dataset_id" required pattern="[A-Za-z0-9][A-Za-z0-9_\-]*" /></label>
        <label className="f">Name (optional)<input type="text" value={name} maxLength={80} placeholder="e.g. Split from November release" onChange={(e) => setName(e.target.value)} /></label>
        <label className="f">Class names, in label order<input type="text" name="class_names" required placeholder="Hexagon, Triangle" /></label>
        <label className="f">{role === "training" ? "Train images folder" : "Images folder"}
          <input type="text" name="train_images_ref" key={role} defaultValue={role === "training" ? "train/images" : "images"} /></label>
        {role === "training" && <label className="f">Validation images folder<input type="text" name="validation_images_ref" defaultValue="val/images" /></label>}
        <label className="f">Remote login from tracking destination<input type="text" name="credentials" list="dvc-destinations" placeholder="optional, e.g. dagshub" />
          <datalist id="dvc-destinations">{(capabilities?.tracking_destinations ?? []).map((d) => <option key={d} value={d} />)}</datalist>
          <span className="h">For a private DVC remote; uses that destination's saved username and token.</span></label>
      </div>
      <button className="btn acc" type="submit" disabled={active}>{active ? "Downloading…" : "Download and register"}</button>
      <ErrorText message={error} />
      {run && active && <p className="mute small">{run.phase}</p>}
      {run?.error && <Banner kind="bad">{run.error.message}</Banner>}
      {done && <Banner kind="ok">Registered <b className="mono">{done.dataset_id}</b> ({done.role}) from commit <span className="mono">{String(done.commit).slice(0, 10)}</span>.</Banner>}
    </form>
  );
}

// ------------------------------------------------------------------ check report

// Per-split counts, then the errors and warnings found by the dataset check.
export function ReportView({ report }: { report: DatasetReport }) {
  return (
    <div style={{ marginTop: 14 }}>
      <div className="row"><b className="small">Check report</b><span className={`pill ${report.valid ? "p-ok" : "p-bad"}`}>{report.valid ? "valid" : "invalid"}</span></div>
      <div className="scroll"><table style={{ marginTop: 6 }}>
        <thead><tr><th>Split</th><th className="num">Images</th><th className="num">Missing labels</th><th>Objects per class</th></tr></thead>
        <tbody>{Object.entries(report.splits).map(([name, s]) => (
          <tr key={name}><td>{name}</td><td className="num">{s.images}</td><td className="num">{s.missing_labels}</td>
            <td className="mute">{Object.entries(s.objects_per_class).map(([k, v]) => `${k}: ${v}`).join(", ")}</td></tr>
        ))}</tbody>
      </table></div>
      {[["Errors", report.error_count, report.errors, "err"], ["Warnings", report.warning_count, report.warnings, "mute"]].map(([title, count, issues, cls]) =>
        (count as number) > 0 && (
          <details key={title as string} className="plain" style={{ marginTop: 8 }} open={title === "Errors"}>
            <summary>{title as string} ({count as number}{(issues as any[]).length < (count as number) ? `, first ${(issues as any[]).length} shown` : ""})</summary>
            <div className="evlog">{(issues as DatasetReport["errors"]).map((i, n) => (
              <div key={n} className={cls as string}>{i.split}{i.file ? ` / ${i.file}` : ""}{i.line ? ` line ${i.line}` : ""}: {i.message}</div>))}</div>
          </details>
        ))}
    </div>
  );
}
