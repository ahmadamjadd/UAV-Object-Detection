// Shared building blocks in the Foxtrot design: pills, tabs, switches, banners, progress,
// sparkline, image galleries with boxes, plus routing and polling helpers.
import { ReactNode, useEffect, useRef, useState } from "react";
import { api, describeError, GenerationRun, GpuReport, PreviewBox } from "./api";

// ------------------------------------------------------------------ routing (hash based)

export type Page = "home" | "datasets" | "train" | "runs" | "evaluate" | "settings";
export const PAGES: [Page, string][] = [
  ["home", "Home"], ["datasets", "Datasets"], ["train", "Train"], ["runs", "Runs"], ["evaluate", "Evaluate"], ["settings", "Settings"],
];

// Read "#/runs?job=abc" as { page: "runs", params: {job: "abc"} }.
function parseHash(): { page: Page; params: URLSearchParams } {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [path, query] = raw.split("?");
  const page = (PAGES.find(([key]) => key === path)?.[0] ?? "home") as Page;
  return { page, params: new URLSearchParams(query ?? "") };
}

// The current page and its parameters; changes when the hash changes.
export function useRoute() {
  const [route, setRoute] = useState(parseHash);
  useEffect(() => {
    const update = () => { setRoute(parseHash()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return route;
}

// Go to a page, optionally with parameters (e.g. go("runs", { job: id })).
export function go(page: Page, params?: Record<string, string>) {
  const query = params && Object.keys(params).length ? `?${new URLSearchParams(params)}` : "";
  window.location.hash = `#/${page}${query}`;
}

// ------------------------------------------------------------------ polling

// Run `callback` now and then every `ms` milliseconds while the component is shown.
export function usePolling(callback: () => void, ms: number, enabled = true) {
  const saved = useRef(callback);
  saved.current = callback;
  useEffect(() => {
    if (!enabled) return;
    saved.current();
    const timer = window.setInterval(() => saved.current(), ms);
    return () => window.clearInterval(timer);
  }, [ms, enabled]);
}

const FINAL_RUN = new Set(["finished", "failed", "cancelled", "interrupted"]);

// Follow one background run (generation, DVC download, prediction preview) until it ends.
export function useRun() {
  const [run, setRun] = useState<GenerationRun | null>(null);
  const timer = useRef<number | null>(null);
  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current); }, []);
  // Poll every 0.7 s while the run is still going.
  function follow(started: GenerationRun) {
    setRun(started);
    api.generationRun(started.run_id).then((status) => {
      setRun(status);
      if (!FINAL_RUN.has(status.state)) timer.current = window.setTimeout(() => follow(status), 700);
    }, () => undefined);
  }
  return { run, follow, active: run !== null && !FINAL_RUN.has(run.state), reset: () => setRun(null) };
}

// ------------------------------------------------------------------ formatting

// The server sends UTC "2026-11-18 10:10:40" or ISO strings; show local time.
export function localTime(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value.includes("T") ? value : `${value.replace(" ", "T")}Z`);
  if (Number.isNaN(date.getTime())) return value;
  const today = new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (date.toDateString() === today.toDateString()) return `Today ${time}`;
  const yesterday = new Date(today.getTime() - 86_400_000);
  if (date.toDateString() === yesterday.toDateString()) return `Yesterday ${time}`;
  return `${date.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`;
}

// "yolo11s" -> "YOLO11 small" using the capabilities' model details.
export function modelName(id: string | null | undefined, details?: { model_id: string; display_name: string }[]): string {
  if (!id) return "—";
  return details?.find((m) => m.model_id === id)?.display_name ?? id;
}

// The first characters of a long ID, or a dash.
export const shortId = (id: string | null | undefined, n = 8) => (id ? id.slice(0, n) : "—");
// A score with fixed decimals, or a dash.
export const score = (value: number | null | undefined, digits = 3) => (typeof value === "number" ? value.toFixed(digits) : "—");
// A byte count as MB or GB.
export const megabytes = (bytes: number) => (bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`);

// ------------------------------------------------------------------ small components

const STATE_PILL: Record<string, string> = {
  queued: "p-mute", preparing: "p-blue", training: "p-acc", evaluating: "p-blue", finished: "p-ok", failed: "p-bad",
  cancelled: "p-mute", interrupted: "p-warn", complete: "p-ok", pruned: "p-mute", running: "p-acc", stopping: "p-warn",
  stopped: "p-mute", asked: "p-mute", preparing_data: "p-blue", pruning: "p-warn", synced: "p-ok", pending: "p-mute",
  syncing: "p-blue", retrying: "p-warn",
};

// A coloured state label (jobs, uploads, trials, studies).
export function StatePill({ state }: { state: string | null | undefined }) {
  if (!state) return <span className="mute">—</span>;
  return <span className={`pill ${STATE_PILL[state] ?? "p-mute"}`}>{state.replace("_", " ")}</span>;
}

// Dataset role: training / validation / test (protected).
export function RolePill({ role }: { role: string }) {
  if (role === "test") return <span className="pill p-warn">test · protected</span>;
  return <span className={`pill ${role === "validation" ? "p-ok" : "p-blue"}`}>{role}</span>;
}

// Upload state, worded as in the design.
export function UploadPill({ state, finished }: { state: string; finished: boolean }) {
  if (!finished) return <span className="mute">—</span>;
  if (state === "synced") return <span className="pill p-ok">in MLflow</span>;
  if (state === "retrying" || state === "failed") return <span className="pill p-warn">upload retrying</span>;
  return <span className="pill p-mute">upload pending</span>;
}

// A segmented control (the design's tabs).
export function Tabs<T extends string>({ value, options, onChange }: {
  value: T; options: [T, ReactNode][]; onChange: (value: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {options.map(([key, label]) => (
        <button key={key} role="tab" aria-selected={key === value} className={key === value ? "on" : ""}
          onClick={() => onChange(key)} type="button">{label}</button>
      ))}
    </div>
  );
}

// An on/off switch.
export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (value: boolean) => void; label?: string }) {
  return (
    <span className="sw"><input type="checkbox" checked={checked} aria-label={label}
      onChange={(e) => onChange(e.target.checked)} /><i /></span>
  );
}

// A switch with its label and help text on one line.
export function SwitchField({ label, help, checked, onChange }: {
  label: string; help?: string; checked: boolean; onChange: (value: boolean) => void;
}) {
  return (
    <label className="swl"><div>{label}{help && <span>{help}</span>}</div><Switch checked={checked} onChange={onChange} label={label} /></label>
  );
}

// A coloured message box.
export function Banner({ kind, icon, children }: { kind: "warn" | "info" | "acc" | "bad" | "ok"; icon?: string; children: ReactNode }) {
  return <div className={`banner ${kind}`}><b>{icon ?? (kind === "warn" || kind === "bad" ? "!" : kind === "ok" ? "✓" : "i")}</b><div>{children}</div></div>;
}

// A progress bar from done / total.
export function Progress({ done, total }: { done: number; total: number }) {
  const width = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  return <div className="bar" role="progressbar" aria-valuenow={done} aria-valuemax={total}><i style={{ width: `${width}%` }} /></div>;
}

// A small area chart of a score over epochs.
export function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const w = 400, h = 60, top = Math.max(...values, 0.05) * 1.1;
  const points = values.map((v, i) => `${(i / (values.length - 1)) * w},${h - (v / top) * h}`).join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-label="score over epochs">
      <polyline points={`0,${h} ${points} ${w},${h}`} fill="var(--accs)" stroke="none" />
      <polyline points={points} fill="none" stroke="var(--acc)" strokeWidth={2.2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// An explanation on hover for an ML term.
export function Tip({ text, children }: { text: string; children: ReactNode }) {
  return <span className="tip" title={text}>{children}</span>;
}

// A centred empty state with a next step.
export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="empty"><b>{title}</b>{children}</div>;
}

// Images with boxes drawn on top (label boxes, or predicted boxes with confidence).
export function BoxGallery({ runId, items }: {
  runId: string; items: { image: string; caption: string; boxes: (PreviewBox & { confidence?: number })[] }[];
}) {
  return (
    <div className="gal">
      {items.map((item) => (
        <figure key={item.image}>
          <div className="im">
            <img src={api.previewImageUrl(runId, item.image)} alt={item.caption} />
            {item.boxes.map((box, index) => (
              <div key={index} className="bx" style={{
                left: `${(box.x - box.width / 2) * 100}%`, top: `${(box.y - box.height / 2) * 100}%`,
                width: `${box.width * 100}%`, height: `${box.height * 100}%`,
              }}><b>{box.class_name}{box.confidence !== undefined ? ` ${box.confidence.toFixed(2)}` : ""}</b></div>
            ))}
          </div>
          <figcaption>{item.caption}</figcaption>
        </figure>
      ))}
    </div>
  );
}

// A form error, worded as the backend explained it.
export function ErrorText({ message }: { message: string }) {
  return message ? <p className="err small" role="alert">{message}</p> : null;
}

export const ICONS: Record<Page | "plus" | "sun" | "moon", string> = {
  home: "M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z",
  datasets: "M12 2c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  train: "M13 2L4 14h7l-1 8 9-12h-7z",
  runs: "M4 6h16M4 12h16M4 18h10",
  evaluate: "M9 12l2 2 4-4M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z",
  settings: "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z",
  plus: "M12 5v14M5 12h14",
  sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
};

// A stroke icon from ICONS.
export function Icon({ name, size = 18 }: { name: keyof typeof ICONS; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.9}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={ICONS[name]} /></svg>
  );
}

// ------------------------------------------------------------------ names

// Dataset ID -> member-chosen name, refreshed every few seconds (for tables that list runs).
export function useDatasetNames(): Record<string, string> {
  const [names, setNames] = useState<Record<string, string>>({});
  usePolling(() => {
    api.datasets().then((list) => setNames(Object.fromEntries(list.filter((d) => d.name).map((d) => [d.dataset_id, d.name as string]))), () => undefined);
  }, 10000);
  return names;
}

// "airfield noisy v2" for a dataset/run with a member-chosen name, otherwise its ID.
export const displayName = (item: { name?: string | null }, id: string | null | undefined, short = false) =>
  item.name || (short ? shortId(id) : id ?? "—");

// For dropdowns: "airfield noisy v2 · gen-cbbd13…" or just the ID.
export const optionLabel = (name: string | null | undefined, id: string) =>
  name ? `${name} · ${id.length > 20 ? `${id.slice(0, 16)}…` : id}` : id;

// The chosen name in bold with the ID underneath (or just the ID in code style).
export function Named({ name, id, short = false }: { name?: string | null; id: string; short?: boolean }) {
  if (!name) return <span className="mono">{short ? shortId(id, 6) : id}</span>;
  return <span><b style={{ fontWeight: 600 }}>{name}</b><span className="mono mute tiny" style={{ display: "block" }}>{short ? shortId(id, 8) : id}</span></span>;
}

// A pencil button that turns into a small "name" field; saving an empty name removes it.
export function RenameButton({ name, onSave, label = "Rename" }: {
  name?: string | null; onSave: (name: string | null) => Promise<unknown>; label?: string;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(name ?? "");
  const [error, setError] = useState("");
  useEffect(() => { if (!editing) setValue(name ?? ""); }, [name, editing]);
  // Save the new name and close the field.
  const save = () => onSave(value.trim() || null).then(() => { setEditing(false); setError(""); }, (e) => setError(describeError(e)));
  if (!editing) return (
    <button className="btn ghost sm" type="button" title={label} onClick={(e) => { e.stopPropagation(); setEditing(true); }}>✎ {name ? label : "Add name"}</button>
  );
  return (
    <span className="row" style={{ gap: 6, flexWrap: "nowrap" }} onClick={(e) => e.stopPropagation()}>
      <input type="text" value={value} autoFocus maxLength={80} placeholder="e.g. airfield noisy v2" style={{ marginTop: 0, minHeight: 32, padding: "4px 8px", width: 200 }}
        onChange={(e) => setValue(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") save(); if (e.key === "Escape") setEditing(false); }} />
      <button className="btn sm acc" type="button" onClick={save}>Save</button>
      <button className="btn sm ghost" type="button" onClick={() => setEditing(false)}>Cancel</button>
      {error && <span className="err tiny">{error}</span>}
    </span>
  );
}

// ------------------------------------------------------------------ GPU

// The GPU report (detected once by the backend; shared by every page).
let gpuPromise: Promise<GpuReport> | null = null;
// The detected GPUs and a function to check again.
export function useGpus(): [GpuReport | null, () => void] {
  const [report, setReport] = useState<GpuReport | null>(null);
  const load = (refresh: boolean) => {
    if (refresh || !gpuPromise) gpuPromise = api.gpu(refresh);
    gpuPromise.then(setReport, () => { gpuPromise = null; });
  };
  useEffect(() => load(false), []);
  return [report, () => { setReport(null); load(true); }];
}

// "Run on" choice: the CPU or a detected GPU, with a plain message when there is no GPU.
export function DeviceField({ value, onChange }: { value: string; onChange: (device: string) => void }) {
  const [gpus] = useGpus();
  return (
    <label className="f">Run on
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="cpu">CPU</option>
        {gpus?.devices.map((d) => <option key={d.id} value={d.id}>GPU {d.index} · {d.name} ({d.memory_gb} GB)</option>)}
        {gpus && !gpus.available && <option disabled>No GPU available</option>}
      </select>
      <span className="h">{!gpus ? "Checking for GPUs…" : gpus.available ? "A GPU trains much faster than the CPU." : gpus.message}</span>
    </label>
  );
}
