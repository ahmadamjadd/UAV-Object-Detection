import { FormEvent, useEffect, useState } from "react";
import { useShared } from "../App";
import { api, describeError, Destination, ModelWeights } from "../api";
import { Banner, ErrorText, megabytes, useGpus } from "../ui";

// Settings: configure this computer.
export function SettingsPage() {
  return (
    <>
      <h2>Settings</h2>
      <p className="sub">Configure this computer.</p>
      <div className="grid g2">
        <div className="grid" style={{ alignContent: "start" }}><SystemCheck /><Destinations /></div>
        <div className="grid" style={{ alignContent: "start" }}><Gpu /><Member /><Weights /><Storage /><About /></div>
      </div>
    </>
  );
}

// Is this installation ready? (Python, libraries, pins, generator, disk, weights, interface)
function SystemCheck() {
  const [report, setReport] = useState<{ ok: boolean; checks: { name: string; ok: boolean; detail: string }[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const run = () => { setBusy(true); api.systemCheck().then(setReport, () => setReport(null)).finally(() => setBusy(false)); };
  useEffect(run, []);
  return (
    <div className="card"><h3>System check{report && <span className={`pill ${report.ok ? "p-ok" : "p-warn"}`}>{report.ok ? "ready" : "needs attention"}</span>}
      <button className="r btn ghost" onClick={run} disabled={busy}>{busy ? "Checking…" : "Check again"}</button></h3>
      {!report ? <p className="mute">Checking…</p> : <div className="list">{report.checks.map((c) => (
        <div key={c.name}><span className={`pill ${c.ok ? "p-ok" : "p-warn"}`}>{c.ok ? "✓" : "!"}</span><div>{c.name}<div className="mute small">{c.detail}</div></div></div>))}</div>}
    </div>
  );
}

// Tracking destinations: where results are uploaded (MLflow). Tokens are write-only.
function Destinations() {
  const { member } = useShared();
  const [items, setItems] = useState<Destination[]>([]);
  const [editing, setEditing] = useState<Destination | "new" | null>(null);
  const [checks, setChecks] = useState<Record<string, { ok?: boolean; message: string }>>({});
  const [error, setError] = useState("");
  const load = () => api.destinations().then(setItems, (e) => setError(describeError(e)));
  useEffect(() => { load(); }, []);

  // Try to reach one destination and show the answer next to it.
  async function test(id: string) {
    setChecks((old) => ({ ...old, [id]: { message: "Testing…" } }));
    try { const r = await api.checkDestination(id); setChecks((old) => ({ ...old, [id]: r })); }
    catch (e) { setChecks((old) => ({ ...old, [id]: { ok: false, message: describeError(e) } })); }
  }

  // Save the form (new or edited destination); blank token keeps the saved one.
  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const text = (name: string) => String(form.get(name) ?? "").trim();
    const value: Record<string, unknown> = { destination_id: text("destination_id"), kind: text("kind"), tracking_uri: text("tracking_uri"),
      browser_url: text("browser_url") || null, experiment_name: text("experiment_name") || "Foxtrot Detection" };
    if (text("username")) value.username = text("username");
    if (text("token")) value.token = text("token");
    try { await api.saveDestination(value); setEditing(null); setError(""); load(); } catch (e) { setError(describeError(e)); }
  }

  const current = editing && editing !== "new" ? editing : null;
  return (
    <div className="card"><h3>Tracking destinations<button className="r btn ghost sm" onClick={() => setEditing("new")}>+ Add</button></h3>
      <p className="mute small" style={{ marginTop: 0 }}>Where finished runs are uploaded. Local MLflow uses <span className="mono">sqlite:///path/mlflow.db</span>; DagsHub uses <span className="mono">https://dagshub.com/&lt;user&gt;/&lt;repo&gt;.mlflow</span>.</p>
      {items.length === 0 && !editing && <Banner kind="info">No destination yet. Add one so runs can be uploaded to MLflow.</Banner>}
      {items.map((d) => (
        <div key={d.destination_id} className="att">
          <div className="row"><b>{d.destination_id}</b><span className="pill p-blue">{d.kind}</span>{d.has_credentials && <span className="pill p-ok">credentials saved</span>}</div>
          <p className="mono mute" style={{ margin: "6px 0 0", overflowWrap: "anywhere" }}>{d.tracking_uri}</p>
          <p className="small" style={{ margin: "4px 0 0" }}>Runs go into the MLflow experiment <b>{member || d.experiment_name}</b>
            {member ? " (your member name)" : " — set your member name below to use it instead"}</p>
          <div className="row">
            <button className="btn" onClick={() => test(d.destination_id)} disabled={checks[d.destination_id]?.message === "Testing…"}>Test connection</button>
            <button className="btn ghost" onClick={() => setEditing(d)}>Edit</button>
            {checks[d.destination_id] && <span className={`small ${checks[d.destination_id].ok ? "okc" : checks[d.destination_id].ok === false ? "err" : "mute"}`}>
              {checks[d.destination_id].ok ? "✓ " : checks[d.destination_id].ok === false ? "✗ " : ""}{checks[d.destination_id].message}</span>}
          </div>
        </div>))}
      {editing && (
        <form className="att" onSubmit={save} key={current?.destination_id ?? "new"}>
          <b className="small">{current ? `Edit ${current.destination_id}` : "New destination"}</b>
          <div className="fg" style={{ marginTop: 10 }}>
            <label className="f">ID<input type="text" name="destination_id" required pattern="[A-Za-z0-9][A-Za-z0-9_\-]*" defaultValue={current?.destination_id ?? ""} readOnly={Boolean(current)} /></label>
            <label className="f">Kind<select name="kind" defaultValue={current?.kind ?? "mlflow"}><option value="mlflow">MLflow server / DagsHub</option><option value="local">Local MLflow</option></select></label>
          </div>
          <label className="f">Tracking URI<input type="text" name="tracking_uri" required defaultValue={current?.tracking_uri ?? ""} placeholder="https://… or sqlite:///…" /></label>
          <label className="f">Experiment when no member name is set<input type="text" name="experiment_name" required maxLength={250} defaultValue={current?.experiment_name ?? "Foxtrot Detection"} />
            <span className="h">Runs always go into an MLflow experiment named after the member who trained them (Settings › Member);
              this name is used only until a member name is set. Uploads already made stay where they are.</span></label>
          <label className="f">Browser URL<input type="text" name="browser_url" defaultValue={current?.browser_url ?? ""} placeholder="Optional link for viewing runs" /></label>
          <div className="fg">
            <label className="f">Username<input type="text" name="username" autoComplete="off" /></label>
            <label className="f">Token<input type="password" name="token" autoComplete="new-password" placeholder={current?.has_credentials ? "Saved — type to replace" : ""} /></label>
          </div>
          <div className="row"><button className="btn acc" type="submit">Save</button><button className="btn ghost" type="button" onClick={() => setEditing(null)}>Cancel</button></div>
          <p className="mute tiny" style={{ marginBottom: 0 }}>Tokens are stored privately by the backend and never shown again.</p>
        </form>)}
      <ErrorText message={error} />
    </div>
  );
}

// GPUs this computer can train on, or why there are none.
function Gpu() {
  const [report, recheck] = useGpus();
  return (
    <div className="card"><h3>GPU{report && <span className={`pill ${report.available ? "p-ok" : "p-mute"}`}>{report.available ? "available" : "none"}</span>}
      <button className="r btn ghost sm" onClick={recheck}>Check again</button></h3>
      {!report ? <p className="mute small">Checking (asks PyTorch for CUDA devices)…</p> : report.available ? (
        <div className="list">{report.devices.map((d) => (
          <div key={d.id}><span className="pill p-ok">{d.id}</span><div>{d.name}<div className="mute small">{d.memory_gb} GB memory</div></div></div>))}</div>
      ) : <Banner kind="info">{report.message}</Banner>}
      {report && <p className="mute tiny" style={{ margin: "8px 0 0" }}>PyTorch {report.torch ?? "?"}{report.cuda_build ? ` · built for CUDA ${report.cuda_build}` : " · CPU-only build"}. Choose the device per run under Train › Run on.</p>}
    </div>
  );
}

// The member name shown on runs in a shared MLflow.
function Member() {
  const { member, setMember } = useShared();
  const [value, setValue] = useState(member ?? "");
  const [message, setMessage] = useState("");
  useEffect(() => { setValue(member ?? ""); }, [member]);
  return (
    <div className="card"><h3>Member name</h3>
      <label className="f">Shown on your runs in shared MLflow<input type="text" value={value} onChange={(e) => setValue(e.target.value)} placeholder="e.g. Ahmad" /></label>
      <div className="row"><button className="btn" onClick={() => api.saveProfile(value).then((p) => { setMember(p.member); setMessage("Saved. New runs carry this name and go into the MLflow experiment with this name."); },
        (e) => setMessage(describeError(e)))}>Save</button>{message && <span className="mute small">{message}</span>}</div>
    </div>
  );
}

// Official pretrained weights, fingerprinted when installed.
function Weights() {
  const [models, setModels] = useState<ModelWeights[]>([]);
  const [choice, setChoice] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const load = () => api.models().then((m) => { setModels(m); setChoice((c) => c || m.find((x) => x.status === "missing")?.model_id || ""); }, (e) => setMessage(describeError(e)));
  useEffect(() => { load(); }, []);
  const present = models.filter((m) => m.status !== "missing"), missing = models.filter((m) => m.status === "missing");
  // Download one model's official weights.
  const install = (id: string) => { setBusy(true); setMessage(`Downloading ${id}.pt…`);
    api.installModel(id).then((r) => { setMessage(`Installed ${id}.pt (${megabytes(r.size_bytes ?? 0)}).`); setChoice(""); load(); },
      (e) => setMessage(describeError(e))).finally(() => setBusy(false)); };
  return (
    <div className="card"><h3>Pretrained weights</h3>
      {present.length === 0 ? <p className="mute small">None installed yet — needed for runs that start from pretrained weights.</p> : (
        <div className="list">{present.map((m) => (
          <div key={m.model_id}><span className="mono" style={{ flex: 1 }}>{m.model_id}.pt</span>
            {m.status === "installed" ? <span className="mute mono">sha {m.sha256?.slice(0, 4)}…{m.sha256?.slice(-3)}</span>
              : <><span className="pill p-bad">changed</span><button className="btn sm" disabled={busy} onClick={() => install(m.model_id)}>Reinstall</button></>}</div>))}</div>)}
      {missing.length > 0 && <div className="row" style={{ marginTop: 10 }}>
        <select style={{ flex: 1, margin: 0 }} value={choice} onChange={(e) => setChoice(e.target.value)} aria-label="Model to install">
          {missing.map((m) => <option key={m.model_id} value={m.model_id}>{m.model_id}.pt</option>)}</select>
        <button className="btn acc" disabled={busy || !choice} onClick={() => install(choice)}>{busy ? "Downloading…" : "Install"}</button></div>}
      {message && <p className="mute small" style={{ marginBottom: 0 }}>{message}</p>}
    </div>
  );
}

// Disk use per area and safe cleanup.
function Storage() {
  const [data, setData] = useState<{ usage: Record<string, number>; candidates: { id: string; path: string; bytes: number; reason: string }[] } | null>(null);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [message, setMessage] = useState("");
  const load = () => api.storage().then((d) => { setData(d); setChosen(new Set()); }, (e) => setMessage(describeError(e)));
  useEffect(() => { load(); }, []);
  const total = data ? Object.values(data.usage).reduce((a, b) => a + b, 0) : 0;
  const names: Record<string, string> = { jobs: "Runs (checkpoints, logs)", datasets: "Datasets", studies: "Studies", generation: "Previews & caches",
    models: "Pretrained weights", "tracking-outbox": "Upload queue" };
  // Remove ticked items; the backend re-checks each first.
  const clean = () => api.cleanup([...chosen]).then((r) => { setMessage(`Freed ${megabytes(r.freed_bytes)}${r.skipped.length ? `; ${r.skipped.length} kept (no longer safe)` : ""}.`); load(); },
    (e) => setMessage(describeError(e)));
  return (
    <div className="card"><h3>Storage{data && <span className="r mute small">{megabytes(total)} total</span>}</h3>
      {data && Object.entries(data.usage).sort((a, b) => b[1] - a[1]).map(([area, bytes]) => (
        <div key={area} style={{ marginBottom: 10 }}>
          <div className="row between small"><span>{names[area] ?? area}</span><span className="mute">{megabytes(bytes)}</span></div>
          <div className="bar"><i style={{ width: `${total ? Math.max(2, (bytes / total) * 100) : 0}%` }} /></div>
        </div>))}
      {data && data.candidates.length > 0 && (<>
        <details className="plain" style={{ marginTop: 6 }}><summary>Safe to remove: {data.candidates.length} item{data.candidates.length > 1 ? "s" : ""} · {megabytes(data.candidates.reduce((a, c) => a + c.bytes, 0))}</summary>
          <div className="scrolly" style={{ marginTop: 8 }}>{data.candidates.map((c) => (
            <label key={c.id} className="row small" style={{ alignItems: "flex-start", gap: 8, padding: "6px 0", borderTop: "1px solid var(--line)" }}>
              <input type="checkbox" checked={chosen.has(c.id)} onChange={(e) => { const next = new Set(chosen); e.target.checked ? next.add(c.id) : next.delete(c.id); setChosen(next); }} />
              <span style={{ flex: 1 }}><span className="mono" style={{ overflowWrap: "anywhere" }}>{c.path}</span><span className="mute"> · {megabytes(c.bytes)} · {c.reason}</span></span>
            </label>))}</div>
          <div className="row" style={{ marginTop: 8 }}>
            <button className="btn ghost sm" onClick={() => setChosen(new Set(data.candidates.map((c) => c.id)))}>Select all</button>
            <button className="btn sm" onClick={clean} disabled={chosen.size === 0}>Remove selected</button></div>
        </details></>)}
      <p className="mute tiny" style={{ margin: "8px 0 0" }}>Datasets, best checkpoints and active, resumable or unsynced work are never offered for cleanup.</p>
      {message && <p className="mute small" style={{ marginBottom: 0 }}>{message}</p>}
    </div>
  );
}

// Version and worker adapter.
function About() {
  const { capabilities } = useShared();
  return (
    <div className="card"><h3>About</h3>
      <p className="mute small" style={{ margin: 0 }}>Foxtrot {capabilities?.version ?? "…"} · worker adapter: <span className="mono">{capabilities?.active_pipeline_adapter ?? "test fixture"}</span>
        {capabilities && !capabilities.real_training ? " (test mode)" : ""} · {capabilities?.models.length ?? 0} models · device {capabilities?.devices.join(", ")}</p>
    </div>
  );
}
