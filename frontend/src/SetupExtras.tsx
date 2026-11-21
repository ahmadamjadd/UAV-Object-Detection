import { useEffect, useState } from "react";
import { api, describeError } from "./api";

// Setup card: is this installation ready? (Python, libraries, pins, generator, disk, weights, interface)
export function SystemCheck() {
  const [report, setReport] = useState<{ ok: boolean; checks: { name: string; ok: boolean; detail: string }[] } | null>(null);
  const run = () => api.systemCheck().then(setReport, () => setReport(null));
  useEffect(() => { run(); }, []);
  return (
    <div className="card">
      <h2>System check {report && <span className={`state ${report.ok ? "finished" : "failed"}`}>{report.ok ? "ready" : "needs attention"}</span>}</h2>
      {!report ? <p className="muted">Checking…</p> : (
        <ul className="plain">{report.checks.map((c) => (
          <li key={c.name}>{c.ok ? "✅" : "⚠️"} <strong>{c.name}</strong> <span className="muted">{c.detail}</span></li>
        ))}</ul>
      )}
      <button onClick={run}>Check again</button>
    </div>
  );
}

// Setup card: the member name shown on every run in shared MLflow (foxtrot.member tag).
export function Profile() {
  const [member, setMember] = useState("");
  const [message, setMessage] = useState("");
  useEffect(() => { api.profile().then((p) => setMember(p.member ?? ""), () => undefined); }, []);
  // Save the member name.
  const save = () => api.saveProfile(member).then(() => setMessage("Saved. New runs carry this name."),
    (e) => setMessage(describeError(e)));
  return (
    <div className="card">
      <h2>Member</h2>
      <p className="muted">Your name or initials, so teammates can tell whose runs are whose in a shared MLflow.</p>
      <div className="actions"><input value={member} onChange={(e) => setMember(e.target.value)} placeholder="e.g. Ahmad" />
        <button onClick={save}>Save</button></div>
      {message && <p className="muted small">{message}</p>}
    </div>
  );
}

// Setup card: disk use per area and safe cleanup (never active, resumable or not-yet-uploaded work).
export function Storage() {
  const [data, setData] = useState<{ usage: Record<string, number>; candidates: { id: string; path: string; bytes: number; reason: string }[] } | null>(null);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [message, setMessage] = useState("");
  const load = () => api.storage().then((d) => { setData(d); setChosen(new Set()); }, (e) => setMessage(describeError(e)));
  useEffect(() => { load(); }, []);
  const mb = (bytes: number) => `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  // Remove the ticked items (the backend re-checks each one first).
  const clean = () => api.cleanup([...chosen]).then((r) => { setMessage(`Freed ${mb(r.freed_bytes)}${r.skipped.length ? `; ${r.skipped.length} no longer safe, kept` : ""}.`); load(); },
    (e) => setMessage(describeError(e)));
  return (
    <div className="card">
      <h2>Storage</h2>
      {data && <p className="muted small">{Object.entries(data.usage).map(([k, v]) => `${k} ${mb(v)}`).join(" · ")}</p>}
      {data && data.candidates.length === 0 && <p className="muted">Nothing to clean up.</p>}
      {data && data.candidates.length > 0 && (<>
        <ul className="plain scroll">{data.candidates.map((c) => (
          <li key={c.id}><label className="check"><input type="checkbox" checked={chosen.has(c.id)} onChange={(e) => {
            const next = new Set(chosen); e.target.checked ? next.add(c.id) : next.delete(c.id); setChosen(next);
          }} /><span><code>{c.path}</code> · {mb(c.bytes)} <span className="muted">· {c.reason}</span></span></label></li>
        ))}</ul>
        <button onClick={clean} disabled={chosen.size === 0}>Remove selected</button>
      </>)}
      <p className="muted small">Datasets, best checkpoints, results, and anything active, resumable or not yet uploaded are never offered.</p>
      {message && <p className="muted small">{message}</p>}
    </div>
  );
}
