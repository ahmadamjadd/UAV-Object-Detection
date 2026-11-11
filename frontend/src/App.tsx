import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  api, Capabilities, Dataset, describeError, Destination, JobDetail, JobEvent, JobView, TERMINAL,
} from "./api";

type Section = "setup" | "train" | "activity";

export function App() {
  const [section, setSection] = useState<Section>("activity");
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    api.health().then(() => setHealthy(true), () => setHealthy(false));
    api.capabilities().then(setCapabilities, () => setCapabilities(null));
  }, []);

  return (
    <div className="shell">
      <header>
        <h1>Foxtrot</h1>
        <span className={`dot ${healthy ? "ok" : healthy === false ? "bad" : ""}`} title="Local API" />
        {capabilities && !capabilities.real_training && (
          <span className="badge">Fixture mode: no real model training yet</span>
        )}
        <nav>
          {(["setup", "train", "activity"] as Section[]).map((name) => (
            <button key={name} className={section === name ? "active" : ""} onClick={() => setSection(name)}>
              {{ setup: "Setup", train: "New training", activity: "Activity" }[name]}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {section === "setup" && <Setup capabilities={capabilities} />}
        {section === "train" && (
          <TrainingForm capabilities={capabilities}
            onSubmitted={(id) => { setSelected(id); setSection("activity"); }} />
        )}
        {section === "activity" && <Activity selected={selected} onSelect={setSelected} />}
      </main>
    </div>
  );
}

function Setup({ capabilities }: { capabilities: Capabilities | null }) {
  const [destinations, setDestinations] = useState<Destination[]>([]);
  const [message, setMessage] = useState("");
  const load = () => api.destinations().then(setDestinations, (e) => setMessage(describeError(e)));
  useEffect(() => { load(); }, []);

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const value: Record<string, unknown> = {
      destination_id: form.get("destination_id"), kind: form.get("kind"),
      tracking_uri: form.get("tracking_uri"), browser_url: form.get("browser_url") || null,
    };
    if (form.get("username")) value.username = form.get("username");
    if (form.get("token")) value.token = form.get("token");
    try {
      await api.saveDestination(value);
      setMessage("Saved. Credentials are stored by the backend and never shown again.");
      event.currentTarget?.reset();
      load();
    } catch (error) { setMessage(describeError(error)); }
  }

  return (
    <section className="grid">
      <div className="card">
        <h2>Environment</h2>
        {capabilities ? (
          <dl>
            <dt>Version</dt><dd>{capabilities.version}</dd>
            <dt>Models</dt><dd>{capabilities.models.join(", ")}</dd>
            <dt>Devices</dt><dd>{capabilities.devices.join(", ")}</dd>
            <dt>Worker adapter</dt><dd>{capabilities.active_pipeline_adapter ?? "H5 infrastructure fixture"}</dd>
          </dl>
        ) : <p className="muted">Loading…</p>}
      </div>
      <div className="card">
        <h2>Tracking destinations</h2>
        {destinations.length === 0 ? <p className="muted">None configured yet.</p> : (
          <ul className="plain">
            {destinations.map((d) => (
              <li key={d.destination_id}>
                <strong>{d.destination_id}</strong> <span className="muted">{d.kind} · {d.tracking_uri}</span>
                {d.has_credentials && <span className="badge small">credentials saved</span>}
              </li>
            ))}
          </ul>
        )}
        <form onSubmit={save} className="form">
          <label>ID<input name="destination_id" required pattern="[A-Za-z0-9][A-Za-z0-9_\-]*" /></label>
          <label>Kind<select name="kind"><option value="mlflow">MLflow server</option><option value="local">Local MLflow</option></select></label>
          <label>Tracking URI<input name="tracking_uri" required placeholder="https://… or file:///…" /></label>
          <label>Browser URL<input name="browser_url" placeholder="Optional link for viewing runs" /></label>
          <label>Username<input name="username" autoComplete="off" /></label>
          <label>Token<input name="token" type="password" autoComplete="new-password" /></label>
          <button type="submit">Save destination</button>
        </form>
        {message && <p className="muted">{message}</p>}
      </div>
    </section>
  );
}

function TrainingForm({ capabilities, onSubmitted }: {
  capabilities: Capabilities | null; onSubmitted: (id: string) => void;
}) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [defaults, setDefaults] = useState<Record<string, any>>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submissionKey = useRef<string | null>(null);

  useEffect(() => {
    api.datasets().then(setDatasets, (e) => setError(describeError(e)));
    // Defaults come from the backend schema, not a separate frontend list.
    api.schemas().then((s) => setDefaults(s.job?.$defs?.TrainingSettings?.properties ?? {}));
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const number = (name: string) => Number(form.get(name));
    // Reuse one idempotency key until success, so a retry never duplicates a job.
    submissionKey.current ??= crypto.randomUUID();
    setBusy(true);
    setError("");
    try {
      const job = await api.submit(submissionKey.current, {
        dataset_id: form.get("dataset_id"),
        model_id: form.get("model_id"),
        tracking_destination_id: form.get("tracking_destination_id"),
        training: {
          epochs: number("epochs"), batch_size: number("batch_size"),
          image_size: number("image_size"), seed: number("seed"), device: form.get("device"),
        },
      });
      submissionKey.current = null;
      onSubmitted(job.job_id);
    } catch (e) { setError(describeError(e)); } finally { setBusy(false); }
  }

  const field = (name: string, label: string) => (
    <label>{label}<input name={name} type="number" min={name === "seed" ? 0 : 1} required
      key={`${name}-${defaults[name]?.default}`} defaultValue={defaults[name]?.default} /></label>
  );

  return (
    <section className="card narrow">
      <h2>New training</h2>
      {datasets.length === 0 && <p className="muted">No datasets are registered in this workspace yet.</p>}
      <form onSubmit={submit} className="form">
        <label>Dataset<select name="dataset_id" required>
          {datasets.map((d) => <option key={d.dataset_id} value={d.dataset_id}>
            {d.dataset_id} ({d.class_names.join(", ")})</option>)}
        </select></label>
        <label>Model<select name="model_id">
          {(capabilities?.models ?? []).map((m) => <option key={m}>{m}</option>)}
        </select></label>
        {field("epochs", "Epochs")}
        {field("batch_size", "Batch size")}
        {field("image_size", "Image size")}
        {field("seed", "Seed")}
        <label>Device<select name="device">
          {(capabilities?.devices ?? []).map((d) => <option key={d}>{d}</option>)}
        </select></label>
        <label>Tracking destination<input name="tracking_destination_id" required
          list="destinations" defaultValue={capabilities?.tracking_destinations[0] ?? "local"} />
          <datalist id="destinations">
            {(capabilities?.tracking_destinations ?? []).map((d) => <option key={d} value={d} />)}
          </datalist></label>
        <button type="submit" disabled={busy || datasets.length === 0}>{busy ? "Submitting…" : "Start training"}</button>
      </form>
      {error && <p className="error">{error}</p>}
    </section>
  );
}

function usePolling(callback: () => void, milliseconds: number) {
  useEffect(() => {
    callback();
    const timer = window.setInterval(callback, milliseconds);
    return () => window.clearInterval(timer);
  }, [callback, milliseconds]);
}

function Activity({ selected, onSelect }: { selected: string | null; onSelect: (id: string) => void }) {
  const [jobs, setJobs] = useState<JobView[]>([]);
  const [error, setError] = useState("");
  const load = useCallback(() => {
    api.jobs().then((value) => { setJobs(value); setError(""); }, (e) => setError(describeError(e)));
  }, []);
  usePolling(load, 2000);

  return (
    <section className="grid wide">
      <div className="card">
        <h2>Jobs</h2>
        {error && <p className="error">{error}</p>}
        {jobs.length === 0 ? <p className="muted">No jobs yet.</p> : (
          <table>
            <thead><tr><th>Job</th><th>State</th><th>Upload</th><th>Created</th></tr></thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.job_id} className={job.job_id === selected ? "selected" : ""}
                  onClick={() => onSelect(job.job_id)}>
                  <td><code>{job.job_id.slice(0, 8)}</code></td>
                  <td><span className={`state ${job.execution_state}`}>{job.execution_state}</span></td>
                  <td className="muted">{job.tracking_state}</td>
                  <td className="muted">{job.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {selected ? <JobPanel key={selected} jobId={selected} /> :
        <div className="card"><p className="muted">Select a job to see its progress.</p></div>}
    </section>
  );
}

function JobPanel({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const cursor = useRef(0);

  // Read only events after the last one seen, so reconnects recover missed updates.
  const refresh = useCallback(() => {
    api.job(jobId).then(setJob, () => undefined);
    api.events(jobId, cursor.current).then((fresh) => {
      if (fresh.length) {
        cursor.current = fresh[fresh.length - 1].sequence;
        setEvents((old) => [...old, ...fresh].slice(-500));
      }
    }, () => undefined);
  }, [jobId]);
  usePolling(refresh, 1000);

  const latest = [...events].reverse().find((e) => e.event_type === "progress")?.payload;
  const done = Number(latest?.epoch ?? latest?.step ?? 0);
  const total = Number(latest?.epochs ?? latest?.total ?? 0);
  const active = job && !TERMINAL.has(job.execution_state);

  return (
    <div className="card">
      <h2>Job <code>{jobId.slice(0, 8)}</code></h2>
      {job && (
        <>
          <p><span className={`state ${job.execution_state}`}>{job.execution_state}</span>
            {job.cancel_requested && active && <span className="muted"> · cancelling…</span>}</p>
          {total > 0 && <progress max={total} value={done} />}
          {total > 0 && <p className="muted">{done} / {total}</p>}
          {job.error && <p className="error">{job.error.code}: {job.error.message}</p>}
          {job.tracking_run?.run_url && <p><a href={job.tracking_run.run_url} target="_blank" rel="noreferrer">View in MLflow</a></p>}
          {active && !job.cancel_requested && (
            <button onClick={() => api.cancel(jobId).then(refresh)}>Cancel</button>
          )}
        </>
      )}
      <h3>Events</h3>
      <ol className="events">
        {[...events].reverse().slice(0, 50).map((e) => (
          <li key={e.sequence}><span className="muted">#{e.sequence} {e.event_type}</span> {summarize(e)}</li>
        ))}
      </ol>
    </div>
  );
}

function summarize(event: JobEvent): string {
  const p = event.payload;
  if (event.event_type === "state_changed") return `${p.domain}: ${p.previous ?? "—"} → ${p.state}`;
  if (event.event_type === "progress") return Object.entries(p).map(([k, v]) => `${k}=${v}`).join(" ");
  if (typeof p.text === "string") return `${p.channel}: ${p.text.trim().slice(0, 160)}`;
  return String(p.action ?? JSON.stringify(p)).slice(0, 160);
}
