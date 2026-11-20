import { FormEvent, useEffect, useState } from "react";
import { api, Capabilities, Dataset, DatasetReport, describeError } from "./api";
import { useRun } from "./Evaluate";

// The Datasets section: registered datasets, importing a folder, and bringing in a DVC version.
export function Datasets({ capabilities }: { capabilities: Capabilities | null }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [role, setRole] = useState<Dataset["role"]>("training");

  // Reload the list of registered datasets from the backend.
  const load = () => api.datasets().then(setDatasets, (e) => setMessage(describeError(e)));
  useEffect(() => { load(); }, []);

  // Send the form to the backend, then show the check report (and refresh the list if imported).
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const classNames = String(form.get("class_names")).split(",").map((n) => n.trim()).filter(Boolean);
    setBusy(true);
    setMessage("");
    setReport(null);
    try {
      const body: Record<string, unknown> = { source: form.get("source"), dataset_id: form.get("dataset_id"),
        class_names: classNames, role };
      if (role === "training") {
        body.train_images_ref = form.get("train_images_ref");
        body.validation_images_ref = form.get("validation_images_ref");
      } else {
        body.images_ref = form.get("images_ref");
        body.test_share = Number(form.get("test_share") ?? 0) / 100;
      }
      const result = await api.importDataset(body);
      setReport(result.report);
      const names = (result.datasets ?? (result.dataset ? [result.dataset] : [])).map((d) => `"${d.dataset_id}" (${d.role})`);
      setMessage(result.imported ? `Imported ${names.join(" and ")}.`
        : "Not imported: fix the errors below and try again.");
      if (result.imported) load();
    } catch (error) { setMessage(describeError(error)); } finally { setBusy(false); }
  }

  return (
    <section className="grid">
      <div className="card">
        <h2>Registered datasets</h2>
        {datasets.length === 0 ? <p className="muted">None yet. Import one with the form.</p> : (
          <ul className="plain">
            {datasets.map((d) => (
              <li key={d.dataset_id}>
                <strong>{d.dataset_id}</strong> <span className="muted">{d.class_names.join(", ")}</span>
                <span className={`badge small role-${d.role}`}>{d.role}</span>
              </li>
            ))}
          </ul>
        )}
        <p className="muted small">training: used to train · validation: real labeled images to compare models ·
          test: real labeled images for the final check only (never trained on or tuned against).</p>
      </div>
      <div className="card">
        <h2>Import a dataset</h2>
        <p className="muted">A YOLO detection folder on this computer. It is checked, then copied into the
          workspace; the original is not changed.</p>
        <form onSubmit={submit} className="form">
          <label>Use<select value={role} onChange={(e) => setRole(e.target.value as Dataset["role"])}>
            <option value="training">Training (train/ and val/ folders)</option>
            <option value="validation">Real labeled images: validation</option>
            <option value="test">Real labeled images: final test</option>
          </select></label>
          <label>Folder (absolute path)<input name="source" required placeholder="/home/you/datasets/shapes" /></label>
          <label>Dataset ID<input name="dataset_id" required pattern="[A-Za-z0-9][A-Za-z0-9_\-]*" placeholder="shapes-v1" /></label>
          <label>Class names, in label order (0, 1, …)<input name="class_names" required placeholder="Hexagon, Triangle" /></label>
          {role === "training" ? (
            <>
              <label>Train images folder<input name="train_images_ref" required defaultValue="train/images" /></label>
              <label>Validation images folder<input name="validation_images_ref" required defaultValue="val/images" /></label>
            </>
          ) : (
            <>
              <label>Images folder (labels in the matching "labels" folder)<input name="images_ref" required defaultValue="images" /></label>
              <label>Also set aside a final test set (% of images, 0 = none)
                <input name="test_share" type="number" min="0" max="90" step="5" defaultValue="0" /></label>
              <p className="muted small">Optional groups.json in the folder ({"{"}"photo.jpg": "flight-3"{"}"}) keeps each
                flight or scene in only one of validation and test. Every image needs a label file.</p>
            </>
          )}
          <button type="submit" disabled={busy}>{busy ? "Checking…" : "Check and import"}</button>
        </form>
        {message && <p className={report && !report.valid ? "error" : "muted"}>{message}</p>}
        {report && <ReportView report={report} />}
      </div>
      <DvcImport capabilities={capabilities} onImported={load} />
    </section>
  );
}

// Bring in a dataset exactly as recorded at a Git/DVC revision (downloaded in the background).
function DvcImport({ capabilities, onImported }: { capabilities: Capabilities | null; onImported: () => void }) {
  const { run, follow, active } = useRun();
  const [role, setRole] = useState<Dataset["role"]>("training");
  const [error, setError] = useState("");
  useEffect(() => { if (run?.state === "finished") onImported(); }, [run?.state]);

  // Start the download; the run reports its phase and the exact commit when done.
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setError("");
    const text = (name: string) => String(form.get(name) ?? "").trim();
    try {
      follow(await api.startDvcImport({
        repository: text("repository"), revision: text("revision"), path: text("path"),
        dataset_id: text("dataset_id"), role,
        class_names: text("class_names").split(",").map((n) => n.trim()).filter(Boolean),
        train_images_ref: text("train_images_ref") || "images",
        ...(role === "training" ? { validation_images_ref: text("validation_images_ref") } : {}),
        ...(text("credentials") ? { credentials_destination_id: text("credentials") } : {}),
      }));
    } catch (e) { setError(describeError(e)); }
  }

  return (
    <div className="card">
      <h2>From Git/DVC</h2>
      <p className="muted">Downloads a DVC-tracked folder exactly as it was at a commit, tag or branch, into the
        workspace. The repository is not changed. The exact commit is recorded and logged to MLflow.</p>
      <form onSubmit={submit} className="form">
        <label>Repository (folder or Git URL)<input name="repository" required
          defaultValue={capabilities?.dvc_default_repository ?? ""} /></label>
        <div className="row">
          <label>Revision<input name="revision" required defaultValue="HEAD" placeholder="commit, tag or branch" /></label>
          <label>DVC folder<input name="path" required placeholder="data/split" /></label>
        </div>
        <label>Use<select value={role} onChange={(e) => setRole(e.target.value as Dataset["role"])}>
          <option value="training">Training</option><option value="validation">Validation (real)</option>
          <option value="test">Final test (real)</option></select></label>
        <label>Dataset ID<input name="dataset_id" required pattern="[A-Za-z0-9][A-Za-z0-9_\-]*" /></label>
        <label>Class names, in label order<input name="class_names" required placeholder="Hexagon, Triangle" /></label>
        <div className="row">
          <label>{role === "training" ? "Train images folder" : "Images folder"}
            <input name="train_images_ref" defaultValue={role === "training" ? "train/images" : "images"} key={role} /></label>
          {role === "training" && <label>Validation images folder<input name="validation_images_ref" defaultValue="val/images" /></label>}
        </div>
        <label>Remote login from tracking destination (optional)<input name="credentials"
          list="dvc-destinations" placeholder="e.g. dagshub" />
          <datalist id="dvc-destinations">{(capabilities?.tracking_destinations ?? []).map((d) => <option key={d} value={d} />)}</datalist></label>
        <button type="submit" disabled={active}>{active ? "Downloading…" : "Download and register"}</button>
      </form>
      {error && <p className="error">{error}</p>}
      {run && active && <p className="muted">{run.phase}</p>}
      {run?.error && <p className="error">{run.error.message}</p>}
      {run?.state === "finished" && run.result && (() => {
        const done = run.result as Record<string, string>;
        return <p className="muted">Registered <strong>{done.dataset_id}</strong> ({done.role}) from
          commit <code>{String(done.commit).slice(0, 10)}</code>.</p>;
      })()}
    </div>
  );
}

// Shows a check report: per-split counts, then the errors and warnings found.
export function ReportView({ report }: { report: DatasetReport }) {
  return (
    <div>
      <h3>Check report: {report.valid ? "valid" : "invalid"}</h3>
      <table>
        <thead><tr><th>Split</th><th>Images</th><th>Missing labels</th><th>Objects per class</th></tr></thead>
        <tbody>
          {Object.entries(report.splits).map(([split, s]) => (
            <tr key={split}>
              <td>{split}</td><td>{s.images}</td><td>{s.missing_labels}</td>
              <td className="muted">{Object.entries(s.objects_per_class).map(([k, v]) => `${k}: ${v}`).join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <IssueList title="Errors" count={report.error_count} issues={report.errors} className="error" />
      <IssueList title="Warnings" count={report.warning_count} issues={report.warnings} className="muted" />
    </div>
  );
}

// Lists problems with their split, file and line, noting when only the first ones are shown.
function IssueList({ title, count, issues, className }: {
  title: string; count: number; issues: DatasetReport["errors"]; className: string;
}) {
  if (count === 0) return null;
  return (
    <>
      <h3>{title} ({count}{issues.length < count ? `, first ${issues.length} shown` : ""})</h3>
      <ol className="events">
        {issues.map((issue, index) => (
          <li key={index} className={className}>
            {issue.split}{issue.file ? ` / ${issue.file}` : ""}{issue.line ? ` line ${issue.line}` : ""}: {issue.message}
          </li>
        ))}
      </ol>
    </>
  );
}
