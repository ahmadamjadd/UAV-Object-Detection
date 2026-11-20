import { FormEvent, useEffect, useState } from "react";
import { api, Dataset, DatasetReport, describeError } from "./api";

// The Datasets section: lists registered datasets and imports a folder from this computer.
export function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

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
      const result = await api.importDataset({
        source: form.get("source"), dataset_id: form.get("dataset_id"), class_names: classNames,
        train_images_ref: form.get("train_images_ref"), validation_images_ref: form.get("validation_images_ref"),
      });
      setReport(result.report);
      setMessage(result.imported
        ? `Imported "${result.dataset?.dataset_id}". It is now available in New training.`
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
                <span className="badge small">{d.validation_status}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="card">
        <h2>Import a dataset</h2>
        <p className="muted">A YOLO detection folder on this computer, e.g. with train/images, train/labels,
          val/images and val/labels. It is checked, then copied into the workspace; the original is not changed.</p>
        <form onSubmit={submit} className="form">
          <label>Folder (absolute path)<input name="source" required placeholder="/home/you/datasets/shapes" /></label>
          <label>Dataset ID<input name="dataset_id" required pattern="[A-Za-z0-9][A-Za-z0-9_\-]*" placeholder="shapes-v1" /></label>
          <label>Class names, in label order (0, 1, …)<input name="class_names" required placeholder="Hexagon, Triangle" /></label>
          <label>Train images folder<input name="train_images_ref" required defaultValue="train/images" /></label>
          <label>Validation images folder<input name="validation_images_ref" required defaultValue="val/images" /></label>
          <button type="submit" disabled={busy}>{busy ? "Checking…" : "Check and import"}</button>
        </form>
        {message && <p className={report && !report.valid ? "error" : "muted"}>{message}</p>}
        {report && <ReportView report={report} />}
      </div>
    </section>
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
