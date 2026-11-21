import { useCallback, useEffect, useState } from "react";
import { api, describeError, formatTime } from "./api";

type Item = {
  job_id: string; execution_state: string; tracking_state: string; finished_at: string | null; kind?: string;
  model_id?: string; dataset_id?: string; request?: Record<string, any>; resumable?: boolean; resume_reason?: string | null;
  error: { code: string; message: string } | null; tracking_error: { code: string; message: string } | null;
};
type Report = {
  interrupted: Item[]; holds: Item[]; uploads: Item[];
  studies: { study_id: string; name: string; state: string; reason: string | null; trials_started: number; planned_trials: number }[];
  uploader: { next_check_seconds: number; last_run: string | null };
};

// The Recovery section: what stopped early or is not uploaded yet, and the safe ways to continue.
// Resume (same job settings, continues from last.pt) ≠ retry (start again) ≠ new experiment (change settings)
// ≠ continue a study (its remaining trials).
export function Recovery({ onOpenJob, onNewExperiment }: {
  onOpenJob: (id: string) => void; onNewExperiment: (request: Record<string, unknown>) => void;
}) {
  const [report, setReport] = useState<Report | null>(null);
  const [message, setMessage] = useState("");
  const load = useCallback(() => { api.recovery().then(setReport, (e) => setMessage(describeError(e))); }, []);
  useEffect(() => { load(); const t = window.setInterval(load, 4000); return () => window.clearInterval(t); }, [load]);

  // Submit a new job built from an earlier one, then show it in Activity.
  async function submit(request: Record<string, unknown>) {
    setMessage("");
    try { onOpenJob((await api.submit(crypto.randomUUID(), request)).job_id); }
    catch (e) { setMessage(describeError(e)); }
  }

  if (!report) return <section className="card"><p className="muted">{message || "Loading…"}</p></section>;
  const nothing = !report.interrupted.length && !report.holds.length && !report.uploads.length && !report.studies.length;
  return (
    <section className="grid">
      <div className="card">
        <h2>Stopped training</h2>
        {nothing && <p className="muted">Nothing needs attention.</p>}
        {report.interrupted.length === 0 ? <p className="muted">No interrupted, failed or cancelled training.</p> : (
          <ul className="plain">
            {report.interrupted.map((job) => {
              const { study, ...base } = job.request ?? {};
              return (
                <li key={job.job_id} className="recovery">
                  <div><code>{job.job_id.slice(0, 8)}</code> <span className={`state ${job.execution_state}`}>{job.execution_state}</span>{" "}
                    <span className="muted">{job.model_id} · {job.dataset_id} · {formatTime(job.finished_at)}</span></div>
                  {job.error && <div className="error small">{job.error.code}: {job.error.message}</div>}
                  {!job.resumable && <div className="muted small">Cannot resume: {job.resume_reason}</div>}
                  <div className="actions">
                    {job.resumable && <button className="primary" title="Continue from the last saved epoch with the same settings"
                      onClick={() => submit({ ...base, initialization: "resume", checkpoint_job_id: job.job_id })}>Resume</button>}
                    <button title="Run the same settings again from the start" onClick={() => submit(base)}>Retry from start</button>
                    <button title="Open the settings in New training to change them" onClick={() => onNewExperiment(base)}>New experiment…</button>
                    <button onClick={() => onOpenJob(job.job_id)}>Details</button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <div className="card">
        <h2>Uploads not in MLflow yet</h2>
        <p className="muted small">Training results are safe locally; uploading is separate. Next automatic try in
          {" "}{Math.round(report.uploader.next_check_seconds)} s{report.uploader.last_run ? ` (last ${formatTime(report.uploader.last_run)})` : ""}.</p>
        {report.uploads.length === 0 ? <p className="muted">Everything is uploaded.</p> : (
          <ul className="plain">
            {report.uploads.map((job) => (
              <li key={job.job_id} className="recovery">
                <code>{job.job_id.slice(0, 8)}</code> <span className="muted">training {job.execution_state} · upload {job.tracking_state}</span>
                {job.tracking_error && <div className="error small">{job.tracking_error.message}</div>}
                <div className="actions"><button onClick={() => api.retrySync(job.job_id).then(load, (e) => setMessage(describeError(e)))}>
                  Retry upload now</button></div>
              </li>
            ))}
          </ul>
        )}
        <h2>Studies with remaining trials</h2>
        {report.studies.length === 0 ? <p className="muted">None.</p> : (
          <ul className="plain">
            {report.studies.map((s) => (
              <li key={s.study_id} className="recovery">
                <strong>{s.name}</strong> <span className="muted">{s.state} · {s.trials_started} / {s.planned_trials} trials
                  {s.reason ? ` · ${s.reason}` : ""}</span>
                <div className="actions"><button onClick={() => api.resumeStudy(s.study_id).then(load, (e) => setMessage(describeError(e)))}>
                  Continue study</button></div>
              </li>
            ))}
          </ul>
        )}
        {report.holds.length > 0 && (<>
          <h2>Recovery holds</h2>
          <p className="muted small">Foxtrot could not prove these jobs' processes stopped, so new training waits.
            The worker re-checks automatically when idle and releases the hold once the processes are gone.</p>
          <ul className="plain">{report.holds.map((job) => <li key={job.job_id}><code>{job.job_id.slice(0, 8)}</code>
            {" "}<span className="muted">{job.execution_state}</span></li>)}</ul>
        </>)}
        {message && <p className="error">{message}</p>}
      </div>
    </section>
  );
}
