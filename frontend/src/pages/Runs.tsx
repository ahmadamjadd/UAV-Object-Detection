import { useEffect, useRef, useState } from "react";
import { useShared } from "../App";
import { api, describeError, JobDetail, JobEvent, JobView, TERMINAL } from "../api";
import { describeEvent, progressOf, useLiveJob } from "../live";
import {
  Banner, displayName, Empty, ErrorText, go, localTime, modelName, Named, Progress, RenameButton, score, shortId,
  StatePill, Tabs, Tip, useDatasetNames, usePolling,
} from "../ui";

// Runs: every training and evaluation job, and anything that needs attention.
export function RunsPage({ params }: { params: URLSearchParams }) {
  const { attention } = useShared();
  const tab = params.get("tab") === "attention" ? "attention" : "all";
  return (
    <>
      <h2>Runs</h2>
      <p className="sub">Every training and evaluation job, and anything that needs attention.</p>
      <Tabs value={tab} onChange={(t) => go("runs", t === "attention" ? { tab: "attention" } : {})}
        options={[["all", "All runs"], ["attention", `Needs attention${attention ? ` · ${attention}` : ""}`]]} />
      {tab === "all" ? <AllRuns selected={params.get("job")} /> : <Attention />}
    </>
  );
}

// The runs table with the selected run's live details beside it.
function AllRuns({ selected }: { selected: string | null }) {
  const { capabilities } = useShared();
  const [jobs, setJobs] = useState<JobView[] | null>(null);
  const [error, setError] = useState("");
  usePolling(() => { api.jobs().then((j) => { setJobs(j); setError(""); }, (e) => setError(describeError(e))); }, 2000);
  const current = selected ?? jobs?.[0]?.job_id ?? null;
  const live = useLiveJob(current);
  const datasetNames = useDatasetNames();
  if (jobs && jobs.length === 0) return <div className="card"><Empty title="No runs yet">Start a training run and follow it here.
    <div className="row" style={{ justifyContent: "center", marginTop: 12 }}><button className="btn acc" onClick={() => go("train")}>Train a model</button></div></Empty></div>;
  return (
    <div className="grid g2">
      <div className="card scroll" style={{ padding: "8px 10px", alignSelf: "start" }}>
        <ErrorText message={error} />
        <table>
          <thead><tr><th>Run</th><th>Kind</th><th>Model</th><th>State</th><th>Upload</th><th>Created</th></tr></thead>
          <tbody>{(jobs ?? []).map((j) => (
            <tr key={j.job_id} className={`click ${j.job_id === current ? "sel" : ""}`} onClick={() => go("runs", { job: j.job_id })}>
              <td className="wrapc" style={{ maxWidth: 220 }}><Named name={j.name} id={j.job_id} short /></td>
              <td>{j.kind ?? "—"}{j.study && <div className="mute tiny">trial {j.study.trial_number}</div>}</td>
              <td>{modelName(j.model_id, capabilities?.model_details)}<div className="mute tiny">{(j.dataset_id && datasetNames[j.dataset_id]) || j.dataset_id}</div></td>
              <td><StatePill state={j.execution_state} /></td>
              <td>{TERMINAL.has(j.execution_state) && j.execution_state === "finished" ? <StatePill state={j.tracking_state} /> : <span className="mute">—</span>}</td>
              <td className="mute">{localTime(j.created_at)}</td>
            </tr>))}</tbody>
        </table>
      </div>
      {current ? <RunDetail key={current} jobId={current} job={live.job} events={live.events} refresh={live.refresh} datasetNames={datasetNames} />
        : <div className="card"><p className="mute">Select a run.</p></div>}
      {current && live.job && <div style={{ gridColumn: "1 / -1" }}><Terminal job={live.job} events={live.events} /></div>}
    </div>
  );
}

// Clean terminal text: remove colour/cursor codes and keep only the last redraw of a progress bar.
function cleanLine(text: string): string[] {
  return text.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\x1b\][^\x07]*\x07/g, "").split("\n")
    .map((line) => line.split("\r").filter((part) => part.trim()).pop() ?? "")
    .filter((line) => line.trim());
}

// The run's output in a large terminal panel: training output plus Foxtrot's own steps.
function Terminal({ job, events }: { job: JobDetail; events: JobEvent[] }) {
  const [onlyOutput, setOnlyOutput] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);  // follow new lines unless the member scrolled up to read
  const lines = events.flatMap((e) => {
    const time = new Date(e.timestamp).toLocaleTimeString();
    if (e.event_type === "log" && typeof e.payload.text === "string") {
      return cleanLine(e.payload.text).map((text, n) => ({ key: `${e.sequence}-${n}`, time, text, kind: e.payload.channel === "stderr" ? "err" : "out" }));
    }
    if (onlyOutput) return [];
    return [{ key: String(e.sequence), time, text: `» ${describeEvent(e)}`, kind: e.event_type === "progress" ? "epoch" : "info" }];
  });
  useEffect(() => { if (stick.current && box.current) box.current.scrollTop = box.current.scrollHeight; }, [lines.length]);
  const active = !TERMINAL.has(job.execution_state);
  return (
    <div className="card terminal-card">
      <h3>Training output{active && <span className="pill p-acc">live</span>}
        <span className="mute tiny">{displayName(job, job.job_id, true)}</span>
        <label className="r row mute small" style={{ gap: 6 }}><input type="checkbox" checked={onlyOutput} onChange={(e) => setOnlyOutput(e.target.checked)} />Only training output</label>
        <button className="btn ghost sm" onClick={() => { stick.current = true; if (box.current) box.current.scrollTop = box.current.scrollHeight; }}>Jump to latest</button></h3>
      <div className="terminal" ref={box} role="log" aria-live="off"
        onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40; }}>
        {lines.length === 0 ? <div className="mute">Waiting for output…</div> : lines.map((line) => (
          <div key={line.key} className={`t-${line.kind}`}><span className="t-time">{line.time}</span>{line.text}</div>))}
      </div>
      <p className="mute tiny" style={{ margin: "8px 0 0" }}>Shows the most recent output lines; the complete logs stay in the run's folder.</p>
    </div>
  );
}

// Live details of one run: name, progress, results, upload state.
function RunDetail({ jobId, job, events, refresh, datasetNames }: {
  jobId: string; job: JobDetail | null; events: JobEvent[]; refresh: () => void; datasetNames: Record<string, string>;
}) {
  const { capabilities } = useShared();
  const [message, setMessage] = useState("");
  if (!job) return <div className="card"><p className="mute">Loading…</p></div>;
  const progress = progressOf(events);
  const active = !TERMINAL.has(job.execution_state);
  const resolved = job.resolved as Record<string, any> | null;
  const evaluation = job.result?.evaluation as (typeof job.result extends null ? never : any);
  return (
    <div className="card" style={{ alignSelf: "start" }}>
      <h3>{job.name ? <span style={{ fontSize: 16 }}>{job.name}</span> : <span className="mono">{shortId(job.job_id)}</span>}<StatePill state={job.execution_state} />
        {job.cancel_requested && active && <span className="mute small">cancelling…</span>}
        {active && !job.cancel_requested && <button className="r btn" onClick={() => api.cancel(jobId).then(refresh, (e) => setMessage(describeError(e)))}>Cancel</button>}</h3>
      <div className="row" style={{ margin: "-4px 0 8px" }}>{job.name && <span className="mono mute tiny">{job.job_id}</span>}
        <RenameButton name={job.name} label="Rename run" onSave={(name) => api.renameJob(jobId, name).then(refresh)} /></div>
      <p className="mute" style={{ marginTop: 0 }}>{resolved?.kind === "evaluate"
        ? <>Scores the model from job <span className="mono">{shortId(resolved.checkpoint_job_id)}</span> on dataset {datasetNames[resolved.dataset_id] ?? resolved.dataset_id}.</>
        : <>{modelName(resolved?.model_id, capabilities?.model_details)} on {datasetNames[resolved?.dataset_id] ?? resolved?.dataset_id}{resolved?.initialization && resolved.initialization !== "pretrained"
          ? ` · started from ${resolved.initialization === "resume" ? "the last epoch of" : resolved.initialization === "checkpoint" ? "the checkpoint of" : "random weights"}${resolved.checkpoint_job_id ? ` ${shortId(resolved.checkpoint_job_id)}` : ""}` : ""}
          {resolved?.study && <> · study trial {resolved.study.trial_number}</>}</>}</p>
      {progress.total > 0 && (active || job.execution_state !== "finished") && (<>
        <div className="row between"><b>Epoch {progress.epoch} / {progress.total}</b>
          <span className="mute">{progress.map !== null ? `mAP50-95 ${score(progress.map)}` : ""}{progress.minutesLeft !== null && active ? (progress.minutesLeft < 1 ? " · under a minute left" : ` · about ${progress.minutesLeft} min left`) : ""}</span></div>
        <div style={{ margin: "8px 0 4px" }}><Progress done={progress.epoch} total={progress.total} /></div></>)}
      {job.error && <Banner kind="bad" icon={job.error.code}>{job.error.message}</Banner>}
      {job.execution_state === "finished" && resolved?.kind !== "evaluate" && progress.total > 0 && progress.epoch < progress.total && (
        <Banner kind="info" icon="⏹">Stopped early at epoch {progress.epoch} of {progress.total}: the validation score (mAP50-95) did not improve
          for {resolved?.training?.patience ?? "the patience"} epochs. The best checkpoint was kept.</Banner>)}
      {resolved?.training?.device && resolved.training.device !== "cpu" && <p className="mute tiny" style={{ marginTop: 0 }}>Ran on {resolved.training.device}</p>}
      {evaluation && (
        <div className="kv">{([["Precision", evaluation.precision, "Of the boxes the model drew, the share that were right."],
          ["Recall", evaluation.recall, "Of the real objects, the share the model found."],
          ["mAP50", evaluation.map50, "Average precision when a box must overlap the real one by at least 50%."],
          ["mAP50-95", evaluation.map50_95, "Average over stricter overlaps (50–95%); the main score."]] as [string, number, string][]).map(([label, value, tip]) =>
          <div key={label}><b>{score(value)}</b><span><Tip text={tip}>{label}</Tip></span></div>)}</div>
      )}
      {evaluation?.per_class_map50_95 && Object.keys(evaluation.per_class_map50_95).length > 0 && (
        <p className="mute small">Per class (mAP50-95): {Object.entries(evaluation.per_class_map50_95 as Record<string, number>).map(([k, v]) => `${k} ${score(v)}`).join(" · ")}</p>)}
      {TERMINAL.has(job.execution_state) && <UploadBox job={job} onChange={refresh} />}
      <p className="mute tiny" style={{ marginBottom: 0 }}>The full output is shown below the table.</p>
      <ErrorText message={message} />
    </div>
  );
}

// Upload to MLflow, kept visually separate from training; optional redirect once uploaded.
function UploadBox({ job, onChange }: { job: JobDetail; onChange: () => void }) {
  const synced = job.tracking_state === "synced";
  const [message, setMessage] = useState("");
  const [autoOpen, setAutoOpen] = useState(() => { try { return localStorage.getItem("foxtrot.autoOpenMlflow") === "1"; } catch { return false; } });
  const wasPending = useRef(!synced);
  useEffect(() => {  // same-tab redirect only when the upload completes while watching (no popup needed)
    if (synced && wasPending.current && autoOpen && job.tracking_run?.run_url) window.location.assign(job.tracking_run.run_url);
    wasPending.current = !synced;
  }, [synced, autoOpen, job.tracking_run?.run_url]);
  const finished = job.execution_state === "finished";
  const failedUpload = job.tracking_state === "retrying" || job.tracking_state === "failed";
  return (
    <div className="upload">
      <div className="row"><b className="small">Upload to MLflow</b><StatePill state={job.tracking_state} /></div>
      <p className="mute small" style={{ margin: "6px 0 10px" }}>
        {synced ? `${finished ? "Training finished" : `Run ${job.execution_state}`} and uploaded.`
          : failedUpload ? `Upload failed: ${job.tracking_error?.message ?? "MLflow could not be reached"}. Foxtrot keeps retrying automatically; results are safe locally.`
          : `${finished ? "Training finished" : `Run ${job.execution_state}`} — uploading to MLflow automatically, usually within a minute.`}</p>
      <div className="row">
        {synced && job.tracking_run?.run_url && <a className="btn acc" href={job.tracking_run.run_url} target="_blank" rel="noreferrer">View in MLflow ↗</a>}
        {failedUpload && <button className="btn" onClick={() => api.retrySync(job.job_id).then(() => { setMessage("Uploading…"); onChange(); }, (e) => setMessage(describeError(e)))}>Retry upload</button>}
        {message && !synced && <span className="mute small">{message}</span>}
      </div>
      <label className="row mute small" style={{ marginTop: 10, gap: 8, flexWrap: "nowrap", alignItems: "flex-start" }}>
        <input type="checkbox" checked={autoOpen} onChange={(e) => {
          setAutoOpen(e.target.checked); try { localStorage.setItem("foxtrot.autoOpenMlflow", e.target.checked ? "1" : ""); } catch { /* storage unavailable */ }
        }} /><span style={{ flex: 1 }}>Open the MLflow run automatically when its upload finishes</span></label>
    </div>
  );
}

// Needs attention: stopped training, pending uploads, unfinished studies, recovery holds.
function Attention() {
  const { setDraft, refreshAttention, capabilities } = useShared();
  const [report, setReport] = useState<any>(null);
  const [message, setMessage] = useState("");
  const load = () => api.recovery().then((r) => { setReport(r); refreshAttention(); }, (e) => setMessage(describeError(e)));
  usePolling(load, 4000);

  // Create a new job from an earlier request, then show it.
  async function submit(request: Record<string, unknown>, name?: string | null) {
    setMessage("");
    try { go("runs", { job: (await api.submit(crypto.randomUUID(), request, name)).job_id }); } catch (e) { setMessage(describeError(e)); }
  }

  if (!report) return <p className="mute">{message || "Loading…"}</p>;
  const nothing = !report.interrupted.length && !report.uploads.length && !report.studies.length && !report.holds.length;
  return (
    <>
      <Banner kind="info"><b>Resume</b> continues the same run from its last saved epoch. <b>Retry from start</b> reruns the same settings from epoch 0.
        <b> New experiment</b> lets you change settings first. <b>Continue study</b> runs a study's remaining trials.</Banner>
      <ErrorText message={message} />
      {nothing && <div className="card"><Empty title="Nothing needs attention">Stopped runs, pending uploads and unfinished studies show up here.</Empty></div>}
      {report.interrupted.length > 0 && (
        <div className="card" style={{ marginBottom: 14 }}><h3>Stopped training</h3>
          {report.interrupted.map((job: any) => {
            const { study: _study, ...base } = job.request ?? {};
            return (
              <div key={job.job_id} className="att">
                <div className="row"><b>{job.name ?? `${modelName(job.model_id, capabilities?.model_details)} on ${job.dataset_id}`}</b><span className="mono mute tiny">{shortId(job.job_id)}</span>
                  <StatePill state={job.execution_state} /><span className="mute small r">{localTime(job.finished_at)}</span></div>
                <p className="mute small" style={{ margin: "6px 0 0" }}>{job.error ? `${job.error.code}: ${job.error.message}` : "Stopped before finishing."}
                  {!job.resumable && job.resume_reason ? ` Can't resume: ${job.resume_reason}` : ""}</p>
                <div className="row">
                  {job.resumable && <button className="btn acc" onClick={() => submit({ ...base, initialization: "resume", checkpoint_job_id: job.job_id }, job.name ? `${job.name} (resumed)` : null)}>Resume from last saved epoch</button>}
                  <button className="btn" onClick={() => submit(base, job.name ? `${job.name} (retry)` : null)}>Retry from start</button>
                  <button className="btn" onClick={() => { setDraft({ ...base, ...(job.name ? { __name: `${job.name} (changed)` } : {}) }); go("train"); }}>New experiment…</button>
                  <button className="btn ghost" onClick={() => go("runs", { job: job.job_id })}>Details</button>
                </div>
              </div>);
          })}
        </div>)}
      {report.uploads.length > 0 && (
        <div className="card" style={{ marginBottom: 14 }}><h3>Uploads that failed</h3>
          <p className="mute small" style={{ marginTop: 0 }}>These could not reach MLflow. Results are safe locally. Next automatic retry in {Math.round(report.uploader.next_check_seconds)} s.</p>
          {report.uploads.map((job: any) => (
            <div key={job.job_id} className="att">
              <div className="row"><b>{job.name ?? `${job.kind === "evaluate" ? "Evaluation" : "Training"} of ${modelName(job.model_id, capabilities?.model_details)}`}</b><span className="mono mute tiny">{shortId(job.job_id)}</span>
                <StatePill state={job.tracking_state} /></div>
              <p className="mute small" style={{ margin: "6px 0 0" }}>{job.tracking_error?.message ?? "Waiting for the next automatic upload."}</p>
              <div className="row"><button className="btn" onClick={() => api.retrySync(job.job_id).then(load, (e) => setMessage(describeError(e)))}>Retry upload now</button></div>
            </div>))}
        </div>)}
      {report.studies.length > 0 && (
        <div className="card" style={{ marginBottom: 14 }}><h3>Studies with remaining trials</h3>
          {report.studies.map((s: any) => (
            <div key={s.study_id} className="att">
              <div className="row"><b>{s.name}</b><StatePill state={s.state} /></div>
              <p className="mute small" style={{ margin: "6px 0 0" }}>{s.trials_started} of {s.planned_trials} trials done{s.reason ? ` · ${s.reason}` : ""}</p>
              <div className="row"><button className="btn acc" onClick={() => api.resumeStudy(s.study_id).then(load, (e) => setMessage(describeError(e)))}>Continue study</button>
                <button className="btn ghost" onClick={() => go("train", { mode: "study", study: s.study_id })}>Open study</button></div>
            </div>))}
        </div>)}
      {report.holds.length > 0 && (
        <div className="card"><h3>Recovery holds</h3>
          <p className="mute small" style={{ marginTop: 0 }}>Foxtrot could not prove these runs' processes stopped, so new training waits. The worker re-checks automatically when idle and releases the hold once they are gone — nothing to do.</p>
          <div className="list">{report.holds.map((job: any) => <div key={job.job_id}><span className="mono">{shortId(job.job_id)}</span><StatePill state={job.execution_state} /></div>)}</div>
        </div>)}
    </>
  );
}
