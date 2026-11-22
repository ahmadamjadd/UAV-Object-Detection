import { useState } from "react";
import { useShared } from "../App";
import { api, JobDetail, JobView, TERMINAL } from "../api";
import { progressOf, useLiveJob } from "../live";
import { Empty, go, localTime, modelName, Named, Progress, score, shortId, Sparkline, StatePill, Tip, UploadPill, usePolling } from "../ui";

// Home: what is happening now, what needs a look, recent results, setup status, quick actions.
export function HomePage() {
  const { capabilities, member } = useShared();
  const [jobs, setJobs] = useState<JobView[] | null>(null);
  const [recent, setRecent] = useState<JobDetail[]>([]);
  const [studies, setStudies] = useState<any[]>([]);
  const [recovery, setRecovery] = useState<any>(null);
  const [check, setCheck] = useState<{ ok: boolean; checks: { name: string; ok: boolean; detail: string }[] } | null>(null);
  const [datasetCount, setDatasetCount] = useState<number | null>(null);

  // Refresh the overview every 3 seconds; recent results need each job's detail for scores and links.
  usePolling(() => {
    api.jobs().then((list) => {
      setJobs(list);
      const finished = list.filter((j) => j.execution_state === "finished").slice(0, 5);
      Promise.all(finished.map((j) => api.job(j.job_id))).then(setRecent, () => undefined);
    }, () => setJobs([]));
    api.studies().then(setStudies, () => undefined);
    api.recovery().then(setRecovery, () => undefined);
    api.datasets().then((d) => setDatasetCount(d.length), () => undefined);
  }, 3000);
  usePolling(() => { api.systemCheck().then(setCheck, () => undefined); }, 60000);

  const active = jobs?.find((j) => !TERMINAL.has(j.execution_state) && j.execution_state !== "queued")
    ?? jobs?.find((j) => j.execution_state === "queued");
  const queued = jobs?.filter((j) => j.execution_state === "queued").length ?? 0;
  const runningStudies = studies.filter((s) => s.state === "running");
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  const attentionItems = recovery ? [
    { count: recovery.interrupted.length, pill: "p-warn", title: "Stopped training",
      detail: recovery.interrupted.filter((j: any) => j.resumable).length ? "Some can resume from their last saved epoch" : "Retry or start a new experiment" },
    { count: recovery.uploads.length, pill: "p-warn", title: "Uploads that failed",
      detail: `Next automatic retry in ${Math.round(recovery.uploader.next_check_seconds)} s` },
    { count: recovery.studies.length, pill: "p-acc", title: "Studies with remaining trials",
      detail: recovery.studies.map((s: any) => `“${s.name}” · ${s.planned_trials - s.trials_started} left`).join(", ") },
    { count: recovery.holds.length, pill: "p-mute", title: "Recovery holds", detail: "Resolved automatically when processes stop" },
  ].filter((item) => item.count > 0) : [];
  const attentionTotal = attentionItems.reduce((sum, item) => sum + item.count, 0);
  const firstRun = jobs !== null && jobs.length === 0 && datasetCount === 0;

  const summary = active ? `${active.execution_state === "queued" ? "A job is waiting to start" : "One model is training now"}`
    : "Nothing is running right now";
  return (
    <>
      <h2>{greeting}{member ? `, ${member}` : ""}</h2>
      <p className="sub">{summary}{attentionTotal ? `, and ${attentionTotal} thing${attentionTotal > 1 ? "s" : ""} need${attentionTotal > 1 ? "" : "s"} a look.` : "."}</p>

      {firstRun && (
        <div className="card" style={{ marginBottom: 16 }}>
          <h3>Get started in three steps</h3>
          <div className="steps">
            <div><b>1 · Install weights</b><span className="mute small">Settings › Pretrained weights, so training can start from a trained model.</span>
              <div style={{ marginTop: 10 }}><button className="btn sm" onClick={() => go("settings")}>Open Settings</button></div></div>
            <div><b>2 · Add a dataset</b><span className="mute small">Generate synthetic images, import a folder, or bring one from DVC.</span>
              <div style={{ marginTop: 10 }}><button className="btn sm" onClick={() => go("datasets", { add: "generate" })}>Add dataset</button></div></div>
            <div><b>3 · Train</b><span className="mute small">Pick a model and start; follow it live in Runs.</span>
              <div style={{ marginTop: 10 }}><button className="btn sm acc" onClick={() => go("train")}>Train a model</button></div></div>
          </div>
        </div>
      )}

      <div className="grid g2">
        {active ? <LiveCard job={active} queued={queued} runningStudies={runningStudies} details={capabilities?.model_details} />
          : <div className="card"><h3>Active work</h3>
              <Empty title="No training running">{queued ? `${queued} job(s) queued.` : "Start a run or a tuning study."}
                <div className="row" style={{ justifyContent: "center", marginTop: 12 }}>
                  <button className="btn acc" onClick={() => go("train")}>Train a model</button></div></Empty>
              {runningStudies.length > 0 && <p className="mute small">{runningStudies.length} tuning study running.</p>}</div>}
        <div className="card"><h3>Needs attention<button className="r btn ghost sm" onClick={() => go("runs", { tab: "attention" })}>View all</button></h3>
          {attentionItems.length === 0 ? <p className="mute" style={{ margin: 0 }}>Nothing needs attention.</p> : (
            <div className="list">{attentionItems.map((item) => (
              <div key={item.title}><span className={`pill ${item.pill}`}>{item.count}</span>
                <div style={{ flex: 1 }}>{item.title}<div className="mute small">{item.detail}</div></div></div>
            ))}</div>
          )}
        </div>
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <div className="card"><h3>Recent results</h3>
          {recent.length === 0 ? <p className="mute" style={{ margin: 0 }}>Finished runs appear here with their scores.</p> : (
            <div className="scroll"><table>
              <thead><tr><th>Job</th><th>Model</th><th className="num"><Tip text="mean Average Precision across overlap thresholds 0.50–0.95. Higher is better; 1.0 is perfect.">mAP50-95</Tip></th><th>Upload</th><th></th></tr></thead>
              <tbody>{recent.map((j) => (
                <tr key={j.job_id} className="click" onClick={() => go("runs", { job: j.job_id })}>
                  <td className="wrapc" style={{ maxWidth: 220 }}><Named name={j.name} id={j.job_id} short /></td>
                  <td>{modelName(String(j.resolved?.model_id ?? ""), capabilities?.model_details)}
                    <div className="mute tiny">{j.resolved?.kind === "evaluate" ? "evaluation" : j.resolved?.study ? "study trial" : "training"} · {String(j.resolved?.dataset_id ?? "")}</div></td>
                  <td className="num"><b>{score(j.result?.evaluation?.map50_95)}</b></td>
                  <td><UploadPill state={j.tracking_state} finished /></td>
                  <td>{j.tracking_state === "synced" && j.tracking_run?.run_url &&
                    <a className="btn ghost sm" href={j.tracking_run.run_url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>View in MLflow ↗</a>}</td>
                </tr>
              ))}</tbody>
            </table></div>
          )}
        </div>
        <div className="grid" style={{ alignContent: "start" }}>
          <div className="card"><h3>Setup status{check && <span className={`r pill ${check.ok ? "p-ok" : "p-warn"}`}>{check.ok ? "Ready" : "Needs attention"}</span>}</h3>
            <p className="mute small" style={{ margin: "0 0 10px" }}>{check ? `${check.checks.filter((c) => c.ok).length} of ${check.checks.length} checks pass.`
              + (check.checks.find((c) => !c.ok) ? ` ${check.checks.find((c) => !c.ok)!.name}: ${check.checks.find((c) => !c.ok)!.detail}.` : "") : "Checking…"}</p>
            <button className="btn ghost" onClick={() => go("settings")}>Open system check</button></div>
          <div className="card"><h3>Quick actions</h3><div className="row">
            <button className="btn" onClick={() => go("datasets", { add: "generate" })}>Add dataset</button>
            <button className="btn" onClick={() => go("train")}>Train a model</button>
            <button className="btn" onClick={() => go("train", { mode: "study" })}>Start a tuning study</button>
            <button className="btn" onClick={() => go("evaluate")}>Evaluate a model</button></div></div>
        </div>
      </div>
    </>
  );
}

// The live training card: epoch, score so far, time left, sparkline.
function LiveCard({ job, queued, runningStudies, details }: {
  job: JobView; queued: number; runningStudies: any[]; details?: { model_id: string; display_name: string }[];
}) {
  const { job: detail, events } = useLiveJob(job.job_id);
  const progress = progressOf(events);
  const state = detail?.execution_state ?? job.execution_state;
  return (
    <div className="card">
      <h3><StatePill state={state} /> {job.name ?? `${modelName(job.model_id, details)} on ${job.dataset_id ?? "—"}`}
        <button className="r btn ghost sm" onClick={() => go("runs", { job: job.job_id })}>Open run</button></h3>
      <div className="row between" style={{ alignItems: "flex-end" }}>
        <div>
          <div className="big">{progress.total ? <>Epoch {progress.epoch} <span className="mute" style={{ fontSize: 22 }}>/ {progress.total}</span></> : state}</div>
          <div className="mute" style={{ marginTop: 6 }}>{progress.minutesLeft !== null ? (progress.minutesLeft < 1 ? "Under a minute left · " : `About ${progress.minutesLeft} min left · `) : ""}job <span className="mono">{shortId(job.job_id)}</span></div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="mute tiny"><Tip text="mean Average Precision across overlap thresholds 0.50–0.95. Higher is better; 1.0 is perfect.">mAP50-95</Tip> so far</div>
          <div style={{ fontSize: 26, fontWeight: 700 }}>{score(progress.map)}</div>
        </div>
      </div>
      <div style={{ marginTop: 14 }}><Progress done={progress.epoch} total={progress.total} /></div>
      <Sparkline values={progress.history} />
      <div className="row mute small" style={{ marginTop: 8 }}>
        <span>{queued} job{queued === 1 ? "" : "s"} queued</span>·
        <span>{runningStudies.length ? runningStudies.map((s) => `study “${s.name}” (trial ${s.trials_started} of ${s.planned_trials})`).join(", ") : "no tuning study running"}</span>
      </div>
      {localTime(job.created_at) && <div className="mute tiny" style={{ marginTop: 4 }}>Started {localTime(job.created_at)}</div>}
    </div>
  );
}
