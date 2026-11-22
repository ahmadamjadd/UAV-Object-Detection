import { useEffect, useState } from "react";
import { useShared } from "../App";
import { api, Dataset, describeError } from "../api";
import { Banner, DeviceField, Empty, ErrorText, go, score, shortId, StatePill, SwitchField, Tip, usePolling } from "../ui";

// One searched setting as typed in the form.
type Param = { name: string; type: "int" | "float" | "categorical"; low: string; high: string; step: string; log: boolean;
  choices: string; conditionParam: string; conditionValues: string };
// The best trial of a study.
type Best = { number: number; value: number; model_id: string; job_id: string; checkpoint_sha256: string | null } | null;

const NEW_PARAM: Param = { name: "lr0", type: "float", low: "0.0001", high: "0.01", step: "", log: true, choices: "", conditionParam: "", conditionValues: "" };

// "8, 16" -> [8, 16]; "SGD, AdamW" -> ["SGD", "AdamW"]; "true" -> true.
function parseList(text: string): (string | number | boolean)[] {
  return text.split(",").map((v) => v.trim()).filter(Boolean)
    .map((v) => (v === "true" ? true : v === "false" ? false : Number.isNaN(Number(v)) ? v : Number(v)));
}

// Tuning studies: list on the left; the selected study or the New study form on the right.
export function StudiesView({ studyId }: { studyId: string | null }) {
  const [studies, setStudies] = useState<any[] | null>(null);
  usePolling(() => { api.studies().then(setStudies, () => setStudies([])); }, 3000);
  const open = (id: string) => go("train", { mode: "study", study: id });
  return (
    <div className="grid g2r">
      <div className="card" style={{ alignSelf: "start" }}>
        <h3>Studies<button className="r btn acc sm" onClick={() => open("new")}>+ New study</button></h3>
        <p className="mute small" style={{ marginTop: 0 }}>Each study runs several training trials on the same data with the same evaluation, and picks the best.</p>
        {studies && studies.length === 0 ? <Empty title="No studies yet">Compare settings or models fairly across many trials.</Empty> : (
          <div className="scroll"><table>
            <thead><tr><th>Study</th><th>State</th><th className="num">Trials</th><th className="num">Best</th></tr></thead>
            <tbody>{(studies ?? []).map((s) => (
              <tr key={s.study_id} className={`click ${s.study_id === studyId ? "sel" : ""}`} onClick={() => open(s.study_id)}>
                <td className="wrapc">{s.name}</td><td><StatePill state={s.state} /></td>
                <td className="num">{s.trials_started} / {s.planned_trials}</td><td className="num">{score(s.best?.value)}</td>
              </tr>))}</tbody>
          </table></div>
        )}
      </div>
      {studyId === "new" ? <StudyForm onCreated={open} />
        : studyId ? <StudyDetail key={studyId} studyId={studyId} onOpen={open} />
          : <div className="card"><Empty title="Tuning studies">Select a study, or create one with “New study”.</Empty></div>}
    </div>
  );
}

// The New study form; "Check plan" shows the number of trials before anything starts.
function StudyForm({ onCreated }: { onCreated: (id: string) => void }) {
  const { capabilities } = useShared();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [options, setOptions] = useState<{ training: string[]; recipe: string[]; objectives: string[] } | null>(null);
  const [source, setSource] = useState<"dataset" | "recipe">("dataset");
  const [v, setV] = useState<Record<string, any>>({
    name: "study-1", dataset_id: "", backgrounds: "", images_per_class: 50, canvas_size: 320, validation_share: 0.2,
    model_ids: ["yolov8n"], model_budget: "equal", initialization: "pretrained", sampler: "random", trials_per_model: 4,
    time_limit_minutes: "", max_epochs_per_trial: 50, objective: "map50_95", failure_policy: "continue", pruning: false, seed: 42,
    base_training: '{"epochs": 3, "image_size": 320, "optimizer": "SGD"}', tracking_destination_id: "", device: "cpu",
  });
  const [params, setParams] = useState<Param[]>([{ ...NEW_PARAM }]);
  const [plan, setPlan] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api.datasets().then((all) => {
      const training = all.filter((d) => d.role === "training");
      setDatasets(training);
      setV((old) => ({ ...old, dataset_id: old.dataset_id || training[0]?.dataset_id || "" }));
    });
    api.studyOptions().then(setOptions, () => undefined);
  }, []);
  useEffect(() => { setV((old) => ({ ...old, tracking_destination_id: old.tracking_destination_id || capabilities?.tracking_destinations[0] || "local" })); }, [capabilities]);
  const set = (name: string, value: unknown) => setV((old) => ({ ...old, [name]: value }));
  const setParam = (i: number, change: Partial<Param>) => setParams((all) => all.map((p, n) => (n === i ? { ...p, ...change } : p)));

  // The request exactly as the backend validates it.
  function definition(): Record<string, unknown> {
    const space = params.filter((p) => p.name).map((p) => ({
      name: p.name, type: p.type,
      ...(p.type === "categorical" ? { choices: parseList(p.choices) } : { low: Number(p.low), high: Number(p.high), log: p.log, ...(p.step ? { step: Number(p.step) } : {}) }),
      ...(p.conditionParam ? { condition: { param: p.conditionParam, values: parseList(p.conditionValues) } } : {}),
    }));
    return {
      name: v.name, model_ids: v.model_ids, model_budget: v.model_budget, initialization: v.initialization,
      tracking_destination_id: v.tracking_destination_id, sampler: v.sampler, trials_per_model: Number(v.trials_per_model),
      seed: Number(v.seed), max_epochs_per_trial: Number(v.max_epochs_per_trial), objective: v.objective,
      failure_policy: v.failure_policy, pruning: v.pruning, search_space: space,
      base_training: { ...JSON.parse(v.base_training || "{}"), ...(v.device !== "cpu" ? { device: v.device } : {}) },
      ...(v.time_limit_minutes ? { time_limit_minutes: Number(v.time_limit_minutes) } : {}),
      ...(source === "dataset" ? { dataset_id: v.dataset_id } : {
        recipe: { backgrounds: v.backgrounds, recipe: { images_per_class: Number(v.images_per_class), canvas_size: Number(v.canvas_size) } },
        validation_share: Number(v.validation_share) }),
    };
  }

  // Validate and count trials (and start the study when asked).
  async function check(start: boolean) {
    setError(""); setBusy(true);
    try {
      const body = definition();
      const result = start ? await api.createStudy(body) : await api.planStudy(body);
      const models = (v.model_ids as string[]).length;
      setPlan(v.sampler === "grid" ? `${result.planned_trials} training trials (every grid combination${models > 1 && v.model_budget === "equal" ? ` for each of ${models} models` : ""})`
        : `${result.planned_trials} training trials at most${models > 1 ? (v.model_budget === "equal" ? ` (${models} models × ${v.trials_per_model} each)` : " (combined search across models)") : ""}`
          + (v.time_limit_minutes ? `, stopping after ${v.time_limit_minutes} min` : ""));
      if (start && "study_id" in result) onCreated(result.study_id as string);
    } catch (e) { setError(e instanceof SyntaxError ? `Fixed settings: ${e.message}` : describeError(e)); } finally { setBusy(false); }
  }

  const searchable = [...(options?.training ?? []), ...(source === "recipe" ? options?.recipe ?? [] : [])];
  const field = (name: string, label: string, type = "number", help?: string) => (
    <label className="f">{label}<input type={type} value={v[name]} onChange={(e) => set(name, e.target.value)} />{help && <span className="h">{help}</span>}</label>);
  return (
    <div className="card">
      <h3>New study</h3>
      {field("name", "Name", "text")}
      <label className="f">Data<select value={source} onChange={(e) => setSource(e.target.value as "dataset" | "recipe")}>
        <option value="dataset">A fixed training dataset — tune training settings</option>
        <option value="recipe">Tune the dataset recipe — training images regenerated per trial, validation frozen</option></select></label>
      {source === "dataset" ? (
        <label className="f">Dataset<select value={v.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
          {datasets.map((d) => <option key={d.dataset_id} value={d.dataset_id}>{d.name ? `${d.name} · ${d.dataset_id}` : d.dataset_id}</option>)}</select><span className="h">Its validation split is the objective; final test data is never used.</span></label>
      ) : (
        <div className="fg">{field("backgrounds", "Backgrounds folder", "text")}{field("images_per_class", "Images per class")}
          {field("canvas_size", "Canvas size")}{field("validation_share", "Validation share of backgrounds", "number", "Kept apart and frozen for every trial.")}</div>
      )}
      <label className="f">Models<select multiple value={v.model_ids} onChange={(e) => set("model_ids", [...e.target.selectedOptions].map((o) => o.value))}>
        {(capabilities?.model_details ?? []).map((m) => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}
      </select><span className="h">Ctrl/Shift-click to compare several models.</span></label>
      {(v.model_ids as string[]).length > 1 && (
        <label className="f">Model comparison<select value={v.model_budget} onChange={(e) => set("model_budget", e.target.value)}>
          <option value="equal">Equal budget per model (recommended for a fair comparison)</option>
          <option value="combined">Combined search (model sampled with the other settings)</option></select></label>)}
      <div className="fg3">
        <label className="f">Search<select value={v.sampler} onChange={(e) => set("sampler", e.target.value)}>
          <option value="grid">Grid (every combination)</option><option value="random">Random</option><option value="tpe">Adaptive (TPE)</option></select></label>
        {v.sampler !== "grid" ? field("trials_per_model", "Trials per model") : <div />}
        {field("seed", "Search seed")}
        <label className="f"><Tip text="The validation score each trial is judged by; higher is better.">Objective</Tip>
          <select value={v.objective} onChange={(e) => set("objective", e.target.value)}>{(options?.objectives ?? ["map50_95"]).map((o) => <option key={o}>{o}</option>)}</select></label>
        {field("max_epochs_per_trial", "Max epochs per trial")}
        {field("time_limit_minutes", "Time limit (minutes)", "number", "Optional.")}
        <label className="f">Start from<select value={v.initialization} onChange={(e) => set("initialization", e.target.value)}>
          <option value="pretrained">Pretrained</option><option value="scratch">Scratch</option></select></label>
        <label className="f">If a trial fails<select value={v.failure_policy} onChange={(e) => set("failure_policy", e.target.value)}>
          <option value="continue">Continue</option><option value="stop">Stop the study</option></select></label>
        {field("tracking_destination_id", "Tracking destination", "text")}
        <DeviceField value={v.device} onChange={(device) => set("device", device)} />
      </div>
      <SwitchField label="Pruning" help="Stop trials early whose scores fall below the median (random/TPE, within one model)." checked={v.pruning} onChange={(b) => set("pruning", b)} />
      <label className="f">Fixed settings for every trial (JSON)<textarea className="code" rows={3} value={v.base_training} onChange={(e) => set("base_training", e.target.value)} /></label>
      <b className="small">Search space</b>
      <div style={{ marginTop: 8 }}>{params.map((p, i) => (
        <div key={i} className="param">
          <div className="fg3">
            <label className="f">Parameter<select value={p.name} onChange={(e) => setParam(i, { name: e.target.value })}>
              {!searchable.includes(p.name) && <option value={p.name}>{p.name || "choose…"}</option>}
              {searchable.map((n) => <option key={n}>{n}</option>)}</select></label>
            <label className="f">Type<select value={p.type} onChange={(e) => setParam(i, { type: e.target.value as Param["type"] })}>
              <option value="float">Decimal range</option><option value="int">Whole-number range</option><option value="categorical">Choices</option></select></label>
            {p.type === "categorical"
              ? <label className="f">Choices<input type="text" value={p.choices} placeholder="e.g. 8, 16, 32" onChange={(e) => setParam(i, { choices: e.target.value })} /></label>
              : <label className="swl" style={{ marginTop: 22 }}><div>Logarithmic<span>For learning rates.</span></div>
                  <input type="checkbox" checked={p.log} onChange={(e) => setParam(i, { log: e.target.checked })} /></label>}
            {p.type !== "categorical" && <>
              <label className="f">Low<input type="text" value={p.low} onChange={(e) => setParam(i, { low: e.target.value })} /></label>
              <label className="f">High<input type="text" value={p.high} onChange={(e) => setParam(i, { high: e.target.value })} /></label>
              <label className="f">Step<input type="text" value={p.step} placeholder="none" onChange={(e) => setParam(i, { step: e.target.value })} /></label></>}
            <label className="f">Only when (optional)<input type="text" value={p.conditionParam} placeholder="e.g. optimizer" onChange={(e) => setParam(i, { conditionParam: e.target.value })} /></label>
            <label className="f">is one of<input type="text" value={p.conditionValues} placeholder="e.g. SGD, AdamW" onChange={(e) => setParam(i, { conditionValues: e.target.value })} /></label>
            <div className="row" style={{ alignItems: "end", marginBottom: 12 }}><button className="btn ghost sm" type="button" onClick={() => setParams((all) => all.filter((_, n) => n !== i))}>Remove</button></div>
          </div>
        </div>
      ))}</div>
      <button className="btn sm" type="button" onClick={() => setParams((all) => [...all, { ...NEW_PARAM, name: "", log: false }])}>+ Add parameter</button>
      <div className="row" style={{ marginTop: 16 }}>
        <button className="btn" disabled={busy} onClick={() => check(false)}>Check plan</button>
        <button className="btn acc" disabled={busy} onClick={() => check(true)}>Start study</button>
        {plan && <span className="mute small">{plan}</span>}
      </div>
      <ErrorText message={error} />
    </div>
  );
}

// One study: progress, best results, every trial, stop / resume / repeat.
function StudyDetail({ studyId, onOpen }: { studyId: string; onOpen: (id: string) => void }) {
  const [study, setStudy] = useState<any>(null);
  const [message, setMessage] = useState("");
  const [seeds, setSeeds] = useState("1, 2, 3");
  const load = () => api.study(studyId).then(setStudy, (e) => setMessage(describeError(e)));
  usePolling(load, 2500);
  if (!study) return <div className="card"><p className="mute">{message || "Loading…"}</p></div>;
  const { summary, state, spec } = study;
  const act = (action: Promise<unknown>) => action.then(load, (e) => setMessage(describeError(e)));
  const running = state.state === "running" || state.state === "stopping";
  const parent = Object.values(study.mlflow as Record<string, { url: string | null }>)[0];
  const best: Best = summary.best;
  return (
    <div className="card">
      <h3>{spec.name} <StatePill state={state.state} />
        {parent?.url && <a className="r btn ghost sm" href={parent.url} target="_blank" rel="noreferrer">Study in MLflow ↗</a>}</h3>
      <p className="mute small" style={{ marginTop: 0 }}>{summary.trials_started} / {summary.planned_trials} trials · {spec.sampler === "tpe" ? "adaptive (TPE)" : spec.sampler} search · {summary.comparison} · objective {spec.objective} (validation){state.reason ? ` · ${state.reason}` : ""}</p>
      <div className="kv" style={{ marginTop: 0 }}>
        {(["complete", "failed", "pruned", "cancelled"] as const).map((k) => <div key={k}><b>{summary.counts[k]}</b><span>{k}</span></div>)}
      </div>
      <p className="mute tiny">Total trial time {Math.round(summary.trial_seconds_total)} s</p>
      <div className="row" style={{ margin: "10px 0" }}>
        {running && <button className="btn" onClick={() => act(api.stopStudy(studyId, "finish"))}>Stop (let the active trial finish)</button>}
        {running && <button className="btn" onClick={() => act(api.stopStudy(studyId, "cancel"))}>Stop now</button>}
        {!running && summary.trials_started < summary.planned_trials && <button className="btn acc" onClick={() => act(api.resumeStudy(studyId))}>Resume</button>}
      </div>
      {best && <Banner kind="acc" icon="★">Best: trial {best.number} · {best.model_id} · <b>{best.value.toFixed(4)}</b>
        <span className="mono"> · checkpoint {best.checkpoint_sha256?.slice(0, 12)}…</span>
        <button className="btn sm" style={{ marginLeft: 8 }} onClick={() => go("runs", { job: best.job_id })}>Open run</button></Banner>}
      {Object.keys(summary.best_per_model).length > 1 && (
        <table style={{ marginBottom: 12 }}><thead><tr><th>Model</th><th>Best trial</th><th className="num">Score</th></tr></thead><tbody>
          {Object.entries(summary.best_per_model as Record<string, Best>).map(([model, b]) => (
            <tr key={model}><td>{model}</td><td>{b ? `trial ${b.number}` : "no complete trial"}</td><td className="num">{score(b?.value, 4)}</td></tr>))}
        </tbody></table>)}
      {summary.repeat_groups?.map((g: any) => (
        <div key={g.configuration} className="att"><b className="small">Repeated across seeds</b>
          <p className="mute small" style={{ margin: "6px 0 0" }}>{g.values.map((x: any) => `seed ${x.seed} → ${score(x.value, 4)}`).join(" · ")}</p>
          <p style={{ margin: "6px 0 0" }}>mean <b>{score(g.mean, 4)}</b>{g.stdev !== null ? ` ± ${g.stdev.toFixed(4)}` : ""}</p></div>))}
      <div className="scroll"><table>
        <thead><tr><th>#</th><th>Model</th><th>Values</th><th>State</th><th className="num">Score</th><th></th></tr></thead>
        <tbody>{study.trials.map((t: any) => (
          <tr key={t.number} className={best?.number === t.number ? "sel" : ""}>
            <td>{t.number}</td><td>{t.model_id}</td>
            <td className="mute wrapc">{Object.entries(t.params).map(([k, x]) => `${k}=${typeof x === "number" ? +x.toPrecision(4) : x}`).join(", ")}</td>
            <td className="wrapc"><StatePill state={t.state} />{t.error && <div className="err tiny">{t.error}</div>}</td>
            <td className="num">{score(t.value, 4)}</td>
            <td>{t.job_id && <button className="btn ghost sm" onClick={() => go("runs", { job: t.job_id })}>{shortId(t.job_id, 6)}</button>}</td>
          </tr>))}</tbody>
      </table></div>
      {spec.sampler !== "repeat" && !spec.recipe && summary.counts.complete > 0 && (
        <div className="row" style={{ marginTop: 14 }}>
          <label className="f" style={{ margin: 0, flex: 1 }}>Repeat the best trial with seeds<input type="text" value={seeds} onChange={(e) => setSeeds(e.target.value)} /></label>
          <button className="btn" style={{ alignSelf: "end" }} onClick={() => api.repeatStudy(studyId, { top: 1, seeds: parseList(seeds) as number[] })
            .then((r) => onOpen(r.study_id), (e) => setMessage(describeError(e)))}>Repeat</button>
        </div>)}
      <ErrorText message={message} />
    </div>
  );
}
