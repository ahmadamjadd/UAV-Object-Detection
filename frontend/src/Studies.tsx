import { useCallback, useEffect, useState } from "react";
import { api, Capabilities, Dataset, describeError } from "./api";

type Param = { name: string; type: "int" | "float" | "categorical"; low: string; high: string; step: string;
  log: boolean; choices: string; conditionParam: string; conditionValues: string };
type Trial = { number: number; model_id: string; state: string; params: Record<string, unknown>; value: number | null;
  error: string | null; job_id: string | null; dataset_id: string | null; intermediate: [number, number][] };
type Best = { number: number; value: number; model_id: string; job_id: string; checkpoint_sha256: string | null } | null;
type StudyDetail = {
  study_id: string; spec: Record<string, any>; record: { planned_trials: number; created_at: string };
  state: { state: string; reason: string | null }; trials: Trial[]; mlflow: Record<string, { url: string | null }>;
  summary: { counts: Record<string, number>; best: Best; best_per_model: Record<string, Best>; comparison: string;
    trial_seconds_total: number; trials_started: number; planned_trials: number;
    repeat_groups?: { configuration: string; values: { seed: number; value: number | null }[]; mean: number | null; stdev: number | null }[] };
};

const EMPTY: Param = { name: "lr0", type: "float", low: "0.0001", high: "0.01", step: "", log: true, choices: "",
  conditionParam: "", conditionValues: "" };

// "8, 16" -> [8, 16]; "SGD, AdamW" -> ["SGD", "AdamW"]; "true" -> true.
function parseList(text: string): (string | number | boolean)[] {
  return text.split(",").map((v) => v.trim()).filter(Boolean)
    .map((v) => (v === "true" ? true : v === "false" ? false : Number.isNaN(Number(v)) ? v : Number(v)));
}

// The Studies section: define a tuning study, follow its trials, compare results.
export function Studies({ capabilities, onOpenJob }: { capabilities: Capabilities | null; onOpenJob: (id: string) => void }) {
  const [studies, setStudies] = useState<{ study_id: string; name: string; state: string; planned_trials: number;
    trials_started: number; best: Best }[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const load = useCallback(() => { api.studies().then(setStudies, () => undefined); }, []);
  useEffect(() => { load(); const t = window.setInterval(load, 3000); return () => window.clearInterval(t); }, [load]);

  return (
    <section className={`grid ${creating ? "wide-right" : "wide"}`}>
      <div className="card">
        <h2>Tuning studies</h2>
        <p className="muted">A study runs several training trials with different settings (grid, random or
          adaptive TPE search) and compares them on the same data with the same evaluation.</p>
        <button className="primary" onClick={() => { setCreating(true); setSelected(null); }}>New study</button>
        {studies.length === 0 ? <p className="muted">No studies yet.</p> : (
          <table>
            <thead><tr><th>Study</th><th>State</th><th>Trials</th><th>Best</th></tr></thead>
            <tbody>
              {studies.map((s) => (
                <tr key={s.study_id} className={s.study_id === selected ? "selected" : ""}
                  onClick={() => { setSelected(s.study_id); setCreating(false); }}>
                  <td>{s.name}</td><td><span className={`state ${s.state}`}>{s.state}</span></td>
                  <td>{s.trials_started} / {s.planned_trials}</td>
                  <td>{s.best ? s.best.value.toFixed(3) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {creating ? <StudyForm capabilities={capabilities} onCreated={(id) => { setCreating(false); setSelected(id); load(); }} />
        : selected ? <StudyView key={selected} studyId={selected} onOpenJob={onOpenJob} onCreated={(id) => { setSelected(id); load(); }} />
          : <div className="card"><p className="muted">Select a study, or create one.</p></div>}
    </section>
  );
}

// The study definition form; "Check plan" shows how many trials will run before starting.
function StudyForm({ capabilities, onCreated }: { capabilities: Capabilities | null; onCreated: (id: string) => void }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [options, setOptions] = useState<{ training: string[]; recipe: string[]; objectives: string[] } | null>(null);
  const [source, setSource] = useState<"dataset" | "recipe">("dataset");
  const [values, setValues] = useState<Record<string, any>>({
    name: "study-1", dataset_id: "", backgrounds: "", images_per_class: 50, canvas_size: 320, validation_share: 0.2,
    model_ids: ["yolov8n"], model_budget: "equal", initialization: "pretrained", sampler: "random", trials_per_model: 4,
    time_limit_minutes: "", max_epochs_per_trial: 50, objective: "map50_95", failure_policy: "continue", pruning: false,
    seed: 42, base_training: '{"epochs": 3, "image_size": 320}',
    tracking_destination_id: capabilities?.tracking_destinations[0] ?? "local",
  });
  const [params, setParams] = useState<Param[]>([{ ...EMPTY }]);
  const [plan, setPlan] = useState("");
  const [error, setError] = useState("");

  // Load training datasets (real evaluation data is never trained on) and the tunable parameter names.
  useEffect(() => {
    api.datasets().then((all) => {
      const training = all.filter((d) => d.role === "training");
      setDatasets(training);
      setValues((v) => ({ ...v, dataset_id: v.dataset_id || training[0]?.dataset_id || "" }));
    });
    api.studyOptions().then(setOptions);
  }, []);

  const set = (name: string, value: unknown) => setValues((v) => ({ ...v, [name]: value }));
  const setParam = (index: number, change: Partial<Param>) =>
    setParams((all) => all.map((p, i) => (i === index ? { ...p, ...change } : p)));

  // Build the request exactly as the backend validates it.
  function definition(): Record<string, unknown> {
    const space = params.filter((p) => p.name).map((p) => ({
      name: p.name, type: p.type,
      ...(p.type === "categorical" ? { choices: parseList(p.choices) } : {
        low: Number(p.low), high: Number(p.high), log: p.log, ...(p.step ? { step: Number(p.step) } : {}) }),
      ...(p.conditionParam ? { condition: { param: p.conditionParam, values: parseList(p.conditionValues) } } : {}),
    }));
    return {
      name: values.name, model_ids: values.model_ids, model_budget: values.model_budget,
      initialization: values.initialization, tracking_destination_id: values.tracking_destination_id,
      sampler: values.sampler, trials_per_model: Number(values.trials_per_model), seed: Number(values.seed),
      max_epochs_per_trial: Number(values.max_epochs_per_trial), objective: values.objective,
      failure_policy: values.failure_policy, pruning: values.pruning, search_space: space,
      base_training: JSON.parse(values.base_training || "{}"),
      ...(values.time_limit_minutes ? { time_limit_minutes: Number(values.time_limit_minutes) } : {}),
      ...(source === "dataset" ? { dataset_id: values.dataset_id } : {
        recipe: { backgrounds: values.backgrounds, recipe: {
          images_per_class: Number(values.images_per_class), canvas_size: Number(values.canvas_size) } },
        validation_share: Number(values.validation_share) }),
    };
  }

  // Ask the backend to validate the definition and count its trials.
  async function check(start: boolean) {
    setError("");
    try {
      const body = definition();
      const result = start ? await api.createStudy(body) : await api.planStudy(body);
      const models = (values.model_ids as string[]).length;
      setPlan(values.sampler === "tpe" || values.sampler === "random"
        ? `${result.planned_trials} training trials at most${models > 1 ? (values.model_budget === "equal"
          ? ` (${models} models × ${values.trials_per_model} trials each)` : " (combined search across models)") : ""}`
          + (values.time_limit_minutes ? `, stopping after ${values.time_limit_minutes} min` : "")
        : `${result.planned_trials} training trials (every grid combination${models > 1 && values.model_budget === "equal" ? ` for each of ${models} models` : ""})`);
      if (start && "study_id" in result) onCreated(result.study_id as string);
    } catch (e) { setError(describeError(e)); }
  }

  const searchable = [...(options?.training ?? []), ...(source === "recipe" ? options?.recipe ?? [] : [])];
  const field = (name: string, label: string, type = "number") => (
    <label>{label}<input type={type} value={values[name]} onChange={(e) => set(name, e.target.value)} /></label>);
  return (
    <div className="card">
      <h2>New study</h2>
      <div className="form">
        {field("name", "Name", "text")}
        <fieldset className="choices"><legend>Data</legend>
          <label className="check"><input type="radio" checked={source === "dataset"} onChange={() => setSource("dataset")} />
            A fixed training dataset (tune training settings)</label>
          <label className="check"><input type="radio" checked={source === "recipe"} onChange={() => setSource("recipe")} />
            Tune the dataset recipe (training images regenerated per trial; validation frozen)</label>
        </fieldset>
        {source === "dataset" ? (
          <label>Dataset<select value={values.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
            {datasets.map((d) => <option key={d.dataset_id}>{d.dataset_id}</option>)}</select></label>
        ) : (
          <div className="row">
            {field("backgrounds", "Backgrounds folder", "text")}
            {field("images_per_class", "Images per class")}
            {field("canvas_size", "Canvas size")}
            {field("validation_share", "Validation share of backgrounds")}
          </div>
        )}
        <label>Models (Ctrl/Shift to choose several)<select multiple size={5} value={values.model_ids}
          onChange={(e) => set("model_ids", [...e.target.selectedOptions].map((o) => o.value))}>
          {(capabilities?.model_details ?? []).map((m) => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}
        </select></label>
        {(values.model_ids as string[]).length > 1 && (
          <label>Model comparison<select value={values.model_budget} onChange={(e) => set("model_budget", e.target.value)}>
            <option value="equal">Equal budget per model (recommended for a fair comparison)</option>
            <option value="combined">Combined search (model sampled with the other settings)</option></select></label>
        )}
        <div className="row">
          <label>Search<select value={values.sampler} onChange={(e) => set("sampler", e.target.value)}>
            <option value="grid">Grid (every combination)</option><option value="random">Random</option>
            <option value="tpe">Adaptive (TPE)</option></select></label>
          {values.sampler !== "grid" && field("trials_per_model", "Trials per model")}
          {field("seed", "Search seed")}
        </div>
        <div className="row">
          <label>Objective (validation)<select value={values.objective} onChange={(e) => set("objective", e.target.value)}>
            {(options?.objectives ?? []).map((o) => <option key={o}>{o}</option>)}</select></label>
          {field("max_epochs_per_trial", "Max epochs per trial")}
          {field("time_limit_minutes", "Time limit (minutes, optional)")}
        </div>
        <div className="row">
          <label>Start from<select value={values.initialization} onChange={(e) => set("initialization", e.target.value)}>
            <option value="pretrained">Pretrained</option><option value="scratch">Scratch</option></select></label>
          <label>If a trial fails<select value={values.failure_policy} onChange={(e) => set("failure_policy", e.target.value)}>
            <option value="continue">Continue</option><option value="stop">Stop the study</option></select></label>
          {field("tracking_destination_id", "Tracking destination", "text")}
        </div>
        <label className="check"><input type="checkbox" checked={values.pruning} onChange={(e) => set("pruning", e.target.checked)} />
          Prune: stop trials early whose scores fall below the median (random/TPE, within one model)</label>
        <label>Fixed settings for every trial (JSON)<textarea className="code" rows={3} value={values.base_training}
          onChange={(e) => set("base_training", e.target.value)} /></label>
        <h3>Search space</h3>
        {params.map((p, index) => (
          <div key={index} className="param">
            <div className="row">
              <label>Parameter<select value={p.name} onChange={(e) => setParam(index, { name: e.target.value })}>
                {searchable.map((n) => <option key={n}>{n}</option>)}</select></label>
              <label>Type<select value={p.type} onChange={(e) => setParam(index, { type: e.target.value as Param["type"] })}>
                <option value="float">Decimal range</option><option value="int">Whole-number range</option>
                <option value="categorical">Choices</option></select></label>
              {p.type === "categorical" ? (
                <label>Choices (comma-separated)<input value={p.choices} onChange={(e) => setParam(index, { choices: e.target.value })} /></label>
              ) : (<>
                <label>Low<input value={p.low} onChange={(e) => setParam(index, { low: e.target.value })} /></label>
                <label>High<input value={p.high} onChange={(e) => setParam(index, { high: e.target.value })} /></label>
                <label>Step<input value={p.step} placeholder="none" onChange={(e) => setParam(index, { step: e.target.value })} /></label>
                <label className="check"><input type="checkbox" checked={p.log} onChange={(e) => setParam(index, { log: e.target.checked })} />log</label>
              </>)}
            </div>
            <div className="row">
              <label>Only when (optional)<input value={p.conditionParam} placeholder="e.g. optimizer"
                onChange={(e) => setParam(index, { conditionParam: e.target.value })} /></label>
              <label>is one of<input value={p.conditionValues} placeholder="e.g. SGD, AdamW"
                onChange={(e) => setParam(index, { conditionValues: e.target.value })} /></label>
              <button type="button" onClick={() => setParams((all) => all.filter((_, i) => i !== index))}>Remove</button>
            </div>
          </div>
        ))}
        <p><button type="button" onClick={() => setParams((all) => [...all, { ...EMPTY, name: "", log: false }])}>Add parameter</button></p>
        <div className="actions">
          <button type="button" onClick={() => check(false)}>Check plan</button>
          <button type="button" className="primary" onClick={() => check(true)}>Start study</button>
        </div>
        {plan && <p className="muted">{plan}</p>}
        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}

// One study: progress, best results, every trial, and stop / resume / repeat.
function StudyView({ studyId, onOpenJob, onCreated }: {
  studyId: string; onOpenJob: (id: string) => void; onCreated: (id: string) => void;
}) {
  const [study, setStudy] = useState<StudyDetail | null>(null);
  const [message, setMessage] = useState("");
  const [seeds, setSeeds] = useState("1, 2, 3");
  const load = useCallback(() => { api.study(studyId).then(setStudy, (e) => setMessage(describeError(e))); }, [studyId]);
  useEffect(() => { load(); const t = window.setInterval(load, 2000); return () => window.clearInterval(t); }, [load]);
  if (!study) return <div className="card"><p className="muted">{message || "Loading…"}</p></div>;

  // Run a study action and show its error, if any.
  const act = (action: Promise<unknown>) => action.then(load, (e) => setMessage(describeError(e)));
  const { summary, state, spec } = study;
  const running = state.state === "running" || state.state === "stopping";
  const parent = Object.values(study.mlflow)[0];
  return (
    <div className="card">
      <h2>{spec.name} <span className={`state ${state.state}`}>{state.state}</span></h2>
      <p className="muted">{summary.trials_started} / {summary.planned_trials} trials · {spec.sampler} search ·
        {" "}{summary.comparison} · objective {spec.objective} (validation){state.reason ? ` · ${state.reason}` : ""}</p>
      <p className="muted small">complete {summary.counts.complete} · failed {summary.counts.failed} · pruned {summary.counts.pruned}
        {" "}· cancelled {summary.counts.cancelled} · trial time {Math.round(summary.trial_seconds_total)} s</p>
      <div className="actions">
        {running && <button onClick={() => act(api.stopStudy(studyId, "finish"))}>Stop (let the active trial finish)</button>}
        {running && <button onClick={() => act(api.stopStudy(studyId, "cancel"))}>Stop now (cancel the active trial)</button>}
        {!running && summary.trials_started < summary.planned_trials &&
          <button onClick={() => act(api.resumeStudy(studyId))}>Resume</button>}
        {parent?.url && <a className="button" href={parent.url} target="_blank" rel="noreferrer">Study in MLflow</a>}
      </div>
      {summary.best && (
        <p>Best: trial {summary.best.number} · {summary.best.model_id} · <strong>{summary.best.value.toFixed(4)}</strong>{" "}
          <button onClick={() => onOpenJob(summary.best!.job_id)}>Open job</button>
          <span className="muted small"> checkpoint {summary.best.checkpoint_sha256?.slice(0, 12)}…</span></p>
      )}
      {Object.keys(summary.best_per_model).length > 1 && (
        <table className="compact"><tbody>
          {Object.entries(summary.best_per_model).map(([model, best]) => (
            <tr key={model}><td>{model}</td><td>{best ? `trial ${best.number}: ${best.value.toFixed(4)}` : "no complete trial"}</td></tr>
          ))}
        </tbody></table>
      )}
      {summary.repeat_groups && summary.repeat_groups.map((group) => (
        <p key={group.configuration} className="muted small">{group.configuration}: {group.values.map((v) =>
          `seed ${v.seed} → ${v.value?.toFixed(4) ?? "—"}`).join(", ")} · mean {group.mean?.toFixed(4) ?? "—"}
          {group.stdev !== null ? ` ± ${group.stdev.toFixed(4)}` : ""}</p>
      ))}
      <h3>Trials</h3>
      <table className="compact">
        <thead><tr><th>#</th><th>Model</th><th>Values</th><th>State</th><th>Score</th></tr></thead>
        <tbody>
          {study.trials.map((t) => (
            <tr key={t.number} className={summary.best?.number === t.number ? "selected" : ""}>
              <td>{t.number}</td><td>{t.model_id}</td>
              <td className="muted">{Object.entries(t.params).map(([k, v]) => `${k}=${typeof v === "number" ? +v.toPrecision(4) : v}`).join(", ")}</td>
              <td><span className={`state ${t.state === "complete" ? "finished" : t.state === "pruned" ? "cancelled" : t.state}`}>{t.state}</span>
                {t.error && <div className="error small">{t.error}</div>}</td>
              <td>{t.value !== null ? t.value.toFixed(4) : "—"}{" "}
                {t.job_id && <button onClick={() => onOpenJob(t.job_id!)}>job</button>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {spec.sampler !== "repeat" && !spec.recipe && summary.counts.complete > 0 && (
        <div className="actions">
          <label>Repeat the best trial with seeds<input value={seeds} onChange={(e) => setSeeds(e.target.value)} /></label>
          <button onClick={() => api.repeatStudy(studyId, { top: 1, seeds: parseList(seeds) as number[] })
            .then((r) => onCreated(r.study_id), (e) => setMessage(describeError(e)))}>Repeat</button>
        </div>
      )}
      {message && <p className="error">{message}</p>}
    </div>
  );
}
