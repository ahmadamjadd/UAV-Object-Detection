import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api, Capabilities, CheckpointOption, CoverageReport, Dataset, describeError, ModelWeights } from "./api";

type Property = {
  type?: string; anyOf?: { type: string; minimum?: number; maximum?: number }[]; enum?: string[];
  default?: unknown; description?: string; group?: string; minimum?: number; maximum?: number;
  exclusiveMinimum?: number; exclusiveMaximum?: number; multipleOf?: number; title?: string;
};
type Values = Record<string, unknown>;
type Start = "pretrained" | "scratch" | "checkpoint";

// New training: dataset, model, how it starts, and every training setting, grouped as the backend says.
// The form is built from the backend's schema; "Edit as JSON" sends the same request through the same checks.
export function TrainingForm({ capabilities, initialDataset, initialRequest, onSubmitted }: {
  capabilities: Capabilities | null; initialDataset: string | null; initialRequest?: Record<string, unknown> | null;
  onSubmitted: (id: string) => void;
}) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [properties, setProperties] = useState<Record<string, Property>>({});
  const [groups, setGroups] = useState<Record<string, string>>({});
  const [weights, setWeights] = useState<ModelWeights[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointOption[]>([]);
  const [datasetId, setDatasetId] = useState(initialDataset ?? "");
  const [modelId, setModelId] = useState("yolov8n");
  const [start, setStart] = useState<Start>("pretrained");
  const [checkpointJob, setCheckpointJob] = useState("");
  const [destination, setDestination] = useState("");
  const [changed, setChanged] = useState<Values>({});
  // V7 "New experiment": start in the JSON editor with an earlier job's request, ready to change.
  const [json, setJson] = useState<string | null>(initialRequest ? JSON.stringify(initialRequest, null, 2) : null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submissionKey = useRef<string | null>(null);

  // Load datasets, the settings schema, installed weights and usable checkpoints.
  useEffect(() => {
    api.datasets().then((all) => {
      const list = all.filter((d) => d.role === "training");  // real validation/test images are never trained on
      setDatasets(list);
      setDatasetId((old) => old || list[0]?.dataset_id || "");
    }, (e) => setError(describeError(e)));
    api.schemas().then((s) => {
      setProperties(s.job?.$defs?.TrainingSettings?.properties ?? {});
      setGroups(s.training_groups ?? {});
    });
    api.models().then(setWeights, () => undefined);
    api.checkpoints().then(setCheckpoints, () => undefined);
  }, []);
  useEffect(() => { setDestination((old) => old || capabilities?.tracking_destinations[0] || "local"); }, [capabilities]);

  const usable = checkpoints.filter((c) => c.model_id === modelId);
  const installed = weights.find((w) => w.model_id === modelId)?.status === "installed";

  // The request exactly as the backend receives it; training holds only settings changed from default.
  const request = useMemo((): Values => ({
    dataset_id: datasetId, model_id: modelId, initialization: start,
    ...(start === "checkpoint" ? { checkpoint_job_id: checkpointJob || usable[0]?.job_id || "" } : {}),
    tracking_destination_id: destination, training: changed,
  }), [datasetId, modelId, start, checkpointJob, usable, destination, changed]);

  // Set one training setting; going back to the default removes it from the request.
  function setValue(name: string, value: unknown) {
    setChanged((old) => {
      const next = { ...old };
      if (value === properties[name]?.default || (value === null && properties[name]?.default === null)) delete next[name];
      else next[name] = value;
      return next;
    });
  }

  // Switch between the form and the JSON editor, carrying the values across.
  function toggleJson() {
    if (json === null) { setJson(JSON.stringify(request, null, 2)); return; }
    try {
      const value = JSON.parse(json);
      setDatasetId(value.dataset_id ?? ""); setModelId(value.model_id ?? "yolov8n");
      setStart(value.initialization ?? "pretrained"); setCheckpointJob(value.checkpoint_job_id ?? "");
      setDestination(value.tracking_destination_id ?? ""); setChanged(value.training ?? {});
      setJson(null); setError("");
    } catch (e) { setError(`JSON: ${describeError(e)}`); }
  }

  // Send the request; the backend validates everything, including combinations of settings.
  async function submit(event: FormEvent) {
    event.preventDefault();
    let body: Values;
    try { body = json === null ? request : JSON.parse(json); } catch (e) { setError(`JSON: ${describeError(e)}`); return; }
    submissionKey.current ??= crypto.randomUUID();  // one key until success: a retry never duplicates a job
    setBusy(true);
    setError("");
    try {
      const job = await api.submit(submissionKey.current, body);
      submissionKey.current = null;
      onSubmitted(job.job_id);
    } catch (e) { setError(describeError(e)); } finally { setBusy(false); }
  }

  const families = new Map<string, NonNullable<Capabilities["model_details"]>>();
  for (const m of capabilities?.model_details ?? []) families.set(m.family, [...(families.get(m.family) ?? []), m]);
  const grouped = Object.keys(groups).map((group) =>
    [group, Object.entries(properties).filter(([, p]) => p.group === group)] as const);

  return (
    <section className="grid wide">
      <div className="card">
        <h2>New training</h2>
        {datasets.length === 0 && <p className="muted">No datasets yet: import or generate one first.</p>}
        <form onSubmit={submit} className="form">
          {json !== null ? (
            <label>Request (JSON): the same checks as the form
              <textarea className="code" rows={18} value={json} onChange={(e) => setJson(e.target.value)} /></label>
          ) : (
            <>
              <div className="row">
                <label>Dataset<select value={datasetId} onChange={(e) => setDatasetId(e.target.value)} required>
                  {datasets.map((d) => <option key={d.dataset_id} value={d.dataset_id}>
                    {d.dataset_id} ({d.class_names.join(", ")})</option>)}
                </select></label>
                <label>Model<select value={modelId} onChange={(e) => { setModelId(e.target.value); setCheckpointJob(""); }}>
                  {[...families].map(([family, models]) => (
                    <optgroup key={family} label={family}>
                      {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}
                    </optgroup>
                  ))}
                </select></label>
              </div>
              <fieldset className="choices">
                <legend>Start from</legend>
                {([["pretrained", "Pretrained weights"], ["scratch", "Random weights (scratch)"],
                  ["checkpoint", "An earlier job's best checkpoint"]] as [Start, string][]).map(([value, label]) => (
                  <label key={value} className="check"><input type="radio" name="start" checked={start === value}
                    onChange={() => setStart(value)} />{label}</label>
                ))}
              </fieldset>
              {start === "pretrained" && !installed &&
                <p className="error small">{modelId}.pt is not installed yet: install it on the Setup page.</p>}
              {start === "checkpoint" && (usable.length === 0
                ? <p className="error small">No finished {modelId} job yet.</p>
                : <label>Earlier job<select value={checkpointJob || usable[0].job_id}
                    onChange={(e) => setCheckpointJob(e.target.value)}>
                    {usable.map((c) => <option key={c.job_id} value={c.job_id}>
                      {c.job_id.slice(0, 8)} · {c.dataset_id}{c.map50_95 !== null ? ` · mAP50-95 ${c.map50_95.toFixed(3)}` : ""}
                    </option>)}
                  </select></label>)}
              <label>Tracking destination<input value={destination} required list="destinations"
                onChange={(e) => setDestination(e.target.value)} />
                <datalist id="destinations">
                  {(capabilities?.tracking_destinations ?? []).map((d) => <option key={d} value={d} />)}
                </datalist></label>
              {grouped.map(([group, fields], index) => (
                <details key={group} open={index === 0}>
                  <summary>{groups[group]}
                    {fields.some(([name]) => name in changed) && <span className="badge small">changed</span>}</summary>
                  <div className="row">
                    {fields.map(([name, property]) => (
                      <Setting key={name} name={name} property={property}
                        value={name in changed ? changed[name] : property.default} onChange={(v) => setValue(name, v)} />
                    ))}
                  </div>
                </details>
              ))}
            </>
          )}
          <div className="actions">
            <button type="submit" disabled={busy || datasets.length === 0}>{busy ? "Submitting…" : "Start training"}</button>
            <button type="button" onClick={toggleJson}>{json === null ? "Edit as JSON" : "Back to form"}</button>
            {json === null && Object.keys(changed).length > 0 &&
              <button type="button" onClick={() => setChanged({})}>Reset settings</button>}
          </div>
        </form>
        {error && <p className="error">{error}</p>}
      </div>
      <Coverage />
    </section>
  );
}

// One setting's input, chosen from its schema type: list, checkbox, or number with its limits.
function Setting({ name, property, value, onChange }: {
  name: string; property: Property; value: unknown; onChange: (value: unknown) => void;
}) {
  const label = name.replace(/_/g, " ");
  if (property.enum) {
    return <label title={property.description}>{label}
      <select value={String(value)} onChange={(e) => onChange(e.target.value)}>
        {property.enum.map((option) => <option key={option}>{option}</option>)}</select></label>;
  }
  if (property.type === "boolean") {
    return <label className="check" title={property.description}>
      <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />{label}</label>;
  }
  const numeric = property.anyOf?.find((option) => option.type !== "null") ?? property;
  const nullable = Boolean(property.anyOf);
  const integer = numeric.type === "integer";
  const step = property.multipleOf ?? (integer ? 1 : "any");
  return <label title={property.description}>{label}
    <input type="number" step={step} min={(numeric as Property).minimum ?? (numeric as Property).exclusiveMinimum}
      max={(numeric as Property).maximum} value={value === null || value === undefined ? "" : String(value)}
      placeholder={nullable ? "default" : undefined}
      onChange={(e) => onChange(e.target.value === "" ? (nullable ? null : property.default)
        : integer ? Math.trunc(e.target.valueAsNumber) : e.target.valueAsNumber)} />
    <span className="hint">{property.description}</span></label>;
}

// What happens to every runtime and generator setting: exposed, managed, other task, or not yet supported.
function Coverage() {
  const [report, setReport] = useState<CoverageReport | null>(null);
  useEffect(() => { api.coverage().then(setReport, () => undefined); }, []);
  if (!report) return <div className="card"><p className="muted">Loading coverage…</p></div>;
  const order = ["exposed", "managed", "unsupported", "inapplicable"];
  const counts = order.map((status) => `${report.runtime.filter((r) => r.status === status).length} ${status}`);
  return (
    <div className="card">
      <h2>Settings coverage</h2>
      <p className="muted">Every YOLO runtime setting: {counts.join(" · ")}.</p>
      {(["runtime", "generator"] as const).map((part) => (
        <details key={part}>
          <summary>{part === "runtime" ? "YOLO training settings" : "Generator settings"}</summary>
          <table className="compact">
            <tbody>
              {[...report[part]].sort((a, b) => order.indexOf(a.status) - order.indexOf(b.status)).map((row) => (
                <tr key={row.key}><td><code>{row.key}</code></td>
                  <td><span className={`state ${row.status}`}>{row.status}</span></td>
                  <td className="muted">{row.detail}</td></tr>
              ))}
            </tbody>
          </table>
        </details>
      ))}
    </div>
  );
}
