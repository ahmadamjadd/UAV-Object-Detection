import { useEffect, useMemo, useRef, useState } from "react";
import { useShared } from "../App";
import { api, CheckpointOption, CoverageReport, Dataset, describeError, Destination, ModelWeights } from "../api";
import { Banner, DeviceField, Empty, ErrorText, go, optionLabel, score, Switch, Tabs, Tip, useGpus } from "../ui";
import { StudiesView } from "./Studies";

// One setting's schema entry (type, limits, group, help text).
type Property = {
  type?: string; anyOf?: { type: string; minimum?: number; maximum?: number; exclusiveMinimum?: number }[]; enum?: string[];
  default?: unknown; description?: string; group?: string; minimum?: number; maximum?: number; exclusiveMinimum?: number;
  multipleOf?: number;
};
// Where training starts from.
type Start = "pretrained" | "scratch" | "checkpoint";

// Train: one run, or a tuning study that compares many settings fairly.
export function TrainPage({ params }: { params: URLSearchParams }) {
  const mode = params.get("mode") === "study" ? "study" : "single";
  return (
    <>
      <h2>Train</h2>
      <p className="sub">Start one training run, or compare many settings fairly with a tuning study.</p>
      <Tabs value={mode} onChange={(m) => go("train", m === "study" ? { mode: "study" } : {})}
        options={[["single", "Single run"], ["study", "Tuning study"]]} />
      {mode === "single" ? <SingleRun initialDataset={params.get("dataset")} /> : <StudiesView studyId={params.get("study")} />}
    </>
  );
}

// The single-run form, built from the backend's settings schema.
function SingleRun({ initialDataset }: { initialDataset: string | null }) {
  const { capabilities, draft, setDraft, member } = useShared();
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [properties, setProperties] = useState<Record<string, Property>>({});
  const [groups, setGroups] = useState<Record<string, string>>({});
  const [weights, setWeights] = useState<ModelWeights[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointOption[]>([]);
  const [datasetId, setDatasetId] = useState(initialDataset ?? "");
  const [modelId, setModelId] = useState("yolov8n");
  const [start, setStart] = useState<Start>("pretrained");
  const [checkpointJob, setCheckpointJob] = useState("");
  const [destination, setDestination] = useState("");
  const [changed, setChanged] = useState<Record<string, unknown>>({});
  // "New experiment…" from Runs arrives as a draft request (opened in the JSON editor) and a suggested name.
  const { __name: draftName, ...draftRequest } = (draft ?? {}) as Record<string, unknown>;
  const [json, setJson] = useState<string | null>(draft ? JSON.stringify(draftRequest, null, 2) : null);
  const [runName, setRunName] = useState(typeof draftName === "string" ? draftName : "");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submissionKey = useRef<string | null>(null);
  const [gpus] = useGpus();
  const [destinations, setDestinations] = useState<Destination[]>([]);
  useEffect(() => { api.destinations().then(setDestinations, () => undefined); }, []);
  const experiment = member || destinations.find((d) => d.destination_id === destination)?.experiment_name;
  // Use the first GPU by default when one is available (the member can switch back to CPU).
  useEffect(() => { if (gpus?.available && !("device" in changed) && !draft) setChanged((c) => ({ ...c, device: gpus.devices[0].id })); }, [gpus]);

  useEffect(() => {
    if (draft) setDraft(null);
    api.datasets().then((all) => {
      const training = all.filter((d) => d.role === "training");  // real validation/test images are never trained on
      setDatasets(training);
      setDatasetId((old) => old || training[0]?.dataset_id || "");
    }, (e) => setError(describeError(e)));
    api.schemas().then((s) => { setProperties(s.job?.$defs?.TrainingSettings?.properties ?? {}); setGroups(s.training_groups ?? {}); });
    api.models().then(setWeights, () => undefined);
    api.checkpoints().then(setCheckpoints, () => undefined);
  }, []);
  useEffect(() => { setDestination((old) => old || capabilities?.tracking_destinations[0] || "local"); }, [capabilities]);

  const usable = checkpoints.filter((c) => c.model_id === modelId);
  const installed = weights.find((w) => w.model_id === modelId)?.status === "installed";
  const request = useMemo((): Record<string, unknown> => ({
    dataset_id: datasetId, model_id: modelId, initialization: start,
    ...(start === "checkpoint" ? { checkpoint_job_id: checkpointJob || usable[0]?.job_id || "" } : {}),
    tracking_destination_id: destination, training: changed,
  }), [datasetId, modelId, start, checkpointJob, usable, destination, changed]);

  // Change one setting; going back to its default removes it from the request.
  function setValue(name: string, value: unknown) {
    setChanged((old) => {
      const next = { ...old };
      if (value === properties[name]?.default || (value === null && properties[name]?.default == null)) delete next[name];
      else next[name] = value;
      return next;
    });
  }

  // Switch between the form and the JSON editor, carrying values across.
  function toggleJson(on: boolean) {
    if (on) { setJson(JSON.stringify(request, null, 2)); return; }
    try {
      const value = JSON.parse(json ?? "{}");
      setDatasetId(value.dataset_id ?? ""); setModelId(value.model_id ?? "yolov8n");
      setStart(["pretrained", "scratch", "checkpoint"].includes(value.initialization) ? value.initialization : "pretrained");
      setCheckpointJob(value.checkpoint_job_id ?? ""); setDestination(value.tracking_destination_id ?? "");
      setChanged(value.training ?? {}); setJson(null); setError("");
    } catch (e) { setError(`JSON: ${describeError(e)}`); }
  }

  // Submit; one submission key is kept until success, so a retry never creates a second job.
  async function submit() {
    let body: Record<string, unknown>;
    try { body = json === null ? request : JSON.parse(json); } catch (e) { setError(`JSON: ${describeError(e)}`); return; }
    submissionKey.current ??= crypto.randomUUID();
    setBusy(true); setError("");
    try {
      const job = await api.submit(submissionKey.current, body, runName.trim() || null);
      submissionKey.current = null;
      go("runs", { job: job.job_id });
    } catch (e) { setError(describeError(e)); } finally { setBusy(false); }
  }

  const families = new Map<string, NonNullable<typeof capabilities>["model_details"]>();
  for (const m of capabilities?.model_details ?? []) families.set(m.family, [...(families.get(m.family) ?? []), m]);
  // "device" is chosen with "Run on" above (it needs the detected GPUs), not in the groups.
  const grouped = Object.keys(groups).map((g) => [g, Object.entries(properties).filter(([name, p]) => p.group === g && name !== "device")] as const);
  const earlyStopping = ("early_stopping" in changed ? changed.early_stopping : properties.early_stopping?.default) !== false;
  const count = Object.keys(changed).length;
  if (datasets && datasets.length === 0) return (
    <div className="card"><Empty title="No training dataset yet">Generate or import one first.
      <div className="row" style={{ justifyContent: "center", marginTop: 12 }}><button className="btn acc" onClick={() => go("datasets", { add: "generate" })}>Add dataset</button></div></Empty></div>
  );
  return (
    <div className="grid g2">
      <div>
        <div className="card" style={{ marginBottom: 14 }}>
          <label className="f">Run name (optional)<input type="text" value={runName} maxLength={80} placeholder="e.g. yolo11 small · noisy airfield"
            onChange={(e) => setRunName(e.target.value)} /><span className="h">Helps you find this model later; also used as the run name in MLflow. You can rename it any time.</span></label>
          {json !== null ? <p className="mute small" style={{ margin: 0 }}>Editing the request as JSON below — the same checks as the form.</p> : (<>
            <div className="fg">
              <label className="f">Dataset<select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                {(datasets ?? []).map((d) => <option key={d.dataset_id} value={d.dataset_id}>{optionLabel(d.name, d.dataset_id)} ({d.class_names.join(", ")})</option>)}
              </select><span className="h">Only training datasets are listed.</span></label>
              <label className="f">Model<select value={modelId} onChange={(e) => { setModelId(e.target.value); setCheckpointJob(""); }}>
                {[...families].map(([family, models]) => <optgroup key={family} label={family.toUpperCase().replace("YOLOV", "YOLOv")}>
                  {models!.map((m) => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}</optgroup>)}
              </select></label>
              <label className="f">Start from<select value={start} onChange={(e) => setStart(e.target.value as Start)}>
                <option value="pretrained">Pretrained weights</option><option value="scratch">Random weights (scratch)</option>
                <option value="checkpoint">An earlier job's best checkpoint</option></select></label>
              <DeviceField value={String(changed.device ?? "cpu")} onChange={(device) => setValue("device", device)} />
              <label className="f">Tracking destination<select value={destination} onChange={(e) => setDestination(e.target.value)}>
                {[...new Set([destination, ...(capabilities?.tracking_destinations ?? [])])].filter(Boolean).map((d) => <option key={d}>{d}</option>)}
              </select><span className="h">{experiment ? <>Uploaded into the MLflow experiment “{experiment}” as run “{runName.trim() || "foxtrot-<id>"}”. {member ? "The experiment is your member name." : "Set your member name in Settings to use it as the experiment."}</> : "Where the finished run is uploaded."}</span></label>
            </div>
            {start === "checkpoint" && (usable.length === 0 ? <Banner kind="warn">No finished {modelId} training job yet.</Banner> : (
              <label className="f">Earlier job<select value={checkpointJob || usable[0].job_id} onChange={(e) => setCheckpointJob(e.target.value)}>
                {usable.map((c) => <option key={c.job_id} value={c.job_id}>{c.name ?? c.job_id.slice(0, 8)} · {c.dataset_id}{c.map50_95 !== null ? ` · mAP50-95 ${score(c.map50_95)}` : ""}</option>)}
              </select></label>))}
            {start === "pretrained" && (installed
              ? <div className="banner acc" style={{ margin: 0 }}><b>✓</b>Pretrained weights for {modelId} are installed.</div>
              : <Banner kind="warn">{modelId}.pt is not installed yet: install it in Settings › Pretrained weights.
                  <button className="btn sm" style={{ marginLeft: 8 }} onClick={() => go("settings")}>Open Settings</button></Banner>)}
          </>)}
        </div>
        <div className="row between" style={{ marginBottom: 10 }}>
          <b>Settings</b>
          <div className="row">
            <label className="row mute small" style={{ gap: 8 }}><Switch checked={json !== null} onChange={toggleJson} label="Edit as JSON" />Edit as JSON</label>
            {json === null && <button className="btn ghost" onClick={() => setChanged({})} disabled={count === 0}>Reset settings</button>}
          </div>
        </div>
        {json !== null ? (
          <textarea className="code" rows={20} value={json} onChange={(e) => setJson(e.target.value)} aria-label="Request JSON" />
        ) : grouped.map(([group, fields], index) => (
          <details key={group} className="grp" open={index === 0}>
            <summary>{groups[group]}{fields.some(([name]) => name in changed) && <span className="pill p-acc r">changed</span>}</summary>
            <div className="in fg">
              {fields.map(([name, property]) => (
                <Setting key={name} name={name} property={property} value={name in changed ? changed[name] : property.default}
                  onChange={(v) => setValue(name, v)} disabled={name === "patience" && !earlyStopping} />
              ))}
              {group === "early_stopping" && <p className="mute tiny" style={{ gridColumn: "1 / -1", margin: "0 0 10px" }}>
                After every epoch the model is scored on the validation images (mAP50-95). {earlyStopping
                  ? `If that score has not improved for ${changed.patience ?? properties.patience?.default} epochs, training stops and the best checkpoint is kept.`
                  : "Early stopping is off: training always runs all epochs (the best checkpoint is still the one kept)."}</p>}
            </div>
          </details>
        ))}
        <div className="row" style={{ marginTop: 14 }}>
          <button className="btn acc" onClick={submit} disabled={busy || !datasets?.length}>{busy ? "Starting…" : "Start training"}</button>
          <span className="mute small">{json !== null ? "Request from the JSON editor" : count ? `${count} changed setting${count > 1 ? "s" : ""} will be sent` : "Using all defaults"}</span>
        </div>
        <ErrorText message={error} />
      </div>
      <Coverage />
    </div>
  );
}

const LABELS: Record<string, string> = {
  epochs: "Epochs", batch_size: "Batch size", image_size: "Training image size", seed: "Seed", patience: "Patience (epochs)", early_stopping: "Early stopping", amp: "Mixed precision (GPU)",
  fraction: "Share of training images", freeze: "Frozen layers", optimizer: "Optimizer", lr0: "Starting learning rate",
  lrf: "Final learning-rate fraction", momentum: "Momentum", weight_decay: "Weight decay", warmup_epochs: "Warm-up epochs",
  warmup_momentum: "Warm-up momentum", warmup_bias_lr: "Warm-up bias learning rate", cos_lr: "Cosine learning-rate schedule",
  box: "Box loss weight", cls: "Class loss weight", dfl: "Distribution focal loss weight", hsv_h: "Hue shift", hsv_s: "Saturation shift",
  hsv_v: "Brightness shift", degrees: "Rotation (degrees)", translate: "Shift", scale: "Zoom", shear: "Shear (degrees)",
  perspective: "Perspective", flipud: "Upside-down flip chance", fliplr: "Horizontal flip chance", bgr: "Colour-channel swap chance",
  mosaic: "Mosaic", close_mosaic: "Mosaic off for last N epochs", mixup: "Mixup", cutmix: "CutMix", device: "Device",
  workers: "Data loader workers", cache: "Cache images", deterministic: "Repeatable operations",
};

// One setting's input, chosen from its schema type.
function Setting({ name, property, value, onChange, disabled = false }: {
  name: string; property: Property; value: unknown; onChange: (v: unknown) => void; disabled?: boolean;
}) {
  const label = LABELS[name] ?? name.replace(/_/g, " ");
  if (property.enum) return (
    <label className="f">{label}<select value={String(value)} onChange={(e) => onChange(e.target.value)}>
      {property.enum.map((o) => <option key={o}>{o}</option>)}</select><span className="h">{property.description}</span></label>
  );
  if (property.type === "boolean") return (
    <label className="swl"><div>{label}<span>{property.description}</span></div>
      <Switch checked={Boolean(value)} onChange={onChange} label={label} /></label>
  );
  const numeric = (property.anyOf?.find((o) => o.type !== "null") ?? property) as Property;
  const nullable = Boolean(property.anyOf);
  const integer = numeric.type === "integer";
  return (
    <label className="f">{label}
      <input type="number" disabled={disabled} step={property.multipleOf ?? (integer ? 1 : "any")} min={numeric.minimum ?? numeric.exclusiveMinimum}
        max={numeric.maximum} value={value === null || value === undefined ? "" : String(value)} placeholder={nullable ? "default" : undefined}
        onChange={(e) => onChange(e.target.value === "" ? (nullable ? null : property.default) : integer ? Math.trunc(e.target.valueAsNumber) : e.target.valueAsNumber)} />
      <span className="h">{property.description}</span>
    </label>
  );
}

// How every YOLO training setting is handled by Foxtrot.
function Coverage() {
  const [report, setReport] = useState<CoverageReport | null>(null);
  useEffect(() => { api.coverage().then(setReport, () => undefined); }, []);
  const rows: [string, string, string][] = [["exposed", "p-ok", "Exposed in this form"], ["managed", "p-blue", "Managed for you (e.g. data paths)"],
    ["unsupported", "p-warn", "Not supported, with reason"], ["inapplicable", "p-mute", "Not applicable to detection"]];
  return (
    <div className="card" style={{ alignSelf: "start" }}>
      <h3>Settings coverage</h3>
      <p className="mute small" style={{ marginTop: 0 }}>How every YOLO training setting is handled by Foxtrot.</p>
      {report && <>
        <div className="list">{rows.map(([status, pill, label]) => (
          <div key={status}><span className={`pill ${pill}`}>{report.runtime.filter((r) => r.status === status).length}</span>{label}</div>))}</div>
        {(["runtime", "generator"] as const).map((part) => (
          <details key={part} className="plain" style={{ marginTop: 10 }}>
            <summary>{part === "runtime" ? "Show full YOLO table" : "Show generator settings"}</summary>
            <div className="scrolly"><table style={{ marginTop: 8, fontSize: 13 }}><tbody>
              {[...report[part]].sort((a, b) => rows.findIndex((r) => r[0] === a.status) - rows.findIndex((r) => r[0] === b.status)).map((row) => (
                <tr key={row.key}><td className="mono">{row.key}</td><td><span className={`pill ${rows.find((r) => r[0] === row.status)?.[1]}`}>{row.status === "inapplicable" ? "n/a" : row.status}</span></td>
                  <td className="mute wrapc">{row.detail}</td></tr>))}
            </tbody></table></div>
          </details>
        ))}
      </>}
      <p className="mute tiny" style={{ marginBottom: 0 }}><Tip text="Settings are checked together before a job is created, e.g. optimizer 'auto' chooses its own learning rate.">Combinations are checked before training.</Tip></p>
    </div>
  );
}
