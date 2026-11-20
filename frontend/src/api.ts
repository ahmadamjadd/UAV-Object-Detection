// Small typed client for the local API. State-changing calls carry the
// per-launch session token; the browser adds the same-origin Origin header.

export type JobView = {
  job_id: string;
  execution_state: string;
  tracking_state: string;
  created_at: string;
  finished_at: string | null;
  exit_code: number | null;
  cancel_requested: boolean;
  recovery_required: boolean;
  last_event_sequence: number;
  error: { code: string; message: string } | null;
  tracking_error: { code: string; message: string } | null;
};

export type JobDetail = JobView & {
  resolved: Record<string, unknown> | null;
  tracking_run: { run_id: string; run_url: string | null } | null;
  result: {
    outcome: string;
    evaluation: { precision: number; recall: number; map50: number; map50_95: number } | null;
  } | null;
};

export type JobEvent = {
  sequence: number;
  timestamp: string;
  event_type: "state_changed" | "progress" | "log" | "error";
  payload: Record<string, unknown>;
};

export type Capabilities = {
  version: string;
  models: string[];
  devices: string[];
  pipeline_adapters: string[];
  active_pipeline_adapter: string | null;
  real_training: boolean;
  tracking_destinations: string[];
  model_details?: { model_id: string; family: string; variant: string; display_name: string; task: string }[];
  dvc_default_repository?: string | null;
};

// A finished job whose best checkpoint a new job can start from (V3).
export type CheckpointOption = {
  job_id: string; model_id: string; dataset_id: string; finished_at: string | null; map50_95: number | null;
};

export type CoverageRow = { key: string; status: "exposed" | "managed" | "inapplicable" | "unsupported"; detail: string };
export type CoverageReport = { runtime: CoverageRow[]; generator: CoverageRow[] };

export type Dataset = {
  dataset_id: string; class_names: string[]; validation_status: string; role: "training" | "validation" | "test";
};

export type DatasetIssue = { split: string; file: string | null; line: number | null; message: string };

export type DatasetReport = {
  valid: boolean;
  errors: DatasetIssue[];
  warnings: DatasetIssue[];
  error_count: number;
  warning_count: number;
  splits: Record<string, { images: number; labelled_images: number; missing_labels: number;
    objects_per_class: Record<string, number> }>;
};

export type DatasetImportResult = { imported: boolean; dataset: Dataset | null; datasets?: Dataset[]; report: DatasetReport };

export type ModelWeights = {
  model_id: string; status: "installed" | "missing" | "changed"; sha256: string | null; size_bytes: number | null;
};

export type Destination = {
  destination_id: string;
  kind: string;
  tracking_uri: string;
  browser_url: string | null;
  experiment_name: string;
  has_credentials: boolean;
};

// V2 dataset generation: settings sent to the backend (see config/generation.py).
export type Range = { low: number; high: number };
export type GenerationRecipe = {
  images_per_class: number; canvas_size: number; seed: number; shape_scale: Range; noise: number; blur: Range;
  brightness_difference: number; contrast: Range; brightness: Range; saturation: Range; desaturate: boolean;
  add_letter: boolean; letter_probability: number; add_light_gradient: boolean; light_probability: number;
  light_opacity: number;
};
export type SplitSettings = { train_ratio: number; seed: number; group_by_background: boolean };
export type GenerationRequest = { recipe: GenerationRecipe; split: SplitSettings; backgrounds: string };

export type GenerationOptions = {
  available: boolean; message: string | null; default_backgrounds: string | null;
  recipe: GenerationRecipe; split: SplitSettings;
};

export type PreviewBox = { class_name: string; x: number; y: number; width: number; height: number };
export type PreviewResult = {
  seed: number; canvas_size: number; class_names: string[]; background_count: number;
  images: { image: string; background: string; boxes: PreviewBox[] }[];
};
export type DatasetBuildResult = {
  dataset_id: string; cached: boolean; class_names: string[];
  counts: { images: number; train: number; val: number; objects_per_class: Record<string, number> } | null;
  groups: Record<string, number>; report: DatasetReport | null;
};
// V4 prediction preview on unlabeled photos: boxes only, never a score.
export type PredictionResult = {
  measured: false; checkpoint_job_id: string; model_id: string; checkpoint_sha256: string; confidence: number;
  note: string; images: { image: string | null; source: string; boxes: (PreviewBox & { confidence: number })[] }[];
};

export type GenerationRun = {
  run_id: string; kind: "preview" | "dataset" | "dvc" | "predict";
  state: "queued" | "running" | "finished" | "failed" | "cancelled" | "interrupted";
  phase: string; done: number; total: number;
  error: { code: string; message: string } | null;
  result: PreviewResult | DatasetBuildResult | PredictionResult | Record<string, any> | null;
};

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : `Request failed (${status})`);
  }
}

let sessionToken: string | null = null;

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (method !== "GET") {
    if (sessionToken === null) {
      sessionToken = (await request<{ token: string }>("GET", "/api/session")).token;
    }
    headers["X-Foxtrot-Session"] = sessionToken;
  }
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {
    method, headers, body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, value?.detail ?? value);
  return value as T;
}

export const api = {
  health: () => request<{ status: string }>("GET", "/api/health"),
  capabilities: () => request<Capabilities>("GET", "/api/capabilities"),
  schemas: () => request<{ job: any; training_groups: Record<string, string> }>("GET", "/api/schemas"),
  checkpoints: () => request<CheckpointOption[]>("GET", "/api/checkpoints"),
  coverage: () => request<CoverageReport>("GET", "/api/coverage"),
  datasets: () => request<Dataset[]>("GET", "/api/datasets"),
  destinations: () => request<Destination[]>("GET", "/api/tracking/destinations"),
  checkDestination: (id: string) =>
    request<{ ok: boolean; message: string }>("POST", `/api/tracking/destinations/${encodeURIComponent(id)}/check`),
  saveDestination: (value: Record<string, unknown>) =>
    request<Destination>("PUT", `/api/tracking/destinations/${encodeURIComponent(String(value.destination_id))}`, value),
  importDataset: (value: Record<string, unknown>) => request<DatasetImportResult>("POST", "/api/datasets", value),
  models: () => request<ModelWeights[]>("GET", "/api/models"),
  installModel: (id: string) => request<ModelWeights>("POST", `/api/models/${encodeURIComponent(id)}/install`),
  jobs: () => request<JobView[]>("GET", "/api/jobs"),
  job: (id: string) => request<JobDetail>("GET", `/api/jobs/${id}`),
  events: (id: string, after: number) => request<JobEvent[]>("GET", `/api/jobs/${id}/events?after=${after}`),
  submit: (submissionKey: string, jobRequest: Record<string, unknown>) =>
    request<JobView>("POST", "/api/jobs", { submission_key: submissionKey, request: jobRequest }),
  cancel: (id: string) => request<JobView>("POST", `/api/jobs/${id}/cancel`),
  retrySync: (id: string) => request<{ queued: boolean }>("POST", `/api/jobs/${id}/retry-sync`),
  generationOptions: () => request<GenerationOptions>("GET", "/api/generation/options"),
  startPreview: (value: GenerationRequest) => request<GenerationRun>("POST", "/api/generation/preview", value),
  startGeneration: (value: GenerationRequest) => request<GenerationRun>("POST", "/api/generation/datasets", value),
  generationRun: (id: string) => request<GenerationRun>("GET", `/api/generation/runs/${id}`),
  cancelGeneration: (id: string) => request<GenerationRun>("POST", `/api/generation/runs/${id}/cancel`),
  previewImageUrl: (runId: string, image: string) => `/api/generation/runs/${runId}/images/${image}`,
  startDvcImport: (value: Record<string, unknown>) => request<GenerationRun>("POST", "/api/datasets/dvc", value),
  startPrediction: (value: Record<string, unknown>) => request<GenerationRun>("POST", "/api/predictions", value),
  studyOptions: () => request<{ training: string[]; recipe: string[]; objectives: string[] }>("GET", "/api/studies/options"),
  planStudy: (value: Record<string, unknown>) => request<{ planned_trials: number }>("POST", "/api/studies/plan", value),
  createStudy: (value: Record<string, unknown>) =>
    request<{ study_id: string; planned_trials: number }>("POST", "/api/studies", value),
  studies: () => request<any[]>("GET", "/api/studies"),
  study: (id: string) => request<any>("GET", `/api/studies/${id}`),
  stopStudy: (id: string, active: "finish" | "cancel") => request<any>("POST", `/api/studies/${id}/stop?active=${active}`),
  resumeStudy: (id: string) => request<any>("POST", `/api/studies/${id}/resume`),
  repeatStudy: (id: string, value: { top: number; seeds: number[] }) =>
    request<{ study_id: string; planned_trials: number }>("POST", `/api/studies/${id}/repeat`, value),
};

export const TERMINAL = new Set(["finished", "failed", "cancelled", "interrupted"]);

// The database stores UTC times like "2026-11-18 10:10:40"; show them in this computer's local time.
export function formatTime(value: string | null): string {
  if (!value) return "";
  const date = new Date(value.includes("T") ? value : `${value.replace(" ", "T")}Z`);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function describeError(error: unknown): string {
  if (error instanceof ApiError && Array.isArray(error.detail)) {
    return error.detail.map((item: any) => `${(item.loc ?? []).join(".")}: ${item.msg}`).join("; ");
  }
  return error instanceof Error ? error.message : String(error);
}
