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
};

export type Dataset = { dataset_id: string; class_names: string[]; validation_status: string };

export type Destination = {
  destination_id: string;
  kind: string;
  tracking_uri: string;
  browser_url: string | null;
  experiment_name: string;
  has_credentials: boolean;
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
  schemas: () => request<{ job: any }>("GET", "/api/schemas"),
  datasets: () => request<Dataset[]>("GET", "/api/datasets"),
  destinations: () => request<Destination[]>("GET", "/api/tracking/destinations"),
  saveDestination: (value: Record<string, unknown>) =>
    request<Destination>("PUT", `/api/tracking/destinations/${encodeURIComponent(String(value.destination_id))}`, value),
  jobs: () => request<JobView[]>("GET", "/api/jobs"),
  job: (id: string) => request<JobDetail>("GET", `/api/jobs/${id}`),
  events: (id: string, after: number) => request<JobEvent[]>("GET", `/api/jobs/${id}/events?after=${after}`),
  submit: (submissionKey: string, jobRequest: Record<string, unknown>) =>
    request<JobView>("POST", "/api/jobs", { submission_key: submissionKey, request: jobRequest }),
  cancel: (id: string) => request<JobView>("POST", `/api/jobs/${id}/cancel`),
};

export const TERMINAL = new Set(["finished", "failed", "cancelled", "interrupted"]);

export function describeError(error: unknown): string {
  if (error instanceof ApiError && Array.isArray(error.detail)) {
    return error.detail.map((item: any) => `${(item.loc ?? []).join(".")}: ${item.msg}`).join("; ");
  }
  return error instanceof Error ? error.message : String(error);
}
