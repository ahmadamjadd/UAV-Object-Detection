// Follow one job live: its details and only the events added since the last look.
import { useCallback, useRef, useState } from "react";
import { api, JobDetail, JobEvent, TERMINAL } from "./api";
import { usePolling } from "./ui";

// Job details + accumulated events, refreshed every second while the job is active.
export function useLiveJob(jobId: string | null) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const cursor = useRef(0);
  const current = useRef(jobId);
  if (current.current !== jobId) {  // switched to another job: start over
    current.current = jobId;
    cursor.current = 0;
  }
  const refresh = useCallback(() => {
    if (!jobId) return;
    api.job(jobId).then((detail) => { if (current.current === jobId) setJob(detail); }, () => undefined);
    const after = cursor.current;
    api.events(jobId, after).then((fresh) => {
      if (current.current !== jobId) return;
      if (after === 0) setEvents(fresh);
      else if (fresh.length) setEvents((old) => [...old, ...fresh].slice(-2000));
      if (fresh.length) cursor.current = fresh[fresh.length - 1].sequence;
    }, () => undefined);
  }, [jobId]);
  const active = !job || !TERMINAL.has(job.execution_state) || job.tracking_state !== "synced";
  usePolling(refresh, active ? 1000 : 5000, Boolean(jobId));
  return { job: job?.job_id === jobId ? job : null, events, refresh };
}

// Epoch progress, the mAP50-95 history, and an estimate of the time left, from progress events.
export function progressOf(events: JobEvent[]) {
  const progress = events.filter((e) => e.event_type === "progress" && typeof e.payload.epoch === "number");
  const last = progress[progress.length - 1]?.payload;
  const history = progress.map((e) => Number(e.payload["metrics/mAP50-95"])).filter((v) => Number.isFinite(v));
  let minutesLeft: number | null = null;
  if (progress.length >= 2 && last) {
    const first = new Date(progress[0].timestamp).getTime(), latest = new Date(progress[progress.length - 1].timestamp).getTime();
    const perEpoch = (latest - first) / (progress.length - 1);
    minutesLeft = Math.max(0, Math.round(((Number(last.epochs) - Number(last.epoch)) * perEpoch) / 60000));
  }
  return {
    epoch: last ? Number(last.epoch) : 0, total: last ? Number(last.epochs) : 0,
    map: history.length ? history[history.length - 1] : null, history, minutesLeft,
  };
}

// One event as a short readable line.
export function describeEvent(event: JobEvent): string {
  const p = event.payload;
  if (event.event_type === "state_changed") return `${p.domain === "tracking" ? "upload" : "job"}: ${p.previous ?? "—"} → ${p.state}`;
  if (event.event_type === "progress") {
    const map = p["metrics/mAP50-95"];
    return `epoch ${p.epoch}${p.epochs ? ` / ${p.epochs}` : ""}${typeof map === "number" ? ` · mAP50-95 ${map.toFixed(3)}` : ""}`;
  }
  if (typeof p.text === "string") return `${p.channel}: ${p.text.trim().slice(0, 200)}`;
  if (p.action) return String(p.action).replace(/_/g, " ") + (p.source ? ` (${p.source})` : "");
  return JSON.stringify(p).slice(0, 200);
}
