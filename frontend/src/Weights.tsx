import { useEffect, useState } from "react";
import { api, describeError, ModelWeights } from "./api";

// Setup card: shows which pretrained weights are installed and installs missing ones.
export function Weights() {
  const [models, setModels] = useState<ModelWeights[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState("");

  // Reload the installed/missing status of every known model.
  const load = () => api.models().then(setModels, (e) => setMessage(describeError(e)));
  useEffect(() => { load(); }, []);

  // Ask the backend to download one model's official weights, then refresh the list.
  async function install(modelId: string) {
    setBusy(modelId);
    setMessage("");
    try {
      const result = await api.installModel(modelId);
      setMessage(`Installed ${modelId}.pt (${Math.round((result.size_bytes ?? 0) / 1e6)} MB).`);
      load();
    } catch (error) { setMessage(describeError(error)); } finally { setBusy(null); }
  }

  return (
    <div className="card">
      <h2>Pretrained weights</h2>
      <p className="muted">Training starts from these unless the worker runs with --initialization scratch.</p>
      <ul className="plain">
        {models.map((m) => (
          <li key={m.model_id}>
            <strong>{m.model_id}</strong>{" "}
            <span className={m.status === "installed" ? "muted" : "error"}>{m.status}</span>
            {m.sha256 && <span className="muted"> · sha256 {m.sha256.slice(0, 12)}…</span>}{" "}
            {m.status !== "installed" && (
              <button onClick={() => install(m.model_id)} disabled={busy !== null}>
                {busy === m.model_id ? "Downloading…" : "Install"}
              </button>
            )}
          </li>
        ))}
      </ul>
      {message && <p className="muted">{message}</p>}
    </div>
  );
}
