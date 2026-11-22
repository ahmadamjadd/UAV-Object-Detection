import { createContext, useContext, useEffect, useState } from "react";
import { api, Capabilities } from "./api";
import { DatasetsPage } from "./pages/Datasets";
import { EvaluatePage } from "./pages/Evaluate";
import { HomePage } from "./pages/Home";
import { RunsPage } from "./pages/Runs";
import { SettingsPage } from "./pages/Settings";
import { TrainPage } from "./pages/Train";
import { go, Icon, Page, PAGES, usePolling, useRoute } from "./ui";

// What every page may need: capabilities, member name, attention count, and a draft request
// ("New experiment…" from Runs opens Train with an earlier job's request).
type Shared = {
  capabilities: Capabilities | null;
  member: string | null;
  setMember: (member: string | null) => void;
  attention: number;
  refreshAttention: () => void;
  draft: Record<string, unknown> | null;
  setDraft: (request: Record<string, unknown> | null) => void;
};
const SharedContext = createContext<Shared | null>(null);

// Access the shared app state from any page.
export function useShared(): Shared {
  const value = useContext(SharedContext);
  if (!value) throw new Error("useShared outside App");
  return value;
}

// Light/dark theme: saved choice, otherwise the system preference; dark is the design default.
function useTheme(): [boolean, () => void] {
  const [light, setLight] = useState(() => {
    try {
      const saved = localStorage.getItem("fx-theme");
      if (saved) return saved === "light";
    } catch { /* storage unavailable */ }
    return window.matchMedia?.("(prefers-color-scheme: light)").matches ?? false;
  });
  useEffect(() => {
    document.documentElement.classList.toggle("light", light);
  }, [light]);
  const toggle = () => setLight((value) => {
    try { localStorage.setItem("fx-theme", value ? "dark" : "light"); } catch { /* storage unavailable */ }
    return !value;
  });
  return [light, toggle];
}

// The app shell: sidebar navigation, top bar, and the current page.
export function App() {
  const { page, params } = useRoute();
  const [light, toggleTheme] = useTheme();
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [member, setMember] = useState<string | null>(null);
  const [attention, setAttention] = useState(0);
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null);

  // Count everything the Runs › Needs attention view lists.
  const refreshAttention = () => api.recovery().then((r) =>
    setAttention(r.interrupted.length + r.uploads.length + r.studies.length + r.holds.length), () => undefined);

  useEffect(() => {
    api.capabilities().then(setCapabilities, () => setCapabilities(null));
    api.profile().then((p) => setMember(p.member), () => undefined);
  }, []);
  usePolling(() => { api.health().then(() => setHealthy(true), () => setHealthy(false)); refreshAttention(); }, 5000);
  useEffect(() => { document.title = `${PAGES.find(([key]) => key === page)?.[1]} · Foxtrot`; }, [page]);

  const shared: Shared = { capabilities, member, setMember, attention, refreshAttention, draft, setDraft };
  const title = PAGES.find(([key]) => key === page)?.[1] ?? "Home";
  return (
    <SharedContext.Provider value={shared}>
      <div className="app">
        <aside>
          <div className="brand"><img src="./fox-logo.svg" alt="" /><span>Foxtrot</span></div>
          <button className="primary" onClick={() => go("train")}><Icon name="plus" size={16} /><span>Train a model</span></button>
          <nav aria-label="Main">
            <div className="navlbl">Workspace</div>
            {PAGES.map(([key, label]) => (
              <a key={key} href={`#/${key}`} className={key === page ? "on" : ""} aria-current={key === page ? "page" : undefined}>
                <Icon name={key} /><span>{label}</span>
                {key === "runs" && attention > 0 && <em className="cnt" title="Needs attention">{attention}</em>}
              </a>
            ))}
          </nav>
          <div className="foot">
            <button onClick={toggleTheme}><Icon name={light ? "moon" : "sun"} size={17} /><span>{light ? "Dark mode" : "Light mode"}</span></button>
            <a className="user" href="#/settings" title="Member name (Settings)">
              <div className="av">{(member ?? "?").slice(0, 1).toUpperCase()}</div>
              <div>{member ?? "Set your name"}<small>Team Foxtrot</small></div>
            </a>
          </div>
        </aside>
        <main>
          <div className="top">
            <h1>{title}</h1>
            <span className={`dot ${healthy ? "ok" : healthy === false ? "bad" : ""}`} />
            <span className="mute small">{healthy === false ? "API offline" : healthy ? "API online" : "Connecting…"}</span>
            {capabilities && !capabilities.real_training &&
              <span className="pill p-warn r" title="The worker runs a test adapter, not real YOLO training">Test mode: no real training</span>}
          </div>
          <div className="wrap" key={page}>
            {page === "home" && <HomePage />}
            {page === "datasets" && <DatasetsPage params={params} />}
            {page === "train" && <TrainPage params={params} />}
            {page === "runs" && <RunsPage params={params} />}
            {page === "evaluate" && <EvaluatePage params={params} />}
            {page === "settings" && <SettingsPage />}
          </div>
        </main>
      </div>
    </SharedContext.Provider>
  );
}

// Page names, for other files.
export type { Page };
