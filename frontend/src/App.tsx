import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";
const THEME_STORAGE_KEY = "xai-traffic-theme";

type Severity = "LOW" | "MEDIUM" | "HIGH" | "SEVERE";
type Theme = "dark" | "light";

type ShapFactor = {
  feature: string;
  display_name?: string;
  value: number | string | boolean | null;
  shap: number;
  impact: string;
};

type CorridorDashboardItem = {
  corridor_name: string;
  display_name: string;
  congestion_pct: number;
  severity: Severity;
  explanation: string;
  top_factors: ShapFactor[];
  trace_id: string;
  confidence: number | null;
};

type DashboardResponse = {
  generated_at: string;
  count: number;
  scenario?: {
    hour: number;
    is_rain: boolean;
    is_festival: boolean;
  };
  summary?: {
    total_corridors: number;
    low: number;
    medium: number;
    high: number;
    severe: number;
    average_congestion: number;
    highest_congestion_corridor: string | null;
  };
  corridors: CorridorDashboardItem[];
};

type PredictionResponse = {
  prediction?: {
    congestion_pct?: number;
    label?: Severity;
  };
  congestion_pct?: number;
  severity?: Severity;
  label?: Severity;
  trace_id?: string;
  segment_id?: string;
  display_name?: string;
  explanation?: string;
  top_factors?: ShapFactor[];
  counterfactual?: string;
  confidence?: number | null;
  predicted_at?: string;
};

type FullTraceResponse = {
  trace_id: string;
  segment_id: string;
  predicted_at: string;
  prediction: {
    congestion_pct: number;
    label: Severity;
  };
  lime_explanation: string;
  shap_values: ShapFactor[];
  counterfactual: string;
  confidence: number | null;
};

type TraceListItem = {
  trace_id: string;
  segment_id: string;
  predicted_at: string;
  congestion_pct: number;
  label: Severity;
  explanation: string;
};

type TraceListResponse = {
  count: number;
  traces: TraceListItem[];
};

type Anomaly = {
  id: string;
  corridor_name: string;
  display_name: string;
  timestamp: string;
  congestion_pct: number;
  severity: Severity;
  confidence: number;
  reason: string;
};

type AnomaliesResponse = {
  count: number;
  anomalies: Anomaly[];
};

type ModelInfoResponse = {
  model_type: string;
  xai_method: string;
  target: string;
  features: string[];
  corridors: string[];
  prototype_note: string;
};

type ReasoningView = {
  traceId: string;
  corridorName: string;
  congestionPct: number;
  severity: Severity;
  explanation: string;
  factors: ShapFactor[];
  counterfactual: string;
  confidence?: number | null;
  predictedAt?: string;
};

type DemoPreset = {
  label: string;
  corridorName: string;
  hour: number;
  isRain: boolean;
  isFestival: boolean;
};

const DISPLAY_NAMES: Record<string, string> = {
  SG_Highway: "SG Highway",
  Ring_Road_132ft: "132 ft Ring Road",
  CG_Road: "CG Road",
  Ashram_Road: "Ashram Road",
  Sardar_Patel_Ring: "Sardar Patel Ring Road",
  Narol_Naroda: "Narol-Naroda",
  Maninagar: "Maninagar",
  Stadium_Motera: "Stadium Motera",
};

const MAP_ROADS = [
  { id: "Sardar_Patel_Ring", path: "M 76 163 C 93 74, 175 35, 286 39 C 404 43, 482 104, 493 198 C 505 291, 429 356, 316 370 C 205 384, 103 340, 69 260 C 51 218, 57 188, 76 163 Z", label: [391, 64] },
  { id: "SG_Highway", path: "M 126 334 C 117 282, 119 229, 127 178 C 134 130, 145 88, 161 51", label: [122, 242] },
  { id: "Ring_Road_132ft", path: "M 151 244 C 181 207, 228 188, 281 193 C 332 197, 372 223, 392 263 C 374 285, 350 300, 319 307", label: [344, 238] },
  { id: "Ashram_Road", path: "M 244 101 C 238 139, 237 181, 240 220 C 242 256, 239 292, 230 327", label: [236, 182] },
  { id: "CG_Road", path: "M 177 170 C 196 161, 218 157, 239 159", label: [191, 151] },
  { id: "Narol_Naroda", path: "M 361 344 C 378 300, 389 259, 400 218 C 414 168, 430 123, 456 79", label: [401, 205] },
  { id: "Maninagar", path: "M 294 263 C 316 277, 333 298, 344 324", label: [329, 306] },
  { id: "Stadium_Motera", path: "M 198 77 C 215 72, 231 72, 247 79", label: [211, 66] },
] as const;

const DEMO_PRESETS: DemoPreset[] = [
  { label: "Morning Rain — SG Highway", corridorName: "SG_Highway", hour: 8, isRain: true, isFestival: false },
  { label: "Festival Evening — Stadium Motera", corridorName: "Stadium_Motera", hour: 18, isRain: false, isFestival: true },
  { label: "Rush Hour — Sardar Patel Ring Road", corridorName: "Sardar_Patel_Ring", hour: 8, isRain: true, isFestival: false },
  { label: "Normal Afternoon — CG Road", corridorName: "CG_Road", hour: 14, isRain: false, isFestival: false },
];

const FEATURE_LABELS: Record<string, string> = {
  current_speed: "Current Speed",
  freeflow_speed: "Free-flow Speed",
  is_morning_rush: "Morning Rush",
  is_evening_rush: "Evening Rush",
  is_school_hour: "School Hour",
  is_rain: "Rain Condition",
  is_festival: "Festival/Event",
  hour: "Hour of Day",
  corridor_id: "Corridor Pattern",
  hour_sin: "Time of Day Pattern",
  hour_cos: "Time of Day Pattern",
  weekday: "Day of Week",
};

function formatFeatureName(feature: string) {
  return FEATURE_LABELS[feature]
    ?? feature.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatExplanation(explanation: string) {
  return Object.entries(FEATURE_LABELS).reduce(
    (formatted, [feature, label]) => formatted.replace(
      new RegExp(`\\b${feature}\\b`, "g"),
      label,
    ),
    explanation,
  );
}

function formatTime(value?: string) {
  if (!value) return "Just now";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "short",
  }).format(date);
}

function formatHourContext(hour: number) {
  if (hour >= 7 && hour <= 10) return `${String(hour).padStart(2, "0")}:00 Morning Rush`;
  if (hour >= 17 && hour <= 20) return `${String(hour).padStart(2, "0")}:00 Evening Rush`;
  if (hour < 6) return `${String(hour).padStart(2, "0")}:00 Overnight`;
  if (hour < 12) return `${String(hour).padStart(2, "0")}:00 Morning`;
  if (hour < 17) return `${String(hour).padStart(2, "0")}:00 Afternoon`;
  return `${String(hour).padStart(2, "0")}:00 Evening`;
}

function getInitialTheme(): Theme {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

async function getJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function StatusIcon({ type }: { type: "refresh" | "rain" | "event" | "history" | "alert" }) {
  const paths = {
    refresh: <><path d="M20 11a8.1 8.1 0 0 0-15.5-3M4 4v4h4"/><path d="M4 13a8.1 8.1 0 0 0 15.5 3M20 20v-4h-4"/></>,
    rain: <><path d="M16 13a4 4 0 1 0-7.6-1.7A3.5 3.5 0 0 0 8.5 18H17a3 3 0 0 0 0-6"/><path d="m10 21 1-2M15 21l1-2"/></>,
    event: <><path d="M8 2v3M16 2v3M3 9h18"/><rect x="3" y="4" width="18" height="17" rx="2"/><path d="m9 15 2 2 4-4"/></>,
    history: <><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/></>,
    alert: <><path d="M10.3 3.7 2.4 17.5A2 2 0 0 0 4.1 20h15.8a2 2 0 0 0 1.7-2.5L13.7 3.7a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/></>,
  };
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[type]}</svg>;
}

function Toggle({
  checked,
  onChange,
  label,
  helper,
  icon,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  helper: string;
  icon: "rain" | "event";
}) {
  return (
    <label className={`scenario-toggle ${checked ? "is-on" : ""}`}>
      <span className="control-icon"><StatusIcon type={icon} /></span>
      <span className="toggle-copy"><strong>{label}</strong><small>{helper}</small></span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-track" aria-hidden="true"><span /></span>
    </label>
  );
}

function Skeleton({ className = "" }: { className?: string }) {
  return <span className={`skeleton ${className}`} />;
}

function App() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [hour, setHour] = useState(8);
  const [isRain, setIsRain] = useState(true);
  const [isFestival, setIsFestival] = useState(false);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [activeCorridor, setActiveCorridor] = useState("SG_Highway");
  const [reasoning, setReasoning] = useState<ReasoningView | null>(null);
  const [dashboardLoading, setDashboardLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [feedsLoading, setFeedsLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);
  const [error, setError] = useState("");

  const corridorLookup = useMemo(
    () => new Map(dashboard?.corridors.map((corridor) => [corridor.corridor_name, corridor]) ?? []),
    [dashboard],
  );

  const loadDashboard = useCallback(async () => {
    setDashboardLoading(true);
    setError("");
    try {
      const query = new URLSearchParams({
        hour: String(hour),
        is_rain: String(isRain),
        is_festival: String(isFestival),
      });
      const data = await getJson<DashboardResponse>(`/api/v1/dashboard?${query}`);
      setApiOnline(true);
      setDashboard(data);
      const active = data.corridors.find((item) => item.corridor_name === activeCorridor) ?? data.corridors[0];
      if (active) {
        setActiveCorridor(active.corridor_name);
        setReasoning({
          traceId: active.trace_id,
          corridorName: active.corridor_name,
          congestionPct: active.congestion_pct,
          severity: active.severity,
          explanation: active.explanation,
          factors: active.top_factors,
          counterfactual: "Select this corridor to generate a focused counterfactual analysis.",
          confidence: active.confidence,
          predictedAt: data.generated_at,
        });
      }
      const traceData = await getJson<TraceListResponse>("/api/v1/traces?limit=10");
      setTraces(traceData.traces);
    } catch {
      setApiOnline(false);
      setError("Live traffic data is unavailable. Confirm the FastAPI service is running on port 8000.");
    } finally {
      setDashboardLoading(false);
    }
  }, [activeCorridor, hour, isFestival, isRain]);

  const loadFeeds = useCallback(async () => {
    setFeedsLoading(true);
    const [anomalyResult, traceResult, modelResult] = await Promise.allSettled([
      getJson<AnomaliesResponse>("/api/v1/anomalies"),
      getJson<TraceListResponse>("/api/v1/traces?limit=10"),
      getJson<ModelInfoResponse>("/api/v1/model-info"),
    ]);

    if (anomalyResult.status === "fulfilled") {
      setAnomalies(anomalyResult.value.anomalies);
    }
    if (traceResult.status === "fulfilled") {
      setTraces(traceResult.value.traces);
    }
    if (modelResult.status === "fulfilled") {
      setModelInfo(modelResult.value);
    }
    if (
      anomalyResult.status === "rejected"
      && traceResult.status === "rejected"
    ) {
      setError("Some operational feeds could not be loaded.");
    }
    setFeedsLoading(false);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // The selected theme still applies when browser storage is unavailable.
    }
  }, [theme]);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => {
      void loadDashboard();
      void loadFeeds();
    }, 0);
    return () => window.clearTimeout(initialLoad);
    // Initial load only; scenario changes are applied by the refresh control.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function selectCorridor(
    corridorName: string,
    scenario?: {
      hour: number;
      isRain: boolean;
      isFestival: boolean;
    },
  ) {
    const selectedHour = scenario?.hour ?? hour;
    const selectedRain = scenario?.isRain ?? isRain;
    const selectedFestival = scenario?.isFestival ?? isFestival;
    setActiveCorridor(corridorName);
    setDetailLoading(true);
    setError("");
    try {
      const data = await getJson<PredictionResponse>("/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corridor_name: corridorName,
          hour: selectedHour,
          is_rain: selectedRain,
          is_festival: selectedFestival,
        }),
      });
      const congestionPct = data.prediction?.congestion_pct ?? data.congestion_pct ?? 0;
      const severity = data.prediction?.label ?? data.severity ?? data.label ?? "LOW";
      const resultCorridor = data.display_name || data.segment_id || corridorName;
      setReasoning({
        traceId: data.trace_id ?? "pending-trace",
        corridorName: resultCorridor,
        congestionPct,
        severity,
        explanation: formatExplanation(data.explanation ?? "The model returned a prediction without a text explanation."),
        factors: data.top_factors ?? [],
        counterfactual: data.counterfactual ?? "No counterfactual suggestion was returned for this prediction.",
        confidence: data.confidence,
        predictedAt: data.predicted_at,
      });
      const traceData = await getJson<TraceListResponse>("/api/v1/traces?limit=10");
      setTraces(traceData.traces);
    } catch {
      setError("The corridor prediction could not be generated.");
    } finally {
      setDetailLoading(false);
    }
  }

  async function runPreset(preset: DemoPreset) {
    setActiveCorridor(preset.corridorName);
    setHour(preset.hour);
    setIsRain(preset.isRain);
    setIsFestival(preset.isFestival);
    await selectCorridor(preset.corridorName, preset);
    document.querySelector(".reasoning-panel")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  async function selectTrace(traceId: string) {
    setDetailLoading(true);
    setError("");
    try {
      const data = await getJson<FullTraceResponse>(`/api/v1/traces/${traceId}`);
      setActiveCorridor(data.segment_id);
      setReasoning({
        traceId: data.trace_id,
        corridorName: data.segment_id,
        congestionPct: data.prediction.congestion_pct,
        severity: data.prediction.label,
        explanation: data.lime_explanation,
        factors: data.shap_values,
        counterfactual: data.counterfactual,
        confidence: data.confidence,
        predictedAt: data.predicted_at,
      });
      document.querySelector(".reasoning-panel")?.scrollIntoView({ behavior: "smooth", block: "center" });
    } catch {
      setError("The selected reasoning trace is no longer available.");
    } finally {
      setDetailLoading(false);
    }
  }

  const maxShap = Math.max(1, ...(reasoning?.factors.map((factor) => Math.abs(factor.shap)) ?? []));
  const severeCount = dashboard?.corridors.filter((item) => item.severity === "SEVERE").length ?? 0;
  const networkAverage = dashboard?.corridors.length
    ? dashboard.corridors.reduce((sum, item) => sum + item.congestion_pct, 0) / dashboard.corridors.length
    : 0;
  const severityCounts = useMemo(
    () => (["LOW", "MEDIUM", "HIGH", "SEVERE"] as Severity[]).reduce(
      (counts, severity) => ({
        ...counts,
        [severity]: dashboard?.corridors.filter((item) => item.severity === severity).length ?? 0,
      }),
      { LOW: 0, MEDIUM: 0, HIGH: 0, SEVERE: 0 } as Record<Severity, number>,
    ),
    [dashboard],
  );
  const corridorRanking = useMemo(
    () => [...(dashboard?.corridors ?? [])].sort(
      (left, right) => right.congestion_pct - left.congestion_pct,
    ),
    [dashboard],
  );
  const highestCorridor = corridorRanking[0];
  const displayedAverage = dashboard?.summary?.average_congestion ?? networkAverage;
  const displayedScenario = dashboard?.scenario ?? {
    hour,
    is_rain: isRain,
    is_festival: isFestival,
  };
  const activePreset = DEMO_PRESETS.find(
    (preset) => (
      preset.corridorName === activeCorridor
      && preset.hour === hour
      && preset.isRain === isRain
      && preset.isFestival === isFestival
    ),
  );

  return (
    <main className={`app-shell ${theme}-theme`} data-theme={theme}>
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div>
            <p className="eyebrow">Ahmedabad XAI Traffic Intelligence</p>
            <h1>Explainable Traffic Congestion Prediction</h1>
            <p className="subtitle">Simulate traffic scenarios, predict congestion using XGBoost, and understand every result through SHAP reasoning.</p>
          </div>
        </div>
        <div className="header-actions">
          <div className="system-status">
            <span className="status-dot" />
            <span>{apiOnline ? "Backend connected" : "Connecting"}</span>
            <small>XGBoost · SHAP</small>
          </div>
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-pressed={theme === "light"}
          >
            <span className="theme-toggle-icon" aria-hidden="true">
              {theme === "dark" ? (
                <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
              ) : (
                <svg viewBox="0 0 24 24"><path d="M20 15.2A8.5 8.5 0 0 1 8.8 4a8.5 8.5 0 1 0 11.2 11.2Z" /></svg>
              )}
            </span>
            <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>
            <i aria-hidden="true"><span /></i>
          </button>
        </div>
      </header>

      <section className="hero-explainer panel" aria-labelledby="how-it-works-title">
        <div className="hero-flow">
          <div className="hero-flow-copy">
            <p className="section-kicker">What this system does</p>
            <h2 id="how-it-works-title">A guided explainable-AI traffic demo</h2>
            <p className="system-summary">
              This prototype predicts congestion for Ahmedabad road corridors using scenario inputs such as corridor, time, rain, and event condition. The backend ML model predicts congestion percentage, and SHAP explains which factors influenced the prediction.
            </p>
          </div>
          <div className="flow-steps" aria-label="Explainable prediction workflow">
            {[
              "Traffic Scenario",
              "XGBoost Model",
              "SHAP Explanation",
              "Decision Insight",
            ].map((step, index) => (
              <div className="flow-step" key={step}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{step}</strong>
                {index < 3 && <i aria-hidden="true">→</i>}
              </div>
            ))}
          </div>
        </div>
        <div className="explanation-cards">
          <article>
            <span>Input Scenario</span>
            <strong>Describe the traffic situation</strong>
            <p>Choose the corridor, hour, rain, and event context that the model should evaluate.</p>
          </article>
          <article>
            <span>Prediction</span>
            <strong>XGBoost estimates congestion</strong>
            <p>A machine learning model used to estimate congestion percentage and severity.</p>
          </article>
          <article>
            <span>Reasoning Trace</span>
            <strong>SHAP explains the result</strong>
            <p>A saved explanation of one prediction for audit and review.</p>
          </article>
        </div>
      </section>

      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button onClick={() => setError("")} aria-label="Dismiss error">Dismiss</button>
        </div>
      )}

      <section className="scenario-workbench panel" aria-label="Scenario controls">
        <div className="scenario-intro">
          <p className="step-label">Step 1</p>
          <h2>Build Scenario</h2>
          <p>Choose a corridor and conditions to simulate a traffic situation.</p>
          <ol className="usage-guide">
            <li><span>1</span>Select corridor and scenario conditions</li>
            <li><span>2</span>Run prediction</li>
            <li><span>3</span>Read the SHAP reasoning trace</li>
          </ol>
        </div>
        <div className="scenario-controls">
          <label className="corridor-control">
            <span>Corridor</span>
            <select value={activeCorridor} onChange={(event) => setActiveCorridor(event.target.value)}>
              {MAP_ROADS.map((road) => (
                <option key={road.id} value={road.id}>{DISPLAY_NAMES[road.id]}</option>
              ))}
            </select>
            <small>Road segment to simulate</small>
          </label>
          <label className="hour-control">
            <span className="hour-label"><strong>Hour</strong><small>Time of day affects rush-hour traffic</small></span>
            <input type="range" min="0" max="23" value={hour} onChange={(event) => setHour(Number(event.target.value))} />
            <output>{formatHourContext(hour)}</output>
          </label>
          <Toggle checked={isRain} onChange={setIsRain} label="Rain" helper="Weather may increase congestion" icon="rain" />
          <Toggle checked={isFestival} onChange={setIsFestival} label="Festival / event" helper="Events affect corridor pressure" icon="event" />
        </div>
        <div className="scenario-actions">
          <button className="primary-action" onClick={() => void selectCorridor(activeCorridor)} disabled={detailLoading}>
            {detailLoading ? "Running model" : "Run prediction"}
          </button>
          <button className="refresh-button" onClick={() => void loadDashboard()} disabled={dashboardLoading}>
            <StatusIcon type="refresh" />
            {dashboardLoading ? "Updating network" : "Refresh live dashboard"}
          </button>
          <div className="scenario-summary">
            <span><strong>{dashboard?.count ?? "—"}</strong> corridors</span>
            <span><strong>{networkAverage.toFixed(1)}%</strong> network avg.</span>
            <span><strong>{severeCount}</strong> severe</span>
          </div>
        </div>
        <div className="preset-row">
          <div className="preset-intro">
            <span>Quick demo presets</span>
            <small>Use a preset for a quick demo, or build your own traffic scenario manually.</small>
          </div>
          <div className="preset-buttons">
            {DEMO_PRESETS.map((preset) => (
              <button
                className={activePreset?.label === preset.label ? "is-active" : ""}
                key={preset.label}
                onClick={() => void runPreset(preset)}
                disabled={detailLoading}
                aria-pressed={activePreset?.label === preset.label}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="analytics-panel panel" aria-labelledby="analytics-title">
        <div className="analytics-heading">
          <div>
            <p className="section-kicker">Dashboard analytics</p>
            <h2 id="analytics-title">Network summary</h2>
          </div>
          <span>{dashboardLoading ? "Loading dashboard predictions..." : `Updated ${formatTime(dashboard?.generated_at)}`}</span>
        </div>

        {dashboardLoading && !dashboard ? (
          <div className="analytics-loading">
            <div className="loading-copy">
              <span className="loading-spinner" />
              <div><strong>Loading dashboard predictions...</strong><small>Running the model across all eight corridors.</small></div>
            </div>
            <Skeleton className="analytics-skeleton" />
          </div>
        ) : dashboard ? (
          <>
            <div className="summary-cards">
              <article>
                <span>Average congestion</span>
                <strong>{displayedAverage.toFixed(1)}%</strong>
                <small>Across {dashboard.count} corridors</small>
              </article>
              <article>
                <span>Highest congestion</span>
                <strong>{highestCorridor?.display_name ?? "—"}</strong>
                <small>{highestCorridor ? `${highestCorridor.congestion_pct.toFixed(1)}% · ${highestCorridor.severity}` : "No data"}</small>
              </article>
              <article className="severity-summary-card">
                <span>Corridor severity</span>
                <div className="severity-counts">
                  {(["LOW", "MEDIUM", "HIGH", "SEVERE"] as Severity[]).map((severity) => (
                    <em className={severity.toLowerCase()} key={severity}>
                      <b>{severityCounts[severity]}</b>{severity}
                    </em>
                  ))}
                </div>
              </article>
              <article>
                <span>Current scenario</span>
                <strong>{formatHourContext(displayedScenario.hour)}</strong>
                <small>
                  {displayedScenario.is_rain ? "Rain on" : "No rain"} · {displayedScenario.is_festival ? "Event active" : "No event"}
                </small>
              </article>
            </div>

            <div className="analytics-grid">
              <article className="distribution-card">
                <div className="visual-title">
                  <div><strong>Severity distribution</strong><span>How the network is classified</span></div>
                  <small>{dashboard.count} total</small>
                </div>
                <div className="distribution-bar" aria-label="Corridor severity distribution">
                  {(["LOW", "MEDIUM", "HIGH", "SEVERE"] as Severity[]).map((severity) => (
                    severityCounts[severity] > 0 && (
                      <span
                        className={severity.toLowerCase()}
                        key={severity}
                        style={{ width: `${(severityCounts[severity] / Math.max(dashboard.count, 1)) * 100}%` }}
                        title={`${severity}: ${severityCounts[severity]}`}
                      />
                    )
                  ))}
                </div>
                <div className="distribution-legend">
                  {(["LOW", "MEDIUM", "HIGH", "SEVERE"] as Severity[]).map((severity) => (
                    <span key={severity}><i className={severity.toLowerCase()} />{severity} <b>{severityCounts[severity]}</b></span>
                  ))}
                </div>
              </article>

              <article className="ranking-card">
                <div className="visual-title">
                  <div><strong>Corridor ranking</strong><span>Highest predicted congestion first</span></div>
                </div>
                <div className="ranking-list">
                  {corridorRanking.map((corridor, index) => (
                    <button key={corridor.corridor_name} onClick={() => void selectCorridor(corridor.corridor_name)}>
                      <span className="rank-number">{index + 1}</span>
                      <span className="rank-name">{corridor.display_name}</span>
                      <span className="rank-track"><i className={corridor.severity.toLowerCase()} style={{ width: `${corridor.congestion_pct}%` }} /></span>
                      <strong>{corridor.congestion_pct.toFixed(1)}%</strong>
                    </button>
                  ))}
                </div>
              </article>
            </div>
          </>
        ) : (
          <div className="analytics-empty">Dashboard predictions are unavailable.</div>
        )}
      </section>

      <section className="primary-grid">
        <article className="panel map-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Ahmedabad overview</p>
              <h2>Corridor map</h2>
              <p className="map-note">Abstract corridor map — not to scale.</p>
            </div>
            <div className="map-legend">
              {(["LOW", "MEDIUM", "HIGH", "SEVERE"] as Severity[]).map((severity) => (
                <span key={severity}><i className={`severity-fill ${severity.toLowerCase()}`} />{severity}</span>
              ))}
            </div>
          </div>

          <div className={`city-map ${dashboardLoading ? "is-loading" : ""}`}>
            <div className="map-meta">
              <span>West Ahmedabad</span>
              <span>East Ahmedabad</span>
            </div>
            <svg viewBox="0 0 540 400" role="img" aria-label="Abstract map of Ahmedabad traffic corridors">
              <defs>
                <pattern id="minorGrid" width="20" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth=".5" />
                </pattern>
              </defs>
              <rect width="540" height="400" className="map-grid" fill="url(#minorGrid)" />
              <path className="river" d="M272 -12 C 258 35, 275 72, 263 112 C 251 153, 268 190, 257 231 C 246 273, 260 313, 249 352 C 244 371, 245 390, 251 412" />
              <text className="river-label" x="277" y="131" transform="rotate(88 277 131)">Sabarmati River</text>
              <g className="minor-roads">
                <path d="M45 112 C 114 123, 174 112, 233 126 C 315 146, 390 138, 499 111" />
                <path d="M42 209 C 116 194, 181 181, 238 185 C 315 190, 389 182, 500 202" />
                <path d="M55 287 C 131 264, 199 258, 259 273 C 337 292, 407 317, 488 308" />
                <path d="M187 38 C 180 104, 178 165, 184 226 C 189 286, 204 340, 220 386" />
                <path d="M351 31 C 338 89, 344 140, 359 188 C 378 248, 380 314, 369 375" />
              </g>
              <g className="corridor-roads">
                {MAP_ROADS.map((road) => {
                  const corridor = corridorLookup.get(road.id);
                  const severity = corridor?.severity ?? "LOW";
                  const isActive = activeCorridor === road.id;
                  return (
                    <g
                      key={road.id}
                      className={[
                        "map-corridor",
                        severity.toLowerCase(),
                        isActive ? "active" : "",
                        road.id === "Sardar_Patel_Ring" ? "outer-ring" : "",
                        road.id === "Ring_Road_132ft" ? "inner-ring" : "",
                      ].filter(Boolean).join(" ")}
                      onClick={() => void selectCorridor(road.id)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") void selectCorridor(road.id);
                      }}
                      aria-label={`${DISPLAY_NAMES[road.id]}, ${severity} congestion`}
                    >
                      <path className="road-hitbox" d={road.path} />
                      <path className="road-base" d={road.path} />
                      <path className="road-status" d={road.path} />
                      <circle cx={road.label[0]} cy={road.label[1]} r={isActive ? 5 : 3.5} />
                      <text x={road.label[0] + 9} y={road.label[1] + 4}>{DISPLAY_NAMES[road.id]}</text>
                    </g>
                  );
                })}
              </g>
            </svg>
            <div className="map-readout">
              <span>Selected corridor</span>
              <strong>{DISPLAY_NAMES[activeCorridor]}</strong>
              <em className={`severity-text ${(corridorLookup.get(activeCorridor)?.severity ?? "LOW").toLowerCase()}`}>
                {corridorLookup.get(activeCorridor)?.congestion_pct?.toFixed(1) ?? "—"}%
              </em>
            </div>
          </div>
        </article>

        <article className="panel reasoning-panel">
          <div className="panel-header">
            <div>
              <p className="step-label">Step 2</p>
              <h2>View Prediction</h2>
              <p className="result-corridor">{reasoning ? DISPLAY_NAMES[reasoning.corridorName] ?? reasoning.corridorName : "Run a traffic scenario"}</p>
            </div>
            <div className="reasoning-actions">
              {reasoning && <span className={`severity-badge ${reasoning.severity.toLowerCase()}`}>{reasoning.severity}</span>}
              <button
                className="run-prediction-button"
                onClick={() => void selectCorridor(activeCorridor)}
                disabled={detailLoading}
              >
                {detailLoading ? "Running model" : "Run prediction"}
              </button>
            </div>
          </div>

          {detailLoading ? (
            <div className="reasoning-loading">
              <div className="loading-copy">
                <span className="loading-spinner" />
                <div><strong>Generating reasoning trace...</strong><small>Calculating prediction and SHAP contributions.</small></div>
              </div>
              <Skeleton className="metric-skeleton" />
              <Skeleton />
              <Skeleton />
              <Skeleton className="large" />
            </div>
          ) : reasoning ? (
            <>
              <div className="prediction-headline">
                <div>
                  <span>Predicted congestion</span>
                  <strong>{reasoning.congestionPct.toFixed(1)}<small>%</small></strong>
                </div>
                <div className="confidence-block">
                  <span>Prototype model output</span>
                  <p className="confidence-status">Model response received</p>
                </div>
              </div>

              <div className="explanation-block">
                <span className="block-label">Model interpretation</span>
                <p>{reasoning.explanation}</p>
              </div>

              <div className="shap-section">
                <div className="block-title-row">
                  <span className="step-label">Step 3 · Understand SHAP Reasoning</span>
                  <small>Feature contribution</small>
                </div>
                <p className="explanation-note">
                  SHAP values show how much each feature influenced this specific prediction.
                  Positive values increased predicted congestion; negative values reduced it.
                </p>
                <div className="shap-axis"><i /><span /></div>
                <div className="shap-list">
                  {reasoning.factors.map((factor, index) => {
                    const width = Math.max(5, (Math.abs(factor.shap) / maxShap) * 47);
                    const positive = factor.shap >= 0;
                    return (
                      <div className="shap-row" key={`${factor.feature}-${index}`}>
                        <div className="factor-meta">
                          <strong>{factor.display_name ?? formatFeatureName(factor.feature)}</strong>
                          <span>Value: {String(factor.value)} · {factor.shap >= 0 ? "Increased predicted congestion" : "Reduced predicted congestion"}</span>
                        </div>
                        <div className="shap-visual">
                          <span className="axis-center" />
                          <i
                            className={positive ? "positive" : "negative"}
                            style={{
                              width: `${width}%`,
                              left: positive ? "50%" : `${50 - width}%`,
                            }}
                          />
                        </div>
                        <b className={positive ? "positive-value" : "negative-value"}>
                          {positive ? "+" : ""}{factor.shap.toFixed(3)}
                        </b>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="counterfactual-block">
                <span className="block-label">What Could Reduce Congestion?</span>
                <small>Counterfactual explains what would need to change to reduce congestion.</small>
                <p>{reasoning.counterfactual}</p>
              </div>

              <footer className="trace-footer">
                <span>Trace <code>{reasoning.traceId.slice(0, 8)}</code></span>
                <span>{formatTime(reasoning.predictedAt)}</span>
              </footer>
            </>
          ) : (
            <div className="empty-state">
              <strong>No prediction yet</strong>
              <span>Choose a corridor and conditions above, then select Run Prediction.</span>
            </div>
          )}
        </article>
      </section>

      <section className="secondary-heading" aria-labelledby="supporting-dashboard-title">
        <div>
          <p className="section-kicker">Explore the wider system</p>
          <h2 id="supporting-dashboard-title">Supporting dashboard</h2>
          <p>Once you understand the prediction, use these views to compare corridors, inspect the network, and revisit saved traces.</p>
        </div>
        <span>Secondary views</span>
      </section>

      <section className="corridor-section">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Live corridor overview</p>
            <h2>All eight corridors</h2>
          </div>
          <span>Updated {formatTime(dashboard?.generated_at)}</span>
        </div>
        <div className="corridor-grid">
          {dashboardLoading && !dashboard
            ? Array.from({ length: 8 }, (_, index) => (
                <div className="corridor-card loading-card" key={index}><Skeleton /><Skeleton className="large" /><Skeleton /></div>
              ))
            : dashboard?.corridors.map((corridor) => (
                <button
                  className={`corridor-card ${corridor.severity.toLowerCase()} ${activeCorridor === corridor.corridor_name ? "active" : ""}`}
                  key={corridor.corridor_name}
                  onClick={() => void selectCorridor(corridor.corridor_name)}
                >
                  <div className="corridor-card-top">
                    <span className="road-index">{String(dashboard.corridors.indexOf(corridor) + 1).padStart(2, "0")}</span>
                    <span className={`severity-badge ${corridor.severity.toLowerCase()}`}>{corridor.severity}</span>
                  </div>
                  <div className="corridor-metric">
                    <h3>{corridor.display_name}</h3>
                    <strong>{corridor.congestion_pct.toFixed(1)}<small>%</small></strong>
                  </div>
                  <p>{corridor.explanation}</p>
                </button>
              ))}
        </div>
      </section>

      <section className="operations-grid">
        <article className="panel feed-panel anomaly-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Secondary information</p>
              <h2>Traffic Alerts</h2>
            </div>
            <span className="panel-icon alert"><StatusIcon type="alert" /></span>
          </div>
          <div className="feed-list">
            {feedsLoading
              ? Array.from({ length: 4 }, (_, index) => <Skeleton className="feed-skeleton" key={index} />)
              : anomalies.length === 0
                ? <div className="feed-empty"><strong>No active anomalies</strong><span>The monitored network is within expected operating ranges.</span></div>
                : anomalies.map((anomaly) => (
                  <button className="alert-item" key={anomaly.id} onClick={() => void selectCorridor(anomaly.corridor_name)}>
                    <span className={`alert-rail ${anomaly.severity.toLowerCase()}`} />
                    <div className="feed-copy">
                      <div><strong>{anomaly.display_name}</strong><time>{formatTime(anomaly.timestamp)}</time></div>
                      <p>{anomaly.reason}</p>
                    </div>
                    <div className="alert-metric">
                      <strong>{anomaly.congestion_pct.toFixed(1)}%</strong>
                      <span className={`severity-text ${anomaly.severity.toLowerCase()}`}>{anomaly.severity}</span>
                    </div>
                  </button>
                ))}
          </div>
        </article>

        <article className="panel feed-panel history-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Saved audit history</p>
              <h2>Recent Saved Explanations</h2>
            </div>
            <span className="panel-icon"><StatusIcon type="history" /></span>
          </div>
          <div className="feed-list">
            {feedsLoading
              ? Array.from({ length: 4 }, (_, index) => <Skeleton className="feed-skeleton" key={index} />)
              : traces.length === 0
                ? <div className="feed-empty"><strong>No recent traces yet.</strong><span>Run a prediction to create one.</span></div>
                : traces.map((trace) => (
                  <button className="trace-item" key={trace.trace_id} onClick={() => void selectTrace(trace.trace_id)}>
                    <span className={`trace-node ${trace.label.toLowerCase()}`} />
                    <div className="feed-copy">
                      <div>
                        <strong>{DISPLAY_NAMES[trace.segment_id] ?? trace.segment_id}</strong>
                        <time>{formatTime(trace.predicted_at)}</time>
                      </div>
                      <p>{trace.explanation}</p>
                    </div>
                    <div className="trace-metric">
                      <strong>{trace.congestion_pct.toFixed(1)}%</strong>
                      <span className={`severity-text ${trace.label.toLowerCase()}`}>{trace.label}</span>
                      <em>Open trace</em>
                    </div>
                  </button>
                ))}
          </div>
        </article>
      </section>

      <section className="model-info-panel panel">
        <div>
          <p className="section-kicker">Model Info</p>
          <h2>Technology behind the prediction</h2>
        </div>
        <div className="model-info-items">
          <article>
            <strong>XGBoost</strong>
            <p>A machine learning model used to estimate congestion percentage.</p>
          </article>
          <article>
            <strong>SHAP</strong>
            <p>An explainability method that shows which features influenced the prediction.</p>
          </article>
          <article>
            <strong>Reasoning trace</strong>
            <p>A saved explanation of one prediction for audit and review.</p>
          </article>
          {modelInfo && (
            <article>
              <strong>{modelInfo.features.length} model features</strong>
              <p>{modelInfo.prototype_note}</p>
            </article>
          )}
        </div>
      </section>

      <footer className="app-footer">
        <span>This is a research prototype using processed Ahmedabad-style traffic features. It demonstrates explainable traffic prediction, not a live city deployment.</span>
        <span>Model xgboost_v1 · Reasoning layer SHAP · API {API_BASE.replace("http://", "")}</span>
      </footer>
    </main>
  );
}

export default App;
