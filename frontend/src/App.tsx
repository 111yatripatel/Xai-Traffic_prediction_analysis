import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

type Severity = "LOW" | "MEDIUM" | "HIGH" | "SEVERE";

type ShapFactor = {
  feature: string;
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
  prediction: {
    congestion_pct: number;
    label: Severity;
  };
  trace_id: string;
  segment_id: string;
  explanation: string;
  top_factors: ShapFactor[];
  counterfactual: string;
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
  Narol_Naroda: "Narol–Naroda",
  Maninagar: "Maninagar",
  Stadium_Motera: "Stadium Motera",
};

const MAP_ROADS = [
  { id: "Sardar_Patel_Ring", path: "M 74 161 C 84 61, 186 23, 322 50 C 448 73, 512 164, 478 262 C 441 361, 294 390, 160 346 C 78 319, 39 247, 74 161 Z", label: [393, 70] },
  { id: "SG_Highway", path: "M 119 317 C 165 274, 183 216, 207 153 C 224 108, 254 75, 291 48", label: [115, 286] },
  { id: "Ring_Road_132ft", path: "M 121 250 C 179 213, 245 199, 316 208 C 378 216, 423 249, 461 301", label: [337, 237] },
  { id: "Ashram_Road", path: "M 263 97 C 252 145, 258 196, 245 239 C 232 279, 213 311, 202 348", label: [191, 173] },
  { id: "CG_Road", path: "M 232 174 C 270 165, 299 169, 328 191", label: [278, 153] },
  { id: "Narol_Naroda", path: "M 116 327 C 190 310, 259 294, 329 261 C 373 240, 405 204, 447 175", label: [354, 283] },
  { id: "Maninagar", path: "M 287 248 C 313 268, 332 296, 344 331", label: [321, 337] },
  { id: "Stadium_Motera", path: "M 226 87 C 242 103, 262 112, 285 115", label: [310, 109] },
] as const;

const DEMO_PRESETS: DemoPreset[] = [
  {
    label: "Morning Rain on SG Highway",
    corridorName: "SG_Highway",
    hour: 8,
    isRain: true,
    isFestival: false,
  },
  {
    label: "Festival Evening at Stadium Motera",
    corridorName: "Stadium_Motera",
    hour: 19,
    isRain: false,
    isFestival: true,
  },
  {
    label: "Normal Afternoon on CG Road",
    corridorName: "CG_Road",
    hour: 14,
    isRain: false,
    isFestival: false,
  },
  {
    label: "Rush Hour on Ashram Road",
    corridorName: "Ashram_Road",
    hour: 18,
    isRain: false,
    isFestival: false,
  },
];

function formatFeatureName(feature: string) {
  return feature.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
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
  icon,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  icon: "rain" | "event";
}) {
  return (
    <label className={`scenario-toggle ${checked ? "is-on" : ""}`}>
      <span className="control-icon"><StatusIcon type={icon} /></span>
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-track" aria-hidden="true"><span /></span>
    </label>
  );
}

function Skeleton({ className = "" }: { className?: string }) {
  return <span className={`skeleton ${className}`} />;
}

function App() {
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
  const [modelInfoLoading, setModelInfoLoading] = useState(true);
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
    setModelInfoLoading(true);
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
    setModelInfoLoading(false);
  }, []);

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
    scenario?: Pick<DemoPreset, "hour" | "isRain" | "isFestival">,
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
      setReasoning({
        traceId: data.trace_id,
        corridorName: data.segment_id,
        congestionPct: data.prediction.congestion_pct,
        severity: data.prediction.label,
        explanation: data.explanation,
        factors: data.top_factors,
        counterfactual: data.counterfactual,
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

  function runPreset(preset: DemoPreset) {
    setActiveCorridor(preset.corridorName);
    setHour(preset.hour);
    setIsRain(preset.isRain);
    setIsFestival(preset.isFestival);
    void selectCorridor(preset.corridorName, preset);
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

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div>
            <p className="eyebrow">Ahmedabad traffic command / XAI-01</p>
            <h1>Ahmedabad XAI Traffic Intelligence</h1>
            <p className="subtitle">Predict traffic congestion and understand every prediction using SHAP-based reasoning traces.</p>
          </div>
        </div>
        <div className="system-status">
          <span className="status-dot" />
          <span>{apiOnline ? "Model online" : "Connecting"}</span>
          <small>{modelInfo ? `${modelInfo.model_type} · SHAP` : "API · 127.0.0.1:8000"}</small>
        </div>
      </header>

      <section className="hero-explainer panel" aria-labelledby="how-it-works-title">
        <div className="hero-flow">
          <div className="hero-flow-copy">
            <p className="section-kicker">How the system works</p>
            <h2 id="how-it-works-title">From traffic context to an explainable decision</h2>
          </div>
          <div className="flow-steps five-step-flow" aria-label="Explainable prediction workflow">
            {[
              "Select Scenario",
              "FastAPI Request",
              "XGBoost Prediction",
              "SHAP Explanation",
              "Saved Audit Trace",
            ].map((step, index) => (
              <div className="flow-step" key={step}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{step}</strong>
                {index < 4 && <i aria-hidden="true">→</i>}
              </div>
            ))}
          </div>
        </div>
        <div className="explanation-cards">
          <article>
            <span>01 / Input scenario</span>
            <strong>Describe the traffic context</strong>
            <p>Choose a corridor, hour, rain condition, and event activity before running the model.</p>
          </article>
          <article>
            <span>02 / Prediction</span>
            <strong>Estimate congestion severity</strong>
            <p>XGBoost returns a congestion percentage and classifies it from LOW through SEVERE.</p>
          </article>
          <article>
            <span>03 / Reasoning trace</span>
            <strong>Understand why it happened</strong>
            <p>SHAP shows which traffic features pushed the result upward or helped reduce congestion.</p>
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
          <p className="section-kicker">Step 01 / Configure input</p>
          <h2>Build a traffic scenario</h2>
          <p>Select a corridor and operating conditions, then run one focused prediction or refresh the full network.</p>
        </div>
        <div className="scenario-controls">
          <label className="corridor-control">
            <span>Corridor</span>
            <select value={activeCorridor} onChange={(event) => setActiveCorridor(event.target.value)}>
              {MAP_ROADS.map((road) => (
                <option key={road.id} value={road.id}>{DISPLAY_NAMES[road.id]}</option>
              ))}
            </select>
          </label>
          <label className="hour-control">
            <span className="hour-label">Hour</span>
            <input type="range" min="0" max="23" value={hour} onChange={(event) => setHour(Number(event.target.value))} />
            <output>{String(hour).padStart(2, "0")}:00</output>
          </label>
          <Toggle checked={isRain} onChange={setIsRain} label="Rain" icon="rain" />
          <Toggle checked={isFestival} onChange={setIsFestival} label="Festival / event" icon="event" />
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
          <span>Demo scenarios</span>
          <div>
            {DEMO_PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => runPreset(preset)}
                disabled={detailLoading}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="primary-grid">
        <article className="panel map-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Network overview</p>
              <h2>Ahmedabad corridor state</h2>
            </div>
            <div className="map-legend">
              {(["LOW", "MEDIUM", "HIGH", "SEVERE"] as Severity[]).map((severity) => (
                <span key={severity}><i className={`severity-fill ${severity.toLowerCase()}`} />{severity}</span>
              ))}
            </div>
          </div>

          <div className={`city-map ${dashboardLoading ? "is-loading" : ""}`}>
            <div className="map-meta">
              <span>AMD / 23.02°N</span>
              <span>72.57°E</span>
            </div>
            <svg viewBox="0 0 540 400" role="img" aria-label="Abstract map of Ahmedabad traffic corridors">
              <defs>
                <pattern id="minorGrid" width="20" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth=".5" />
                </pattern>
                <filter id="roadGlow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>
              <rect width="540" height="400" className="map-grid" fill="url(#minorGrid)" />
              <path className="river" d="M247 -10 C221 55 244 89 231 139 C215 198 238 238 218 290 C201 335 191 374 201 410" />
              <g className="minor-roads">
                <path d="M38 103 C143 130 199 104 260 132 C322 161 400 151 515 112" />
                <path d="M31 221 C111 199 159 168 221 170 C306 173 400 151 514 203" />
                <path d="M54 281 C141 253 206 253 276 278 C353 306 421 333 508 314" />
                <path d="M156 32 C153 107 133 182 143 245 C150 303 176 352 208 394" />
                <path d="M363 20 C333 86 350 134 380 187 C411 243 417 310 398 386" />
              </g>
              <g className="corridor-roads">
                {MAP_ROADS.map((road) => {
                  const corridor = corridorLookup.get(road.id);
                  const severity = corridor?.severity ?? "LOW";
                  const isActive = activeCorridor === road.id;
                  return (
                    <g
                      key={road.id}
                      className={`map-corridor ${severity.toLowerCase()} ${isActive ? "active" : ""}`}
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
                      <path className="road-status" d={road.path} filter={isActive ? "url(#roadGlow)" : undefined} />
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
              <p className="section-kicker">Active reasoning trace</p>
              <h2>{reasoning ? DISPLAY_NAMES[reasoning.corridorName] ?? reasoning.corridorName : "Select a corridor"}</h2>
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
                  <span>Model confidence</span>
                  <strong>{reasoning.confidence != null ? `${Math.round(reasoning.confidence * 100)}%` : "Calibrated"}</strong>
                  <div><i style={{ width: reasoning.confidence != null ? `${reasoning.confidence * 100}%` : "85%" }} /></div>
                </div>
              </div>

              <div className="explanation-block">
                <span className="block-label">Model interpretation</span>
                <p>{reasoning.explanation}</p>
              </div>

              <div className="shap-section">
                <div className="block-title-row">
                  <span className="block-label">SHAP contribution profile</span>
                  <small>← reduces / increases →</small>
                </div>
                <p className="explanation-note">
                  Positive SHAP values push congestion higher. Negative SHAP values reduce predicted congestion.
                </p>
                <div className="shap-axis"><i /><span /></div>
                <div className="shap-list">
                  {reasoning.factors.map((factor, index) => {
                    const width = Math.max(5, (Math.abs(factor.shap) / maxShap) * 47);
                    const positive = factor.shap >= 0;
                    return (
                      <div className="shap-row" key={`${factor.feature}-${index}`}>
                        <div className="factor-meta">
                          <strong>{formatFeatureName(factor.feature)}</strong>
                          <span>Value {String(factor.value)}</span>
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
                <span className="block-label">Counterfactual guidance</span>
                <small>Counterfactual explains what would need to change to reduce congestion.</small>
                <p>{reasoning.counterfactual}</p>
              </div>

              <footer className="trace-footer">
                <span>Trace <code>{reasoning.traceId.slice(0, 8)}</code></span>
                <span>{formatTime(reasoning.predictedAt)}</span>
              </footer>
            </>
          ) : (
            <div className="empty-state">Select any road on the network map to generate its reasoning trace.</div>
          )}
        </article>
      </section>

      <section className="corridor-section">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Live model output</p>
            <h2>All monitored corridors</h2>
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
                  <div className="mini-factors">
                    {corridor.top_factors.slice(0, 2).map((factor) => (
                      <span key={factor.feature}>
                        <i className={factor.shap >= 0 ? "up" : "down"} />
                        {formatFeatureName(factor.feature)}
                        <b>{factor.shap >= 0 ? "+" : ""}{factor.shap.toFixed(2)}</b>
                      </span>
                    ))}
                  </div>
                </button>
              ))}
        </div>
      </section>

      <section className="operations-grid">
        <article className="panel feed-panel anomaly-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Operational awareness</p>
              <h2>Anomaly alert feed</h2>
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
              <p className="section-kicker">Reasoning archive</p>
              <h2>Recent trace history</h2>
            </div>
            <span className="panel-icon"><StatusIcon type="history" /></span>
          </div>
          <div className="feed-list">
            {feedsLoading
              ? Array.from({ length: 4 }, (_, index) => <Skeleton className="feed-skeleton" key={index} />)
              : traces.length === 0
                ? <div className="feed-empty"><strong>No reasoning traces yet</strong><span>Run a corridor prediction to create the first trace.</span></div>
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
        <div className="model-info-heading">
          <div>
            <p className="section-kicker">Model transparency</p>
            <h2>What powers this prototype</h2>
            <p>The model metadata below is loaded directly from the backend.</p>
          </div>
          <span className="api-source">GET /api/v1/model-info</span>
        </div>
        {modelInfoLoading ? (
          <div className="model-info-loading">
            <Skeleton className="large" />
            <Skeleton className="large" />
            <Skeleton className="large" />
          </div>
        ) : modelInfo ? (
          <div className="model-info-grid">
            <article><span>Prediction model</span><strong>{modelInfo.model_type}</strong></article>
            <article><span>Explainability method</span><strong>{modelInfo.xai_method}</strong></article>
            <article><span>Prediction target</span><strong>{formatFeatureName(modelInfo.target)}</strong></article>
            <article className="feature-inventory">
              <span>Features used · {modelInfo.features.length}</span>
              <div>{modelInfo.features.map((feature) => <em key={feature}>{formatFeatureName(feature)}</em>)}</div>
            </article>
            <p className="prototype-note">{modelInfo.prototype_note}</p>
          </div>
        ) : (
          <div className="feed-empty">
            <strong>Model information unavailable</strong>
            <span>Confirm the backend exposes /api/v1/model-info.</span>
          </div>
        )}
      </section>

      <footer className="app-footer">
        <span>Ahmedabad XAI Traffic Intelligence</span>
        <span>Model xgboost_v1 · Reasoning layer SHAP · API {API_BASE.replace("http://", "")}</span>
      </footer>
    </main>
  );
}

export default App;
