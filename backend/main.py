import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import joblib
import shap
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if not (ROOT_DIR / "models").exists():
    # backend/main.py is one level shallower than backend/app/main.py.
    ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "xgboost_traffic_model.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "model_features.pkl"
DATA_PATH = ROOT_DIR / "data" / "processed" / "ahmedabad_training_data.csv"
TRACE_DIR = ROOT_DIR / "outputs" / "traces"

TRACE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="XAI Traffic Prediction API",
    description="Traffic congestion prediction system with SHAP reasoning traces",
    version="1.0.0",
)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

# Normal CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Extra manual CORS middleware to force headers on every response
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        response = Response(status_code=200)
    else:
        response = await call_next(request)

    origin = request.headers.get("origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"

    return response


print("MAIN.PY LOADED WITH CORS FIX")
print("ROOT_DIR:", ROOT_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("FEATURES_PATH:", FEATURES_PATH)
print("DATA_PATH:", DATA_PATH)


# -----------------------------
# File checks
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")


# -----------------------------
# Load model and data
# -----------------------------
print("Loading model, features and dataset...")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
df = pd.read_csv(DATA_PATH)

for col in df.columns:
    if df[col].dtype == "bool":
        df[col] = df[col].astype(int)

explainer = shap.TreeExplainer(model)

corridor_names = sorted(df["corridor_name"].unique().tolist())

print("Model loaded successfully")
print("Available corridors:", corridor_names)


# -----------------------------
# Request model
# -----------------------------
class PredictionRequest(BaseModel):
    corridor_name: str = "SG_Highway"
    hour: int = Field(default=8, ge=0, le=23)
    is_rain: bool = False
    is_festival: bool = False


# -----------------------------
# Helper functions
# -----------------------------
def get_severity(pct: float) -> str:
    if pct < 25:
        return "LOW"
    if pct < 50:
        return "MEDIUM"
    if pct < 75:
        return "HIGH"
    return "SEVERE"


DISPLAY_NAMES = {
    "SG_Highway": "SG Highway",
    "Ring_Road_132ft": "132 ft Ring Road",
    "CG_Road": "CG Road",
    "Ashram_Road": "Ashram Road",
    "Sardar_Patel_Ring": "Sardar Patel Ring Road",
    "Narol_Naroda": "Narol-Naroda",
    "Maninagar": "Maninagar",
    "Stadium_Motera": "Stadium Motera",
}


def get_display_name(corridor_name: str) -> str:
    return DISPLAY_NAMES.get(corridor_name, corridor_name.replace("_", " "))


def make_json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def apply_scenario(
    sample_row: pd.DataFrame,
    hour: int,
    is_rain: bool,
    is_festival: bool,
) -> pd.DataFrame:
    sample_row = sample_row.copy()
    sample_row["hour"] = hour
    sample_row["is_rain"] = int(is_rain)
    sample_row["is_festival"] = int(is_festival)

    derived_values = {
        "is_morning_rush": int(8 <= hour <= 10),
        "is_evening_rush": int(17 <= hour <= 20),
        "is_school_hour": int(7 <= hour <= 14),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
    }
    for column, value in derived_values.items():
        if column in sample_row.columns:
            sample_row[column] = value

    return sample_row


def build_trace(sample_row: pd.DataFrame):
    X_sample = sample_row[features]

    prediction = float(np.clip(model.predict(X_sample)[0], 0.0, 100.0))
    severity = get_severity(prediction)

    shap_values = explainer.shap_values(X_sample)
    base_value = explainer.expected_value

    if isinstance(base_value, list):
        base_value = base_value[0]

    shap_values_for_row = shap_values[0]

    contributions = []

    for feature, feature_value, shap_value in zip(
        features,
        X_sample.iloc[0].values,
        shap_values_for_row,
    ):
        contributions.append({
            "feature": feature,
            "value": make_json_safe(feature_value),
            "shap": round(float(shap_value), 4),
            "impact": "increases congestion" if shap_value > 0 else "decreases congestion",
        })

    contributions_sorted = sorted(
        contributions,
        key=lambda x: abs(x["shap"]),
        reverse=True,
    )

    top_contributors = contributions_sorted[:5]

    positive_factors = [c for c in top_contributors if c["shap"] > 0]

    if positive_factors:
        reason_text = ", ".join(
            [f"{c['feature']}={c['value']}" for c in positive_factors[:3]]
        )
        explanation = f"{severity} congestion predicted mainly because of {reason_text}."
    else:
        explanation = f"{severity} congestion predicted. No strong positive congestion factors found."

    trace_id = str(uuid.uuid4())

    trace = {
        "trace_id": trace_id,
        "segment_id": str(sample_row["corridor_name"].iloc[0]),
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "prediction": {
            "congestion_pct": round(prediction, 2),
            "label": severity,
        },
        "model_version": "xgboost_v1",
        "base_value": round(float(base_value), 4),
        "shap_values": top_contributors,
        "all_shap_values": contributions_sorted,
        "lime_explanation": explanation,
        "confidence": round(float(sample_row["confidence"].iloc[0]), 2),
        "counterfactual": (
            "Congestion could reduce if current speed increases, "
            "rush-hour pressure reduces, or rain/festival effect is absent."
        ),
        "input_snapshot": {
            col: make_json_safe(sample_row[col].iloc[0])
            for col in sample_row.columns
            if col != "severity"
        },
    }

    trace_path = TRACE_DIR / f"{trace_id}.json"

    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    return trace


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "XAI Traffic Prediction API is running",
        "status": "ok",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "backend": "running",
        "model_loaded": True,
        "message": "XAI Traffic Prediction API is working"
    }

@app.get("/api/v1/segments")
def get_segments():
    return {
        "count": len(corridor_names),
        "segments": corridor_names,
    }


@app.post("/api/v1/predict")
def predict(request: PredictionRequest):
    if request.corridor_name not in corridor_names:
        raise HTTPException(
            status_code=404,
            detail=f"Corridor not found. Available corridors: {corridor_names}",
        )

    sample = df[df["corridor_name"] == request.corridor_name].copy()

    if sample.empty:
        raise HTTPException(status_code=404, detail="No data found for corridor")

    sample_row = sample.sample(1, random_state=None).copy()

    sample_row = apply_scenario(
        sample_row,
        request.hour,
        request.is_rain,
        request.is_festival,
    )

    trace = build_trace(sample_row)

    return {
        "prediction": trace["prediction"],
        "trace_id": trace["trace_id"],
        "segment_id": trace["segment_id"],
        "explanation": trace["lime_explanation"],
        "top_factors": trace["shap_values"],
        "counterfactual": trace["counterfactual"],
    }


@app.get("/api/v1/dashboard")
def get_dashboard(
    hour: int = Query(default=8, ge=0, le=23),
    is_rain: bool = False,
    is_festival: bool = False,
):
    corridors = []

    for corridor_name in corridor_names:
        corridor_rows = df[df["corridor_name"] == corridor_name]
        matching_hour_rows = corridor_rows[corridor_rows["hour"] == hour]
        source_rows = matching_hour_rows if not matching_hour_rows.empty else corridor_rows

        if source_rows.empty:
            continue

        sample_row = source_rows.tail(1).copy()
        sample_row = apply_scenario(sample_row, hour, is_rain, is_festival)
        trace = build_trace(sample_row)

        corridors.append({
            "corridor_name": corridor_name,
            "display_name": get_display_name(corridor_name),
            "congestion_pct": trace["prediction"]["congestion_pct"],
            "severity": trace["prediction"]["label"],
            "explanation": trace["lime_explanation"],
            "top_factors": trace["shap_values"],
            "trace_id": trace["trace_id"],
            "confidence": trace["confidence"],
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(corridors),
        "corridors": corridors,
    }


@app.get("/api/v1/traces")
def list_traces(limit: int = Query(default=20, ge=1, le=100)):
    trace_files = sorted(
        TRACE_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    traces = []
    for trace_path in trace_files:
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        prediction = trace.get("prediction", {})
        traces.append({
            "trace_id": trace.get("trace_id", trace_path.stem),
            "segment_id": trace.get("segment_id"),
            "predicted_at": trace.get("predicted_at"),
            "congestion_pct": prediction.get("congestion_pct"),
            "label": prediction.get("label"),
            "explanation": trace.get("lime_explanation"),
        })

        if len(traces) >= limit:
            break

    return {
        "count": len(traces),
        "traces": traces,
    }


@app.get("/api/v1/traces/{trace_id}")
def get_trace(trace_id: str):
    trace_path = TRACE_DIR / f"{trace_id}.json"

    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="Trace not found")

    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    return trace


@app.get("/api/v1/anomalies")
def get_anomalies():
    congestion = pd.to_numeric(df["congestion_pct"], errors="coerce")
    preferred_rows = df[(df["severity"] == "SEVERE") | (congestion > 75)].copy()
    preferred_rows["_timestamp"] = pd.to_datetime(
        preferred_rows["timestamp"],
        errors="coerce",
        utc=True,
    )
    preferred_rows["_congestion"] = pd.to_numeric(
        preferred_rows["congestion_pct"],
        errors="coerce",
    )
    preferred_rows = preferred_rows.sort_values(
        by=["_timestamp", "_congestion"],
        ascending=[False, False],
        na_position="last",
    ).head(10)

    anomalies = []

    for _, row in preferred_rows.iterrows():
        corridor_name = str(row["corridor_name"])
        timestamp = str(row["timestamp"])
        anomaly_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"traffic-anomaly:{corridor_name}:{timestamp}",
            )
        )
        anomalies.append({
            "id": anomaly_id,
            "corridor_name": corridor_name,
            "display_name": get_display_name(corridor_name),
            "timestamp": timestamp,
            "congestion_pct": round(float(row["congestion_pct"]), 2),
            "severity": "SEVERE",
            "confidence": round(float(row.get("confidence", 0.85)), 2),
            "reason": (
                "SEVERE congestion detected due to speed drop, rush-hour effect, "
                "and contextual traffic pressure."
            ),
        })

    return {
        "count": len(anomalies),
        "anomalies": anomalies,
    }
