import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "xgboost_traffic_model.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "model_features.pkl"
DATA_PATH = ROOT_DIR / "data" / "processed" / "ahmedabad_training_data.csv"
TRACE_DIR = ROOT_DIR / "outputs" / "traces"

TRACE_DIR.mkdir(parents=True, exist_ok=True)

for required_path, label in (
    (MODEL_PATH, "Model"),
    (FEATURES_PATH, "Features file"),
    (DATA_PATH, "Dataset"),
):
    if not required_path.exists():
        raise FileNotFoundError(f"{label} not found: {required_path}")


# -----------------------------
# Application and CORS
# -----------------------------
app = FastAPI(
    title="XAI Traffic Prediction API",
    description="Traffic congestion prediction system with SHAP reasoning traces",
    version="1.1.0",
)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cors_safety_headers(request: Request, call_next):
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


@app.options("/api/v1/predict")
def predict_options():
    return Response(status_code=200)


# -----------------------------
# Model and data
# -----------------------------
print("Loading model, features and dataset...")

model = joblib.load(MODEL_PATH)
features = list(joblib.load(FEATURES_PATH))
df = pd.read_csv(DATA_PATH)

for column in df.columns:
    if df[column].dtype == "bool":
        df[column] = df[column].astype(int)

explainer = shap.TreeExplainer(model)
corridor_names = sorted(df["corridor_name"].unique().tolist())

print("Model loaded successfully")
print("Available corridors:", corridor_names)


# -----------------------------
# Schemas and constants
# -----------------------------
class PredictionRequest(BaseModel):
    corridor_name: str = "SG_Highway"
    hour: int = Field(default=8, ge=0, le=23)
    is_rain: bool = False
    is_festival: bool = False


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


# -----------------------------
# Helpers
# -----------------------------
def get_display_name(corridor_name: str) -> str:
    return DISPLAY_NAMES.get(corridor_name, corridor_name.replace("_", " "))


def get_severity(congestion_pct: float) -> str:
    if congestion_pct < 25:
        return "LOW"
    if congestion_pct < 50:
        return "MEDIUM"
    if congestion_pct < 75:
        return "HIGH"
    return "SEVERE"


def make_json_safe(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Series):
        return [make_json_safe(item) for item in value.tolist()]
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


def build_trace(sample_row: pd.DataFrame) -> dict:
    model_input = sample_row[features]
    prediction = float(np.clip(model.predict(model_input)[0], 0.0, 100.0))
    severity = get_severity(prediction)

    shap_explanation = explainer(model_input)
    shap_values = shap_explanation.values[0]
    base_value = np.asarray(shap_explanation.base_values).reshape(-1)[0]

    contributions = []
    for feature_name, raw_value, shap_value in zip(
        features,
        model_input.iloc[0].values,
        shap_values,
    ):
        contributions.append({
            "feature": feature_name,
            "value": make_json_safe(raw_value),
            "shap": round(float(shap_value), 4),
            "impact": (
                "increases congestion"
                if shap_value > 0
                else "decreases congestion"
            ),
        })

    contributions_sorted = sorted(
        contributions,
        key=lambda item: abs(item["shap"]),
        reverse=True,
    )
    top_contributors = contributions_sorted[:5]
    positive_factors = [
        factor for factor in top_contributors if factor["shap"] > 0
    ]

    if positive_factors:
        reason_text = ", ".join(
            f"{factor['feature']}={factor['value']}"
            for factor in positive_factors[:3]
        )
        explanation = (
            f"{severity} congestion predicted mainly because of {reason_text}."
        )
    else:
        explanation = (
            f"{severity} congestion predicted. "
            "No strong positive congestion factors found."
        )

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
        "confidence": (
            round(float(sample_row["confidence"].iloc[0]), 2)
            if "confidence" in sample_row.columns
            else None
        ),
        "counterfactual": (
            "Congestion could reduce if current speed increases, "
            "rush-hour pressure reduces, or rain/festival effect is absent."
        ),
        "input_snapshot": {
            column: make_json_safe(sample_row[column].iloc[0])
            for column in sample_row.columns
            if column != "severity"
        },
    }

    trace_path = TRACE_DIR / f"{trace_id}.json"
    with trace_path.open("w", encoding="utf-8") as trace_file:
        json.dump(trace, trace_file, indent=2, ensure_ascii=False)

    return trace


def get_dashboard_summary(corridors: list[dict]) -> dict:
    severity_counts = {
        severity: sum(
            item["severity"] == severity for item in corridors
        )
        for severity in ("LOW", "MEDIUM", "HIGH", "SEVERE")
    }

    if corridors:
        average_congestion = round(
            sum(item["congestion_pct"] for item in corridors) / len(corridors),
            2,
        )
        highest_corridor = max(
            corridors,
            key=lambda item: item["congestion_pct"],
        )["corridor_name"]
    else:
        average_congestion = 0.0
        highest_corridor = None

    return {
        "total_corridors": len(corridors),
        "low": severity_counts["LOW"],
        "medium": severity_counts["MEDIUM"],
        "high": severity_counts["HIGH"],
        "severe": severity_counts["SEVERE"],
        "average_congestion": average_congestion,
        "highest_congestion_corridor": highest_corridor,
    }


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
        "corridor_count": len(corridor_names),
        "message": "XAI Traffic Prediction API is working",
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

    corridor_rows = df[df["corridor_name"] == request.corridor_name]
    if corridor_rows.empty:
        raise HTTPException(status_code=404, detail="No data found for corridor")

    sample_row = apply_scenario(
        corridor_rows.sample(1).copy(),
        request.hour,
        request.is_rain,
        request.is_festival,
    )
    trace = build_trace(sample_row)

    # Keep the original full trace response and add aliases used by the UI.
    return {
        **trace,
        "explanation": trace["lime_explanation"],
        "top_factors": trace["shap_values"],
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
        source_rows = (
            matching_hour_rows
            if not matching_hour_rows.empty
            else corridor_rows
        )
        if source_rows.empty:
            continue

        sample_row = apply_scenario(
            source_rows.tail(1).copy(),
            hour,
            is_rain,
            is_festival,
        )
        trace = build_trace(sample_row)

        corridors.append({
            "corridor_name": corridor_name,
            "display_name": get_display_name(corridor_name),
            "congestion_pct": trace["prediction"]["congestion_pct"],
            "severity": trace["prediction"]["label"],
            "confidence": trace["confidence"],
            "explanation": trace["lime_explanation"],
            "top_factors": trace["shap_values"],
            "trace_id": trace["trace_id"],
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": {
            "hour": hour,
            "is_rain": is_rain,
            "is_festival": is_festival,
        },
        "summary": get_dashboard_summary(corridors),
        # Retained for compatibility with the existing frontend.
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
            with trace_path.open("r", encoding="utf-8") as trace_file:
                trace = json.load(trace_file)
        except (OSError, json.JSONDecodeError):
            continue

        segment_id = trace.get("segment_id")
        prediction = trace.get("prediction", {})
        traces.append({
            "trace_id": trace.get("trace_id", trace_path.stem),
            "segment_id": segment_id,
            "display_name": (
                get_display_name(segment_id) if segment_id else None
            ),
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

    try:
        with trace_path.open("r", encoding="utf-8") as trace_file:
            return json.load(trace_file)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail="Trace file could not be read",
        ) from error


@app.get("/api/v1/anomalies")
def get_anomalies():
    congestion = pd.to_numeric(df["congestion_pct"], errors="coerce")
    preferred_rows = df[
        (df["severity"].isin(["HIGH", "SEVERE"])) | (congestion > 75)
    ].copy()
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
        congestion_pct = round(float(row["congestion_pct"]), 2)
        severity = get_severity(congestion_pct)

        anomalies.append({
            "id": str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"traffic-anomaly:{corridor_name}:{timestamp}",
                )
            ),
            "corridor_name": corridor_name,
            "display_name": get_display_name(corridor_name),
            "timestamp": timestamp,
            "congestion_pct": congestion_pct,
            "severity": severity,
            "confidence": round(float(row.get("confidence", 0.85)), 2),
            "reason": (
                f"{severity} congestion detected due to speed drop, "
                "rush-hour effect, and contextual traffic pressure."
            ),
        })

    return {
        "count": len(anomalies),
        "anomalies": anomalies,
    }


@app.get("/api/v1/model-info")
def get_model_info():
    return {
        "model_type": "XGBoost Regressor",
        "xai_method": "SHAP TreeExplainer",
        "target": "congestion_pct",
        "features": features,
        "corridors": corridor_names,
        "prototype_note": (
            "Prototype using processed Ahmedabad-style traffic features."
        ),
    }
