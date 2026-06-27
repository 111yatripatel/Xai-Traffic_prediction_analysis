import hashlib
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
    simulate_live_data: bool = False


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

FEATURE_LABELS = {
    "current_speed": "Current Speed",
    "freeflow_speed": "Free-flow Speed",
    "is_rain": "Rain Condition",
    "is_festival": "Festival/Event",
    "hour_cos": "Time of Day Pattern",
    "hour_sin": "Time of Day Pattern",
    "corridor_id": "Corridor Pattern",
    "weekday": "Day of Week",
    "is_morning_rush": "Morning Rush",
    "is_evening_rush": "Evening Rush",
    "is_school_hour": "School Hour",
    "hour": "Hour of Day",
}

DEFAULT_WEEKDAY = 2


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


def get_feature_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(
        feature_name,
        feature_name.replace("_", " ").title(),
    )


def join_reasons(reasons: list[str]) -> str:
    if len(reasons) == 1:
        return reasons[0]
    if len(reasons) == 2:
        return f"{reasons[0]} and {reasons[1]}"
    return f"{', '.join(reasons[:-1])}, and {reasons[-1]}"


def get_time_features(hour: int, weekday: int = DEFAULT_WEEKDAY) -> dict:
    is_weekend = int(weekday >= 5)
    return {
        "hour": hour,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "is_morning_rush": int(8 <= hour <= 10 and not is_weekend),
        "is_evening_rush": int(17 <= hour <= 20 and not is_weekend),
        "is_school_hour": int(7 <= hour <= 14 and not is_weekend),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
    }


def build_features(
    corridor_name: str,
    hour: int,
    is_rain: bool,
    is_festival: bool,
    simulate_live_data: bool = False,
) -> pd.DataFrame:
    corridor_rows = df[df["corridor_name"] == corridor_name]
    if corridor_rows.empty:
        raise HTTPException(status_code=404, detail="No data found for corridor")

    if simulate_live_data:
        sample_row = corridor_rows.sample(1).copy()
        return apply_scenario(sample_row, hour, is_rain, is_festival)

    seed_source = f"{corridor_name}|{hour}"
    seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    sample_row = corridor_rows.sample(1, random_state=seed).copy()
    return apply_scenario(sample_row, hour, is_rain, is_festival)


def build_human_explanation(
    severity: str,
    sample_row: pd.DataFrame,
    top_contributors: list[dict],
) -> str:
    negative_features = {
        factor["feature"]
        for factor in top_contributors
        if factor["shap"] < 0
    }
    positive_features = {
        factor["feature"]
        for factor in top_contributors
        if factor["shap"] > 0
    }
    reasons = []

    if severity == "LOW" and negative_features:
        if negative_features.intersection({"current_speed", "freeflow_speed"}):
            reasons.append("the observed speed profile reduces congestion")

        reducing_reasons = {
            "is_rain": "the absence of rain reduces traffic pressure",
            "is_festival": "no festival or event pressure is present",
            "corridor_id": "this corridor pattern lowers predicted congestion",
            "is_morning_rush": "morning rush-hour pressure is limited",
            "is_evening_rush": "evening rush-hour pressure is limited",
            "is_school_hour": "school-hour pressure is limited",
            "hour_sin": "the time-of-day pattern reduces traffic pressure",
            "hour_cos": "the time-of-day pattern reduces traffic pressure",
            "weekday": "the day-of-week pattern reduces traffic pressure",
            "hour": "the selected hour reduces traffic pressure",
        }
        for factor in top_contributors:
            feature_name = factor["feature"]
            if factor["shap"] >= 0 or feature_name in {
                "current_speed",
                "freeflow_speed",
            }:
                continue
            reason = reducing_reasons.get(
                feature_name,
                f"{get_feature_label(feature_name).lower()} reduces predicted congestion",
            )
            if reason not in reasons:
                reasons.append(reason)
            if len(reasons) == 3:
                break

        return (
            f"Low congestion is predicted because "
            f"{join_reasons(reasons)}."
        )

    speed_is_influential = bool(
        positive_features.intersection({"current_speed", "freeflow_speed"})
    )
    if speed_is_influential:
        current_speed = float(sample_row["current_speed"].iloc[0])
        freeflow_speed = float(sample_row["freeflow_speed"].iloc[0])
        if current_speed < freeflow_speed * 0.75:
            reasons.append(
                "current speed is much lower than the free-flow speed"
            )
        elif current_speed < freeflow_speed:
            reasons.append("current speed is below the free-flow speed")
        else:
            reasons.append("the current speed pattern is influencing traffic")

    factor_reasons = {
        "is_rain": "rain is active",
        "is_festival": "a festival or event is active",
        "corridor_id": (
            "this corridor pattern contributes to higher congestion"
        ),
        "is_morning_rush": "morning rush-hour pressure is active",
        "is_evening_rush": "evening rush-hour pressure is active",
        "is_school_hour": "school-hour traffic is active",
        "hour_sin": "the time-of-day pattern increases traffic pressure",
        "hour_cos": "the time-of-day pattern increases traffic pressure",
        "weekday": "the day-of-week pattern increases traffic pressure",
        "hour": "the selected hour increases traffic pressure",
    }
    for factor in top_contributors:
        feature_name = factor["feature"]
        if factor["shap"] <= 0 or feature_name in {
            "current_speed",
            "freeflow_speed",
        }:
            continue
        reason = factor_reasons.get(
            feature_name,
            f"{get_feature_label(feature_name).lower()} contributes to higher congestion",
        )
        if reason not in reasons:
            reasons.append(reason)
        if len(reasons) == 3:
            break

    severity_text = severity.title()
    if reasons:
        return (
            f"{severity_text} congestion is predicted because "
            f"{join_reasons(reasons)}."
        )

    reducing_factors = [
        get_feature_label(factor["feature"]).lower()
        for factor in top_contributors
        if factor["shap"] < 0
    ][:3]
    if reducing_factors:
        return (
            f"{severity_text} congestion is predicted. "
            f"The strongest reducing influences are "
            f"{join_reasons(reducing_factors)}."
        )
    return (
        f"{severity_text} congestion is predicted, with no single feature "
        "dominating this result."
    )


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

    derived_values = get_time_features(hour)
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
            "display_name": get_feature_label(feature_name),
            "value": make_json_safe(raw_value),
            "shap": round(float(shap_value), 4),
            "impact": (
                "Increased predicted congestion"
                if shap_value > 0
                else "Reduced predicted congestion"
            ),
        })

    contributions_sorted = sorted(
        contributions,
        key=lambda item: abs(item["shap"]),
        reverse=True,
    )
    top_contributors = contributions_sorted[:5]
    explanation = build_human_explanation(
        severity,
        sample_row,
        top_contributors,
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
        "feature_snapshot": {
            column: make_json_safe(model_input[column].iloc[0])
            for column in model_input.columns
        },
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

    sample_row = build_features(
        request.corridor_name,
        request.hour,
        request.is_rain,
        request.is_festival,
        request.simulate_live_data,
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
    simulate_live_data: bool = False,
):
    corridors = []

    for corridor_name in corridor_names:
        sample_row = build_features(
            corridor_name,
            hour,
            is_rain,
            is_festival,
            simulate_live_data,
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
            "feature_snapshot": trace["feature_snapshot"],
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": {
            "hour": hour,
            "is_rain": is_rain,
            "is_festival": is_festival,
            "simulate_live_data": simulate_live_data,
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
