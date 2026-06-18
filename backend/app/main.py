import os
import json
import uuid
from datetime import datetime, timezone

import joblib
import shap
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request

app = FastAPI(
    title="XAI Traffic Prediction API",
    description="Traffic congestion prediction system with SHAP reasoning traces",
    version="1.0.0"
)


@app.middleware("http")
async def log_options_request(request: Request, call_next):
    if request.method == "OPTIONS":
        print("---- OPTIONS REQUEST START ----")
        for k, v in request.headers.items():
            print(f"{k}: {v}")
        print("---- OPTIONS REQUEST END ----")
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/api/v1/predict")
def predict_options():
    return Response(status_code=200)

MODEL_PATH = "models/xgboost_traffic_model.pkl"
FEATURES_PATH = "models/model_features.pkl"
DATA_PATH = "data/processed/ahmedabad_training_data.csv"
TRACE_DIR = "outputs/traces"

os.makedirs(TRACE_DIR, exist_ok=True)

print("Loading model, features and dataset...")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
df = pd.read_csv(DATA_PATH)

for col in df.columns:
    if df[col].dtype == "bool":
        df[col] = df[col].astype(int)

explainer = shap.TreeExplainer(model)

corridor_names = sorted(df["corridor_name"].unique().tolist())

class PredictionRequest(BaseModel):
    corridor_name: str = "SG_Highway"
    hour: int = 8
    is_rain: bool = False
    is_festival: bool = False


def get_severity(pct: float) -> str:
    if pct < 25:
        return "LOW"
    if pct < 50:
        return "MEDIUM"
    if pct < 75:
        return "HIGH"
    return "SEVERE"


def make_json_safe(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Series):
        return value.tolist()
    return value


def create_trace(sample_row: pd.DataFrame, prediction: float, shap_values: np.ndarray, base_value: float):
    severity = get_severity(prediction)

    contributions = []
    for feature_name, shap_value, raw_value in zip(
        features,
        shap_values,
        sample_row[features].iloc[0].tolist()
    ):
        impact = "increases congestion" if shap_value > 0 else "decreases congestion"
        contributions.append({
            "feature": feature_name,
            "value": make_json_safe(raw_value),
            "shap": float(round(shap_value, 4)),
            "impact": impact,
        })

    contributions_sorted = sorted(
        contributions,
        key=lambda x: abs(x["shap"]),
        reverse=True
    )

    top_contributors = contributions_sorted[:5]
    positive_factors = [c for c in top_contributors if c["shap"] > 0]

    if positive_factors:
        reason_text = ", ".join([f"{c['feature']}={c['value']}" for c in positive_factors[:3]])
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
            "label": severity
        },
        "model_version": "xgboost_v1",
        "base_value": round(float(base_value), 4),
        "shap_values": top_contributors,
        "all_shap_values": contributions_sorted,
        "lime_explanation": explanation,
        "confidence": round(float(sample_row["confidence"].iloc[0]), 2) if "confidence" in sample_row.columns else None,
        "counterfactual": (
            "Congestion could reduce if current speed increases, "
            "rush-hour pressure reduces, or rain/festival effect is absent."
        ),
        "input_snapshot": {
            col: make_json_safe(sample_row[col].iloc[0])
            for col in sample_row.columns
            if col != "severity"
        }
    }

    trace_path = os.path.join(TRACE_DIR, f"{trace_id}.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    return trace


@app.get("/")
def root():
    return {
        "message": "XAI Traffic Prediction API is running",
        "status": "ok"
    }


@app.get("/api/v1/segments")
def get_segments():
    return {
        "count": len(corridor_names),
        "segments": corridor_names
    }


@app.post("/api/v1/predict")
def predict(request: PredictionRequest):
    if request.corridor_name not in corridor_names:
        raise HTTPException(
            status_code=404,
            detail=f"Corridor not found. Available corridors: {corridor_names}"
        )

    sample = df[df["corridor_name"] == request.corridor_name].copy()
    if sample.empty:
        raise HTTPException(status_code=404, detail="No data found for corridor")

    sample_row = sample.sample(1, random_state=None).copy()
    sample_row["hour"] = request.hour
    sample_row["is_rain"] = int(request.is_rain)
    sample_row["is_festival"] = int(request.is_festival)

    sample_row["is_morning_rush"] = int(8 <= request.hour <= 10)
    sample_row["is_evening_rush"] = int(17 <= request.hour <= 20)
    sample_row["is_school_hour"] = int(7 <= request.hour <= 14)

    X_row = sample_row[features]
    prediction = float(model.predict(X_row)[0])

    shap_explanation = explainer(X_row)
    shap_values = shap_explanation.values[0]
    base_value = float(shap_explanation.base_values[0])

    trace = create_trace(sample_row, prediction, shap_values, base_value)
    return trace
