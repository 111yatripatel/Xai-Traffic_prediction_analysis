import os
import json
import uuid
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_severity(pct):
    if pct < 25:
        return "LOW"
    if pct < 50:
        return "MEDIUM"
    if pct < 75:
        return "HIGH"
    return "SEVERE"


def make_json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    print("Loading model and data...")

    model_path = "models/xgboost_traffic_model.pkl"
    features_path = "models/model_features.pkl"
    dataset_path = "data/processed/ahmedabad_training_data.csv"
    output_dir = "outputs"

    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    df = pd.read_csv(dataset_path)

    if "severity" in df.columns:
        df = df.drop(columns=["severity"])

    print("Dataset loaded, rows:", len(df))
    print("Using features:", features)

    sample_row = df.head(1)
    X_sample = sample_row[features]

    print("Generating prediction and SHAP explanation for sample row...")

    prediction = float(model.predict(X_sample)[0])
    severity = get_severity(prediction)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    base_value = float(shap_values.base_values[0])
    shap_values_for_row = shap_values.values[0]

    contributions = []
    for feature_name, shap_value, raw_value in zip(
        features,
        shap_values_for_row,
        X_sample.iloc[0].tolist()
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
        reason_text = ", ".join([
            f"{c['feature']}={c['value']}" for c in positive_factors[:3]
        ])
        lime_style_summary = (
            f"{severity} congestion predicted mainly because of {reason_text}."
        )
    else:
        lime_style_summary = (
            f"{severity} congestion predicted. No strong positive congestion factors found."
        )

    counterfactual = (
        "Congestion could reduce if current speed increases, rush-hour/event effects reduce, "
        "or rain/festival-related pressure is absent."
    )

    trace = {
        "trace_id": str(uuid.uuid4()),
        "segment_id": str(sample_row["corridor_name"].iloc[0]) if "corridor_name" in sample_row.columns else "unknown",
        "predicted_at": datetime.utcnow().isoformat() + "Z",
        "prediction": {
            "congestion_pct": round(prediction, 2),
            "label": severity,
        },
        "model_version": "xgboost_v1",
        "base_value": round(base_value, 4),
        "shap_values": top_contributors,
        "all_shap_values": contributions_sorted,
        "lime_explanation": lime_style_summary,
        "confidence": round(float(sample_row["confidence"].iloc[0]), 2) if "confidence" in sample_row.columns else None,
        "counterfactual": counterfactual,
        "input_snapshot": {
            col: make_json_safe(sample_row[col].iloc[0])
            for col in sample_row.columns
            if col != "severity"
        },
    }

    trace_path = os.path.join(output_dir, "reasoning_trace_sample.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    print(f"Saved reasoning trace JSON to {trace_path}")

    print("Saving SHAP waterfall chart...")
    explanation = shap.Explanation(
        values=shap_values_for_row,
        base_values=base_value,
        data=X_sample.iloc[0],
        feature_names=features,
    )

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "shap_waterfall.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP chart to {chart_path}")

    print("\nTop SHAP contributors:")
    for c in top_contributors:
        print(f"{c['feature']}: value={c['value']}, shap={c['shap']} → {c['impact']}")

    print("\nHuman-readable explanation:")
    print(lime_style_summary)

    print("\nDone.")
