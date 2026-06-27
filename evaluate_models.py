import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "ahmedabad_training_data.csv"
FEATURES_PATH = ROOT_DIR / "models" / "model_features.pkl"
OUTPUT_PATH = ROOT_DIR / "outputs" / "model_comparison.json"
TARGET = "congestion_pct"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def make_json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


def evaluate_model(name, estimator, X_train, X_test, y_train, y_test):
    train_started = time.perf_counter()
    estimator.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - train_started

    inference_started = time.perf_counter()
    predictions = estimator.predict(X_test)
    inference_time_seconds = time.perf_counter() - inference_started

    mse = mean_squared_error(y_test, predictions)
    return {
        "model_name": name,
        "mae": make_json_safe(mean_absolute_error(y_test, predictions)),
        "rmse": make_json_safe(np.sqrt(mse)),
        "r2": make_json_safe(r2_score(y_test, predictions)),
        "training_time_seconds": make_json_safe(training_time_seconds),
        "average_inference_time_ms": make_json_safe(
            (inference_time_seconds / max(len(X_test), 1)) * 1000
        ),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    features = list(joblib.load(FEATURES_PATH))
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing model features in dataset: {missing_features}")
    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    for column in df.columns:
        if df[column].dtype == "bool":
            df[column] = df[column].astype(int)

    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = [
        ("Linear Regression", LinearRegression()),
        (
            "Random Forest Regressor",
            RandomForestRegressor(
                n_estimators=120,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "Gradient Boosting Regressor",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
        ),
        (
            "Current XGBoost Regressor",
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
            ),
        ),
    ]

    results = [
        evaluate_model(name, estimator, X_train, X_test, y_train, y_test)
        for name, estimator in models
    ]

    best_by_mae = min(results, key=lambda item: item["mae"])
    best_by_r2 = max(results, key=lambda item: item["r2"])
    payload = {
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "dataset_row_count": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "target": TARGET,
        "feature_count": int(len(features)),
        "features": features,
        "random_state": RANDOM_STATE,
        "models": results,
        "best_model_by_mae": best_by_mae["model_name"],
        "best_model_by_r2": best_by_r2["model_name"],
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)

    print(f"Saved model comparison to {OUTPUT_PATH}")
    for result in results:
        print(
            f"{result['model_name']}: "
            f"MAE={result['mae']:.4f}, "
            f"RMSE={result['rmse']:.4f}, "
            f"R2={result['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
