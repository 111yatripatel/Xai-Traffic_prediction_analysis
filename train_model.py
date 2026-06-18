import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

print("Loading dataset...")

df = pd.read_csv("data/processed/ahmedabad_training_data.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Convert boolean columns to integers
bool_cols = [
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_school_hour",
    "is_rain",
    "is_festival",
]

for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Features used for prediction
features = [
    "current_speed",
    "freeflow_speed",
    "confidence",
    "hour",
    "weekday",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_school_hour",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "temp_celsius",
    "humidity",
    "wind_speed",
    "is_rain",
    "is_festival",
    "corridor_id",
]

target = "congestion_pct"

# Keep only available columns
features = [col for col in features if col in df.columns]

X = df[features]
y = df[target]

print("Training features:", features)
print("Target:", target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

print("Evaluating model...")

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nModel Results")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R2 Score:", round(r2, 4))

joblib.dump(model, "models/xgboost_traffic_model.pkl")
joblib.dump(features, "models/model_features.pkl")

print("\nSaved model to models/xgboost_traffic_model.pkl")
print("Saved features to models/model_features.pkl")
print("Done.")
