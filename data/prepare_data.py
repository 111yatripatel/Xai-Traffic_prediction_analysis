import pandas as pd
import numpy as np
import h5py
import os

print("Step 1: Loading METR-LA...")

# Load METR-LA
with h5py.File("data/raw/METR-LA.h5", "r") as f:
    data = f["df"]["block0_values"][:]
    index = f["df"]["axis1"][:]

# Convert timestamp index safely
if hasattr(index[0], "decode"):
    timestamps = pd.to_datetime([t.decode("utf-8") for t in index])
else:
    timestamps = pd.to_datetime(index)

df_raw = pd.DataFrame(data, index=timestamps)

# Take first 8 columns and map to Ahmedabad corridors
corridor_names = [
    "SG_Highway",
    "Ring_Road_132ft",
    "CG_Road",
    "Ashram_Road",
    "Sardar_Patel_Ring",
    "Narol_Naroda",
    "Maninagar",
    "Stadium_Motera",
]

df_corridors = df_raw.iloc[:, :8].copy()
df_corridors.columns = corridor_names

print(f"Loaded {len(df_corridors)} rows × {len(corridor_names)} corridors")
print(f"Date range: {df_corridors.index[0]} → {df_corridors.index[-1]}")

# Free flow speed assumptions
FREE_FLOW = {
    "SG_Highway": 70,
    "Ring_Road_132ft": 60,
    "CG_Road": 50,
    "Ashram_Road": 50,
    "Sardar_Patel_Ring": 80,
    "Narol_Naroda": 70,
    "Maninagar": 40,
    "Stadium_Motera": 60,
}

print("\nStep 2: Converting speed to congestion %...")

rows = []

for ts, row in df_corridors.iterrows():
    for corridor in corridor_names:
        speed = float(row[corridor])

        if speed <= 0:
            continue

        ff_speed = FREE_FLOW[corridor]
        congestion_pct = round(max(0, (1 - speed / ff_speed) * 100), 2)

        rows.append({
            "timestamp": ts,
            "corridor_name": corridor,
            "current_speed": round(speed, 2),
            "freeflow_speed": ff_speed,
            "congestion_pct": congestion_pct,
            "confidence": 0.85,
        })

df = pd.DataFrame(rows)

print(f"Total rows after conversion: {len(df)}")

print("\nStep 3: Engineering time features...")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday
df["is_weekend"] = df["weekday"] >= 5
df["is_morning_rush"] = (df["hour"].between(8, 10)) & (~df["is_weekend"])
df["is_evening_rush"] = (df["hour"].between(17, 20)) & (~df["is_weekend"])
df["is_school_hour"] = (df["hour"].between(7, 14)) & (~df["is_weekend"])
df["month"] = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

print("\nStep 4: Adding Ahmedabad weather patterns...")

np.random.seed(42)
n = len(df)
month = df["month"]

temp = np.where(
    month.between(4, 6), np.random.normal(40, 3, n),
    np.where(
        month.between(7, 9), np.random.normal(32, 4, n),
        np.where(
            month.between(10, 12), np.random.normal(25, 5, n),
            np.random.normal(22, 4, n)
        )
    )
)

rain_prob = np.where(
    month.between(7, 9), 0.35,
    np.where(month.between(6, 10), 0.08, 0.02)
)

is_rain = np.random.random(n) < rain_prob

df["temp_celsius"] = np.round(temp, 1)
df["humidity"] = np.where(
    month.between(7, 9),
    np.random.randint(70, 95, n),
    np.random.randint(30, 60, n)
)
df["wind_speed"] = np.round(np.random.exponential(3, n), 1)
df["is_rain"] = is_rain

df.loc[df["is_rain"], "congestion_pct"] = (
    df.loc[df["is_rain"], "congestion_pct"] *
    np.random.uniform(1.15, 1.35, df["is_rain"].sum())
).clip(0, 100).round(2)

print("\nStep 5: Adding Ahmedabad festival calendar...")

festival_dates = {
    (1, 14): ("Uttarayan", "VERY_HIGH"),
    (3, 25): ("Holi", "MEDIUM"),
    (4, 14): ("Ambedkar_Jayanti", "MEDIUM"),
    (7, 7): ("Rath_Yatra", "VERY_HIGH"),
    (8, 15): ("Independence_Day", "MEDIUM"),
    (10, 2): ("Gandhi_Jayanti", "LOW"),
    (10, 3): ("Navratri", "HIGH"),
    (10, 4): ("Navratri", "HIGH"),
    (10, 5): ("Navratri", "HIGH"),
    (10, 6): ("Navratri", "HIGH"),
    (10, 7): ("Navratri", "VERY_HIGH"),
    (10, 8): ("Navratri", "VERY_HIGH"),
    (10, 9): ("Navratri", "VERY_HIGH"),
    (11, 1): ("Diwali", "HIGH"),
    (11, 2): ("Diwali", "HIGH"),
}

df["is_festival"] = False
df["event_name"] = "none"
df["event_severity"] = "NONE"

for (m, d), (name, severity) in festival_dates.items():
    mask = (df["timestamp"].dt.month == m) & (df["timestamp"].dt.day == d)
    df.loc[mask, "is_festival"] = True
    df.loc[mask, "event_name"] = name
    df.loc[mask, "event_severity"] = severity

festival_mask = df["is_festival"]
high_mask = festival_mask & df["event_severity"].isin(["HIGH", "VERY_HIGH"])

df.loc[high_mask, "congestion_pct"] = (
    df.loc[high_mask, "congestion_pct"] *
    np.random.uniform(1.2, 1.5, high_mask.sum())
).clip(0, 100).round(2)

ipl_days = [22, 35, 49, 63, 77, 91]

ipl_mask = (
    (df["corridor_name"] == "Stadium_Motera") &
    (df["day_of_year"].isin(ipl_days)) &
    (df["hour"].between(17, 22))
)

df.loc[ipl_mask, "congestion_pct"] = (
    df.loc[ipl_mask, "congestion_pct"] *
    np.random.uniform(1.4, 1.8, ipl_mask.sum())
).clip(0, 100).round(2)

df.loc[ipl_mask, "is_festival"] = True
df.loc[ipl_mask, "event_name"] = "IPL_Match"
df.loc[ipl_mask, "event_severity"] = "HIGH"

corridor_map = {c: i for i, c in enumerate(corridor_names)}
df["corridor_id"] = df["corridor_name"].map(corridor_map)

def get_severity(pct):
    if pct < 25:
        return "LOW"
    if pct < 50:
        return "MEDIUM"
    if pct < 75:
        return "HIGH"
    return "SEVERE"

df["severity"] = df["congestion_pct"].apply(get_severity)

print("\nStep 6: Saving processed dataset...")

os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/ahmedabad_training_data.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")
print(f"Total rows: {len(df):,}")
print(f"Corridors: {df['corridor_name'].nunique()}")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print("\nSeverity split:")
print(df["severity"].value_counts().to_string())
print(f"\nFestival rows: {df['is_festival'].sum():,}")
print(f"Rain rows: {df['is_rain'].sum():,}")
print("\nDone! Ready for model training.")