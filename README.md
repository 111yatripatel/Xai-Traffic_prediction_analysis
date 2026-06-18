# Explainable Traffic Congestion Prediction using XGBoost and SHAP

An explainable machine-learning prototype that predicts traffic congestion for Ahmedabad-style road corridors and shows why the model produced each result.

## Problem Statement

Traffic congestion prediction can support planning and traffic-management decisions, but a prediction alone is difficult to trust. A model may estimate a congestion percentage without explaining which road, speed, weather, event, or time-related conditions influenced it.

This project addresses that limitation by combining traffic prediction with a human-readable explanation for every result.

## Core Idea

The user builds a traffic scenario by selecting a road corridor, hour, rain condition, and festival/event condition. The FastAPI backend prepares the scenario features, runs an XGBoost regression model, classifies the congestion severity, and calculates SHAP feature contributions.

Each prediction produces:

- A congestion percentage
- A severity level: `LOW`, `MEDIUM`, `HIGH`, or `SEVERE`
- A human-readable explanation
- Positive and negative SHAP contributions
- A counterfactual suggestion
- A saved reasoning trace for later review

```text
Traffic scenario
      |
      v
XGBoost congestion prediction
      |
      v
SHAP feature contributions
      |
      v
Human-readable explanation and saved reasoning trace
```

## Why Explainability Matters

Explainability helps users understand whether a prediction is reasonable instead of treating the model as a black box.

SHAP values describe how much each feature influenced one specific prediction:

- A positive SHAP value increased predicted congestion.
- A negative SHAP value reduced predicted congestion.

For example, the interface can show whether current speed, free-flow speed, rain, corridor patterns, or time-of-day patterns pushed a prediction higher or lower. This makes the result easier to inspect, communicate, and discuss during evaluation.

## Tech Stack

- **Frontend:** React, TypeScript, Vite
- **Backend:** FastAPI
- **ML model:** XGBoost
- **Explainability:** SHAP
- **Data processing:** Pandas, NumPy
- **Visualization:** CSS-based dashboard visuals

## Dataset and Features

The prototype uses processed Ahmedabad-style traffic data. Representative features include:

- Corridor
- Current speed
- Free-flow speed
- Hour and cyclical time patterns
- Rain condition
- Festival/event condition
- Morning, evening, and school-hour flags
- Day-of-week and corridor patterns

The dataset is intended to demonstrate the complete prediction and explainability workflow. It is not a live municipal traffic feed.

## Features

- Scenario-based traffic congestion prediction
- Congestion severity classification
- SHAP-based feature explanations
- Human-readable prediction reasoning
- Positive and negative SHAP contribution bars
- Saved reasoning trace history
- Counterfactual congestion guidance
- Traffic alerts
- Corridor ranking and network analytics
- Abstract clickable corridor map
- Four one-click demo presets
- Responsive guided demo interface

## Backend APIs

The active FastAPI application is `backend.app.main:app`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check API, model, and data availability |
| `GET` | `/api/v1/segments` | List supported corridor identifiers |
| `POST` | `/api/v1/predict` | Generate a prediction and SHAP reasoning trace |
| `GET` | `/api/v1/dashboard` | Generate corridor-level dashboard predictions |
| `GET` | `/api/v1/anomalies` | Return traffic alerts derived from predictions |
| `GET` | `/api/v1/traces` | List recent reasoning traces |
| `GET` | `/api/v1/traces/{trace_id}` | Retrieve one complete reasoning trace |
| `GET` | `/api/v1/model-info` | Return model, feature, and prototype metadata |

Example prediction request:

```json
{
  "corridor_name": "Sardar_Patel_Ring",
  "hour": 8,
  "is_rain": true,
  "is_festival": false
}
```

Example response shape:

```json
{
  "prediction": {
    "congestion_pct": 63.74,
    "label": "HIGH"
  },
  "trace_id": "...",
  "segment_id": "Sardar_Patel_Ring",
  "explanation": "...",
  "top_factors": [],
  "counterfactual": "..."
}
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs` while the backend is running.

## How to Run

Open two PowerShell terminals.

### Backend

```powershell
cd "D:\xai-project\Xai-Reasoning Traces"
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

If the environment has not been created yet:

```powershell
cd "D:\xai-project\Xai-Reasoning Traces"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Frontend

```powershell
cd "D:\xai-project\Xai-Reasoning Traces\frontend"
npm install
npm run dev
```

### Browser

Open:

```text
http://localhost:5173
```

## Demo Presets

The scenario section includes four presentation-friendly presets:

- **Morning Rain — SG Highway**
- **Festival Evening — Stadium Motera**
- **Rush Hour — Sardar Patel Ring Road**
- **Normal Afternoon — CG Road**

Selecting a preset updates all scenario controls, runs the prediction automatically, highlights the selected corridor, and opens the prediction and SHAP reasoning flow.

## Final Demo Script

### Scenario

**Sardar Patel Ring Road, 8 AM, Rain On**

1. Open the dashboard at `http://localhost:5173`.
2. Select **Sardar Patel Ring Road** from the corridor list.
3. Set the hour to **8 AM**.
4. Turn **Rain** on.
5. Keep **Festival / event** off.
6. Click **Run Prediction**.
7. Show the predicted congestion percentage.
8. Show the congestion severity level.
9. Explain the SHAP factors, such as current speed, free-flow speed, rain condition, corridor pattern, and time-of-day pattern.
10. Point out that positive SHAP values increased predicted congestion and negative values reduced it.
11. Open **Recent Saved Explanations** and select the newest reasoning trace to demonstrate that the explanation was saved.

For a faster presentation, click **Rush Hour — Sardar Patel Ring Road**. The preset applies the same scenario and runs the prediction automatically.

> Prediction values may vary between requests because the prototype selects a representative processed row for the chosen corridor before applying the scenario conditions.

## Limitations

> This is a prototype using processed Ahmedabad-style traffic features. It demonstrates explainable traffic prediction and is not a live deployed city traffic system.

Additional practical limitations:

- The dashboard does not consume live traffic sensors or municipal APIs.
- The corridor map is an abstract visualization and is not navigation-grade GIS data.
- SHAP explains model behavior; it does not prove that a feature caused real-world congestion.
- Counterfactual guidance is explanatory prototype text rather than an optimized traffic-control plan.
- Predictions should not be used as live travel or public-safety guidance.

## Future Improvements

- Real-time traffic API integration
- GIS mapping with Leaflet and OpenStreetMap
- A larger real-world traffic dataset
- Model comparison and evaluation
- Application deployment
