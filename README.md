# Ahmedabad XAI Traffic Intelligence

An explainable traffic-congestion prediction prototype for Ahmedabad corridors. The system combines an XGBoost regression model, SHAP feature attribution, a FastAPI backend, and a React command-center dashboard.

## Problem Statement

Traffic prediction models can estimate congestion, but a percentage alone does not tell an operator why the model produced that result. Without an explanation, predictions are harder to validate, communicate, and use in decision support.

## Core Idea

The user selects a traffic scenario containing a corridor, hour, rain condition, and festival/event condition. The backend passes these features to an XGBoost model, classifies the predicted congestion severity, calculates SHAP contributions, and stores a reasoning trace for audit and history.

```text
Scenario selection
      |
      v
FastAPI prediction request
      |
      v
XGBoost congestion prediction
      |
      v
SHAP feature explanation
      |
      v
Saved reasoning trace and decision-support dashboard
```

## Why Explainability Matters

Explainability helps users distinguish between a model result and the factors that produced it. In this prototype:

- Positive SHAP values indicate features that pushed predicted congestion higher.
- Negative SHAP values indicate features that reduced predicted congestion.
- Counterfactual guidance describes conditions that would need to change to reduce congestion.
- Saved reasoning traces make previous predictions inspectable and auditable.

## Technology Stack

- Backend: Python, FastAPI, Pydantic, Pandas
- Machine learning: XGBoost, scikit-learn, Joblib
- Explainability: SHAP TreeExplainer
- Frontend: React, TypeScript, Vite
- Storage: processed CSV input and JSON reasoning traces

## Features

- Predictions for eight Ahmedabad corridors
- Scenario controls for corridor, hour, rain, and festival/event activity
- Four one-click demo scenarios
- Live network dashboard with severity-colored corridor cards
- Abstract Ahmedabad map with clickable corridor paths
- Congestion percentage, severity, confidence, and explanation
- Visual positive and negative SHAP contribution bars
- Counterfactual guidance
- Severe/high congestion anomaly feed
- Persistent trace history with full trace retrieval
- Model metadata and feature inventory
- Responsive loading, empty, and error states

## Project Structure

```text
Xai-Reasoning Traces/
|-- backend/
|   |-- app/
|   |   `-- main.py                     # Active FastAPI entry point
|   `-- main.py                         # Alternate backend module
|-- data/
|   `-- processed/
|       `-- ahmedabad_training_data.csv
|-- frontend/
|   |-- public/
|   |-- src/
|   |   |-- App.tsx                     # UI and typed API integration
|   |   |-- App.css                     # Dashboard visual system
|   |   |-- index.css
|   |   `-- main.tsx
|   |-- index.html
|   `-- package.json
|-- models/
|   |-- model_features.pkl
|   `-- xgboost_traffic_model.pkl
|-- outputs/
|   `-- traces/                         # Saved reasoning traces
|-- notebooks/
|-- tests/
|-- train_model.py
|-- generate_trace.py
|-- requirements.txt
`-- README.md
```

## Setup

Run the following commands from the project root using PowerShell.

### Python environment

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Frontend dependencies

```powershell
cd frontend
npm.cmd install
cd ..
```

## Run the Project

Open two terminals.

### Backend

From the project root:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Useful backend URLs:

- API root: `http://127.0.0.1:8000`
- Health check: `http://127.0.0.1:8000/health`
- Interactive documentation: `http://127.0.0.1:8000/docs`

### Frontend

```powershell
cd frontend
npm.cmd run dev
```

Open the URL printed by Vite, normally:

```text
http://localhost:5173
```

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic API status |
| `GET` | `/health` | Backend and model health |
| `GET` | `/api/v1/segments` | Supported corridor identifiers |
| `POST` | `/api/v1/predict` | Generate one prediction and reasoning trace |
| `GET` | `/api/v1/dashboard` | Predict all eight corridors for one scenario |
| `GET` | `/api/v1/anomalies` | Return demo-friendly congestion alerts |
| `GET` | `/api/v1/traces?limit=10` | List recent traces |
| `GET` | `/api/v1/traces/{trace_id}` | Retrieve a complete trace |
| `GET` | `/api/v1/model-info` | Return model and feature metadata |

Example prediction request:

```json
{
  "corridor_name": "SG_Highway",
  "hour": 8,
  "is_rain": true,
  "is_festival": false
}
```

## Demo Scenario

### SG Highway, 8 AM, Rain Detected

1. Start the backend and frontend.
2. Open `http://localhost:5173`.
3. Click **Morning Rain on SG Highway**, or configure the scenario manually:
   - Select **SG Highway**.
   - Set the hour to **08:00**.
   - Turn **Rain** on.
   - Leave **Festival / event** off.
4. Click **Run prediction**.
5. Explain the congestion percentage and severity returned by XGBoost.
6. Point to the SHAP bars:
   - Positive values push congestion higher.
   - Negative values reduce predicted congestion.
7. Explain the counterfactual as a description of what could change to reduce congestion.
8. Open the newest trace in **Recent trace history** to show that the reasoning result was saved.
9. Refresh the live dashboard to compare all eight corridors under the same scenario.

## Additional Demo Presets

- Morning Rain on SG Highway
- Festival Evening at Stadium Motera
- Normal Afternoon on CG Road
- Rush Hour on Ashram Road

Each preset fills the scenario controls and immediately runs a focused prediction.

## Verification

Frontend:

```powershell
cd frontend
npm.cmd run lint
npm.cmd run build
```

Backend syntax:

```powershell
.\venv\Scripts\python.exe -m py_compile backend\app\main.py
```

## Limitations

This is a prototype using processed and Ahmedabad-style mapped traffic features. It is not a deployed city-scale traffic system.

Additional limitations:

- The dashboard does not consume live municipal traffic sensors.
- The map is an abstract corridor visualization, not a navigation-grade geospatial map.
- Confidence values come from the prototype dataset rather than a separately calibrated uncertainty model.
- Counterfactual guidance is explanatory rule-based text, not an optimized traffic-control intervention.
- JSON files are suitable for MVP trace storage but not high-volume production workloads.
- Predictions should be treated as demonstrations of the ML/XAI workflow rather than real-time travel guidance.

## Future Improvements

- Integrate validated live or periodically updated traffic feeds.
- Add geospatial corridor geometry and map tiles.
- Calibrate predictive uncertainty independently from model confidence.
- Generate data-driven counterfactual scenarios with feasibility constraints.
- Add automated backend and browser interaction tests.
- Evaluate model drift and explanation stability over time.
- Introduce production storage only if deployment scale requires it.
