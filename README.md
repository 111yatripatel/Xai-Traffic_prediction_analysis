# Ahmedabad XAI Traffic Intelligence

Ahmedabad XAI Traffic Intelligence is an explainable traffic-congestion prediction prototype. It combines an XGBoost regression model with SHAP feature attribution and presents the results through a FastAPI backend and a React command-center dashboard.

## Problem Statement

Traffic prediction systems can estimate congestion, but a prediction alone does not explain why the model reached that result. For traffic operators, planners, and students evaluating machine-learning systems, this lack of transparency makes model output harder to assess and act upon.

This project addresses that problem by pairing each congestion prediction with a saved reasoning trace. The trace records the model output, influential features, SHAP contributions, contextual inputs, and a simple counterfactual recommendation.

## Core Idea

The system accepts a corridor and scenario context such as hour, rain, and festival activity. An existing XGBoost model predicts congestion percentage and severity. SHAP then estimates how each feature increased or decreased the prediction, and the backend saves this explanation as a JSON trace for later inspection.

System flow:

```text
Ahmedabad-style dataset
        |
        v
XGBoost congestion prediction
        |
        v
SHAP reasoning trace
        |
        v
Traffic decision-support dashboard
```

## Technology Stack

- Backend: Python, FastAPI, Pydantic, Pandas
- Machine learning: XGBoost, scikit-learn, Joblib
- Explainability: SHAP
- Frontend: React 19, TypeScript, Vite
- Data and trace storage: CSV input and JSON reasoning traces

## Features

- Congestion prediction for eight Ahmedabad corridors
- Scenario controls for hour, rainfall, and festival/event activity
- Network dashboard with congestion percentage and severity
- Abstract map-based corridor visualization
- SHAP contribution bars showing positive and negative feature effects
- Human-readable explanation and counterfactual guidance
- Persistent JSON reasoning traces
- Recent trace-history viewer
- Demo-friendly severe-congestion anomaly feed
- Responsive dark civic-tech interface
- Explicit loading, empty, and API error states

## Monitored Corridors

- SG Highway
- 132 ft Ring Road
- CG Road
- Ashram Road
- Sardar Patel Ring Road
- Narol-Naroda
- Maninagar
- Stadium Motera

## Project Structure

```text
Xai-Reasoning Traces/
├── backend/
│   ├── main.py                         # Active FastAPI application
│   └── app/
│       └── main.py                     # Earlier backend implementation
├── data/
│   └── processed/
│       └── ahmedabad_training_data.csv
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.tsx                     # Dashboard logic and typed API client
│   │   ├── App.css                     # Command-center interface styling
│   │   ├── index.css
│   │   └── main.tsx
│   ├── index.html
│   └── package.json
├── models/
│   ├── model_features.pkl
│   └── xgboost_traffic_model.pkl
├── outputs/
│   └── traces/                         # Generated SHAP reasoning traces
├── notebooks/
├── tests/
├── generate_trace.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup

The commands below use PowerShell on Windows.

### 1. Create and activate a Python environment

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, the environment's Python executable can be used directly in all commands:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Install frontend dependencies

```powershell
cd frontend
npm.cmd install
cd ..
```

## Running the Application

Open two terminals at the project root.

### Backend

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URLs:

- API root: `http://127.0.0.1:8000`
- Interactive documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Frontend

```powershell
cd frontend
npm.cmd run dev
```

Open the URL printed by Vite, normally:

```text
http://localhost:5173
```

The backend permits local frontend origins on ports `5173`, `5174`, and `5175`.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic API status |
| `GET` | `/health` | Model and corridor health information |
| `GET` | `/api/v1/segments` | List supported corridor identifiers |
| `POST` | `/api/v1/predict` | Generate one prediction and SHAP reasoning trace |
| `GET` | `/api/v1/dashboard` | Generate dashboard predictions for all eight corridors |
| `GET` | `/api/v1/anomalies` | Return severe, demo-friendly anomaly records |
| `GET` | `/api/v1/traces?limit=10` | List recent reasoning traces |
| `GET` | `/api/v1/traces/{trace_id}` | Retrieve one complete reasoning trace |

### Prediction Request Example

```json
{
  "corridor_name": "SG_Highway",
  "hour": 8,
  "is_rain": true,
  "is_festival": false
}
```

### Dashboard Query Example

```text
GET /api/v1/dashboard?hour=8&is_rain=true&is_festival=false
```

## Recommended Demo Flow

1. Start the FastAPI backend and Vite frontend.
2. Open the dashboard and confirm that eight corridor cards and map lines load.
3. Review the network severity colors and operational anomaly feed.
4. Change the scenario controls and refresh the network dashboard.
5. Select a corridor on the map or from the corridor cards.
6. Review congestion severity, explanation, SHAP contribution bars, and counterfactual guidance.
7. Open a recent trace to demonstrate that reasoning results are persisted and retrievable.

## Demo Story: SG Highway, 8 AM, Rain Detected

Scenario: morning traffic on SG Highway with rainfall.

1. Select **SG Highway**.
2. Set the hour to **08:00**.
3. Enable the **Rain** toggle.
4. Leave the festival/event toggle off.
5. Click **Run Prediction**.
6. The system predicts a congestion percentage and severity.
7. The SHAP trace identifies the strongest contributing features and whether each feature increased or reduced congestion.
8. The counterfactual section suggests conditions that could reduce congestion, such as improved current speed or reduced rush-hour pressure.
9. Open the latest item in trace history to show that the complete reasoning trace was saved.

## Verification Commands

Frontend static checks:

```powershell
cd frontend
npm.cmd run lint
npm.cmd run build
```

Backend syntax check:

```powershell
.\venv\Scripts\python.exe -m py_compile backend\main.py
```

## Prototype Scope and Limitations

This is an academic and portfolio prototype, not a deployed city-scale traffic-management system. It uses transfer/simulated Ahmedabad-style traffic features from a processed dataset rather than a live municipal sensor network. Predictions should therefore be interpreted as demonstrations of the machine-learning and explainability workflow, not as real-time operational traffic guidance.

Additional current limitations:

- The map is an abstract corridor visualization rather than a geospatial navigation map.
- Data is loaded from a local CSV file and is not streamed from live traffic sensors.
- Confidence is inherited from the prototype dataset rather than produced by a separate calibrated uncertainty model.
- Counterfactual guidance is rule-based explanatory text, not an optimized intervention plan.
- JSON trace storage is suitable for an MVP but not intended for high-volume production workloads.
- No authentication or multi-user access controls are included.

## Submission Summary

The project demonstrates an end-to-end explainable AI workflow: structured traffic features are processed by an XGBoost model, predictions are interpreted using SHAP, reasoning traces are persisted, and the results are presented in an interactive decision-support interface.
