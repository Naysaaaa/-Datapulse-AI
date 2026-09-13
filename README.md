# ⚡ DataPulse AI

> End-to-End Analytics, MLOps Monitoring & Ollama Executive Narration Dashboard

## Screenshots

### Real-Time Data Streams
![Real-Time Dashboard](screenshots/dashboard_realtime.png)

### MLOps Status — DRIFTING State
![MLOps Drifting](screenshots/mlops_drifting.png)

### KPI Forecasting & Health Risk
![Forecasting and Health](screenshots/forecasting_health.png)

## Tech Stack
| Layer | Tools |
|---|---|
| Backend API | FastAPI, SQLAlchemy, SQLite |
| Frontend | Streamlit, Plotly |
| ML Models | XGBoost, Isolation Forest, Logistic Regression |
| Background Engine | APScheduler |
| AI Narration | Ollama (Mistral) with rule-based fallback |
| Auth | JWT Bearer Tokens |

## Quick Start
```bash
pip install -r requirements.txt
python -m backend.app.main        # Terminal 1
streamlit run frontend/app.py     # Terminal 2
```
Login: `admin` / `admin123`
