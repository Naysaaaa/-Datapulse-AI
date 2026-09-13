## System Architecture

### Request Flow
1. APScheduler fires every N seconds
2. Generates synthetic telemetry point
3. Runs through Isolation Forest, XGBoost, Logistic Regression
4. Saves telemetry and predictions to SQLite
5. Computes PSI drift against baseline
6. Creates alerts if thresholds breached
7. Streamlit frontend polls API every 5 seconds
8. Dashboard updates all charts and metrics

### Components
FastAPI Backend
├── api/endpoints.py — all REST routes
├── ml/pipeline.py — ML model manager
├── services/scheduler.py — background engine
├── services/ollama_service.py — LLM narrator
├── services/pdf_service.py — report generator
└── db/models.py — SQLAlchemy ORM tables

Streamlit Frontend
├── Executive Narrative tab
├── Real-Time Streams tab
├── MLOps and Drift tab
└── System Alarms tab
