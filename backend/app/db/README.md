## Database Layer

Uses SQLite via SQLAlchemy ORM.

### Tables
- TelemetryData: stores every simulated telemetry event
- DriftHistory: stores PSI drift audit snapshots  
- SystemAlert: stores anomaly and drift triggered alarms
- ExecutiveNarrative: stores Ollama or fallback generated reports
