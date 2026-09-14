## Services Layer

### scheduler.py
APScheduler background job firing every N seconds.
Generates telemetry, runs ML inference, saves to DB, checks drift.
Supports drift injection, anomaly forcing, and noise level control.

### ollama_service.py
Connects to local Ollama Mistral to generate executive narratives.
Falls back to rule-based CDO-level generator if Ollama is unreachable.

### pdf_service.py
Generates downloadable PDF executive reports from latest telemetry and narrative.
