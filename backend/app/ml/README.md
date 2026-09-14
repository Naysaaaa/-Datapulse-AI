## ML Pipeline

Three models run on every telemetry tick:

### Isolation Forest
Detects anomalies in incoming telemetry. Returns -1 for anomaly, 1 for normal.

### XGBoost Forecaster
Predicts next KPI (transaction volume) value using lag features from recent history.

### Logistic Regression
Classifies system health and outputs failure probability between 0 and 1.

All models serialized to data/models/ as .pkl files and reloaded on startup.
