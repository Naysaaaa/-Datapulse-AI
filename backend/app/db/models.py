from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, JSON, Text
from datetime import datetime
from backend.app.db.database import Base

class TelemetryData(Base):
    __tablename__ = "telemetry_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_utilization = Column(Float, nullable=False)
    memory_utilization = Column(Float, nullable=False)
    network_latency = Column(Float, nullable=False)
    error_rate = Column(Float, nullable=False)
    kpi_value = Column(Float, nullable=False)
    source = Column(String, default="stream")  # 'stream' or 'csv'
    
    # ML Inference Results
    is_anomaly = Column(Boolean, default=False)
    forecast_value = Column(Float, nullable=True)  # Forecasted value for next timestamp
    failure_probability = Column(Float, default=0.05)  # Health failure probability


class DriftHistory(Base):
    __tablename__ = "drift_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    system_status = Column(String, default="STABLE")  # STABLE, DEGRADED, DRIFTING
    feature_psi = Column(JSON, nullable=False)  # Dictionary of feature PSI values
    prediction_drift = Column(Float, nullable=False)  # PSI of predictions
    data_quality_report = Column(JSON, nullable=False)  # Data quality metrics dict


class SystemAlert(Base):
    __tablename__ = "system_alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    type = Column(String, nullable=False)  # 'drift', 'anomaly', 'health', 'quality'
    message = Column(String, nullable=False)
    severity = Column(String, default="WARNING")  # WARNING, CRITICAL
    is_resolved = Column(Boolean, default=False)


class ExecutiveNarrative(Base):
    __tablename__ = "executive_narratives"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    narrative = Column(Text, nullable=False)
    is_mocked = Column(Boolean, default=False)  # True if generated via rule-fallback
