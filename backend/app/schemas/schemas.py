from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, List, Optional, Any

# Telemetry Schemas
class TelemetryDataBase(BaseModel):
    cpu_utilization: float = Field(..., ge=0.0, le=100.0)
    memory_utilization: float = Field(..., ge=0.0, le=100.0)
    network_latency: float = Field(..., ge=0.0)
    error_rate: float = Field(..., ge=0.0, le=100.0)
    kpi_value: float = Field(..., ge=0.0)

class TelemetryCreate(TelemetryDataBase):
    timestamp: Optional[datetime] = None
    source: Optional[str] = "stream"

class TelemetryResponse(TelemetryDataBase):
    id: int
    timestamp: datetime
    source: str
    is_anomaly: bool
    forecast_value: Optional[float] = None
    failure_probability: float

    class Config:
        from_attributes = True

# Drift Schemas
class DriftResponse(BaseModel):
    id: int
    timestamp: datetime
    system_status: str
    feature_psi: Dict[str, Any]
    prediction_drift: float
    data_quality_report: Dict[str, Any]

    class Config:
        from_attributes = True

# Alert Schemas
class AlertResponse(BaseModel):
    id: int
    timestamp: datetime
    type: str
    message: str
    severity: str
    is_resolved: bool

    class Config:
        from_attributes = True

class AlertUpdate(BaseModel):
    is_resolved: bool

# Narrative Schemas
class NarrativeResponse(BaseModel):
    id: int
    timestamp: datetime
    narrative: str
    is_mocked: bool

    class Config:
        from_attributes = True

# Authentication Schemas
class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
