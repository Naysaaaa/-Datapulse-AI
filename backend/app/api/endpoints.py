import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.app.db.database import engine, Base, get_db
from backend.app.core.config import settings
from backend.app.db import models
from backend.app import schemas
from backend.app.ml.pipeline import pipeline_manager
from backend.app.services.scheduler import telemetry_scheduler
from backend.app.services.ollama_service import OllamaNarrativeService
from backend.app.services.pdf_service import pdf_report_generator

logger = logging.getLogger(__name__)

router = APIRouter()

# Security Setup
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

# Mock User Credentials for JWT Auth Stub
MOCK_USER = {
    "username": "admin",
    "hashed_password": pwd_context.hash("admin123"), # Default password: admin123
    "disabled": False
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username != MOCK_USER["username"]:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data


# --- AUTHENTICATION ENDPOINT ---
@router.post("/auth/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != MOCK_USER["username"] or not verify_password(form_data.password, MOCK_USER["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# --- TELEMETRY STREAM ENDPOINTS ---
@router.get("/telemetry", response_model=List[schemas.TelemetryResponse])
def get_telemetry(source: Optional[str] = None, limit: int = 100, db: Session = Depends(get_db)):
    """Fetch recent telemetry logs."""
    query = db.query(models.TelemetryData)
    if source:
        query = query.filter(models.TelemetryData.source == source)
    return query.order_by(models.TelemetryData.timestamp.desc()).limit(limit).all()

@router.post("/telemetry/stream/start")
def start_stream(current_user: schemas.TokenData = Depends(get_current_user)):
    """Starts the real-time simulation engine."""
    if telemetry_scheduler.is_streaming:
        return {"status": "already running"}
    telemetry_scheduler.start_stream()
    return {"status": "started"}

@router.post("/telemetry/stream/stop")
def stop_stream(current_user: schemas.TokenData = Depends(get_current_user)):
    """Stops the real-time simulation engine."""
    if not telemetry_scheduler.is_streaming:
        return {"status": "not running"}
    telemetry_scheduler.stop_stream()
    return {"status": "stopped"}

@router.get("/telemetry/stream/status")
def get_stream_status():
    """Returns background ingestion engine details."""
    return {
        "is_streaming": telemetry_scheduler.is_streaming,
        "drift_active": telemetry_scheduler.drift_active,
        "anomaly_active": telemetry_scheduler.anomaly_active,
        "noise_level": telemetry_scheduler.noise_level,
        "interval_seconds": settings.SIMULATION_INTERVAL_SECONDS
    }

@router.post("/telemetry/stream/config")
def configure_stream(drift: Optional[bool] = None, anomaly: Optional[bool] = None, noise: Optional[float] = None, current_user: schemas.TokenData = Depends(get_current_user)):
    """Configure stream profile (drift, anomalies, noise level)."""
    telemetry_scheduler.configure_simulation(drift=drift, anomaly=anomaly, noise=noise)
    return {"status": "configured"}


# --- CSV UPLOAD & RETRAINING ENDPOINT ---
@router.post("/telemetry/upload")
async def upload_csv_file(file: UploadFile = File(...), current_user: schemas.TokenData = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Accepts CSV uploads of telemetry, validates structure, saves as new baseline,
    and retrains pipelines.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        # Validate columns
        required_cols = pipeline_manager.feature_cols
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV is missing required telemetry fields: {missing_cols}. Columns must match: {required_cols}"
            )
            
        # Parse or create timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            start_time = datetime.utcnow() - timedelta(hours=len(df))
            df['timestamp'] = [start_time + timedelta(hours=i) for i in range(len(df))]
            
        # Sort values
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Save custom baseline & Retrain models
        retrain_success = pipeline_manager.set_custom_baseline(df)
        if not retrain_success:
            raise HTTPException(status_code=500, detail="Failed to retrain models on new baseline dataset.")
            
        # Clear existing telemetry data to align with new baseline
        db.query(models.TelemetryData).delete()
        db.query(models.DriftHistory).delete()
        db.query(models.SystemAlert).delete()
        
        # Load CSV rows as Initial Telemetry Records (source = 'csv')
        for _, row in df.iterrows():
            telemetry = models.TelemetryData(
                timestamp=row['timestamp'],
                cpu_utilization=float(row['cpu_utilization']),
                memory_utilization=float(row['memory_utilization']),
                network_latency=float(row['network_latency']),
                error_rate=float(row['error_rate']),
                kpi_value=float(row['kpi_value']),
                source="csv",
                is_anomaly=False,
                forecast_value=None,
                failure_probability=0.05
            )
            db.add(telemetry)
            
        # Record ModelMetadata/History in alerts
        alert = models.SystemAlert(
            timestamp=datetime.utcnow(),
            type="health",
            message=f"Model retraining completed successfully on newly uploaded baseline ({len(df)} points).",
            severity="WARNING"
        )
        db.add(alert)
        db.commit()
        
        return {
            "status": "success",
            "rows_ingested": len(df),
            "message": "Model pipelines retrained successfully on new baseline dataset."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling CSV upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


# --- MLOps DRIFT ENDPOINTS ---
@router.get("/drift/latest", response_model=Optional[schemas.DriftResponse])
def get_latest_drift(db: Session = Depends(get_db)):
    """Fetch latest computed MLOps drift audit report."""
    return db.query(models.DriftHistory).order_by(models.DriftHistory.timestamp.desc()).first()

@router.get("/drift/history", response_model=List[schemas.DriftResponse])
def get_drift_history(limit: int = 50, db: Session = Depends(get_db)):
    """Fetch historical record of drift check audits."""
    return db.query(models.DriftHistory).order_by(models.DriftHistory.timestamp.desc()).limit(limit).all()


# --- ALERTS ENDPOINTS ---
@router.get("/alerts", response_model=List[schemas.AlertResponse])
def get_alerts(unresolved_only: bool = True, limit: int = 50, db: Session = Depends(get_db)):
    """Fetch system alarms (anomalies, failures, drift)."""
    query = db.query(models.SystemAlert)
    if unresolved_only:
        query = query.filter(models.SystemAlert.is_resolved == False)
    return query.order_by(models.SystemAlert.timestamp.desc()).limit(limit).all()

@router.post("/alerts/{alert_id}/resolve", response_model=schemas.AlertResponse)
def resolve_alert(alert_id: int, payload: schemas.AlertUpdate, current_user: schemas.TokenData = Depends(get_current_user), db: Session = Depends(get_db)):
    """Mark a system alarm as resolved."""
    alert = db.query(models.SystemAlert).filter(models.SystemAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_resolved = payload.is_resolved
    db.commit()
    db.refresh(alert)
    return alert


# --- NARRATIVE RETRIEVAL ---
@router.post("/narrative/generate", response_model=schemas.NarrativeResponse)
async def generate_narrative(db: Session = Depends(get_db)):
    """
    Assembles recent drift logs, alerts, and telemetry trends.
    Calls local Ollama (or fallback) to synthesize plain-English executive analysis.
    Saves and returns the report.
    """
    # Fetch overview details
    latest_tele = db.query(models.TelemetryData).order_by(models.TelemetryData.timestamp.desc()).first()
    if not latest_tele:
        raise HTTPException(status_code=404, detail="No telemetry available to generate reports. Please start simulation stream.")
        
    latest_drift = db.query(models.DriftHistory).order_by(models.DriftHistory.timestamp.desc()).first()
    active_alarms = db.query(models.SystemAlert).filter(models.SystemAlert.is_resolved == False).all()
    total_points = db.query(models.TelemetryData).count()
    anomaly_count = db.query(models.TelemetryData).filter(models.TelemetryData.is_anomaly == True).count()
    
    # Pack report variables
    report_data = {
        "system_status": latest_drift.system_status if latest_drift else "STABLE",
        "feature_psi": latest_drift.feature_psi if latest_drift else {},
        "prediction_drift": latest_drift.prediction_drift if latest_drift else 0.0,
        "anomaly_count": anomaly_count,
        "total_points": total_points,
        "latest_telemetry": {
            "cpu_utilization": latest_tele.cpu_utilization,
            "memory_utilization": latest_tele.memory_utilization,
            "network_latency": latest_tele.network_latency,
            "error_rate": latest_tele.error_rate,
            "kpi_value": latest_tele.kpi_value
        },
        "forecast_value": latest_tele.forecast_value,
        "active_alerts": [a.message for a in active_alarms]
    }
    
    # Query Ollama narrative service
    service = OllamaNarrativeService()
    narrative_text, is_mocked = await service.generate_narrative(report_data)
    
    # Save narrative
    narrative_entry = models.ExecutiveNarrative(
        timestamp=datetime.utcnow(),
        narrative=narrative_text,
        is_mocked=is_mocked
    )
    db.add(narrative_entry)
    db.commit()
    db.refresh(narrative_entry)
    
    return narrative_entry

@router.get("/narrative/latest", response_model=Optional[schemas.NarrativeResponse])
def get_latest_narrative(db: Session = Depends(get_db)):
    """Fetch the latest saved executive narrative."""
    return db.query(models.ExecutiveNarrative).order_by(models.ExecutiveNarrative.timestamp.desc()).first()


# --- PDF REPORT EXPORT ---
@router.get("/reports/pdf")
def download_pdf_report(db: Session = Depends(get_db)):
    """Generates the executive narrative and details in PDF format for immediate download."""
    latest_tele = db.query(models.TelemetryData).order_by(models.TelemetryData.timestamp.desc()).first()
    if not latest_tele:
        raise HTTPException(status_code=400, detail="Cannot generate report. Telemetry data is empty.")
        
    latest_drift = db.query(models.DriftHistory).order_by(models.DriftHistory.timestamp.desc()).first()
    latest_narrative = db.query(models.ExecutiveNarrative).order_by(models.ExecutiveNarrative.timestamp.desc()).first()
    active_alarms = db.query(models.SystemAlert).filter(models.SystemAlert.is_resolved == False).all()
    total_points = db.query(models.TelemetryData).count()
    
    # If no narrative has been generated yet, create one synchronously
    if not latest_narrative:
        # Fallback simulated builder done synchronously for simplicity
        # (avoid await in sync endpoints, call local _generate_simulated_narrative directly)
        service = OllamaNarrativeService()
        psi_dict = latest_drift.feature_psi if latest_drift else {}
        pred_drift_val = latest_drift.prediction_drift if latest_drift else 0.0
        sys_status_val = latest_drift.system_status if latest_drift else "STABLE"
        
        narrative_text = service._generate_simulated_narrative(
            system_status=sys_status_val,
            feature_psi=psi_dict,
            prediction_drift=pred_drift_val,
            anomaly_count=0,
            total_points=total_points,
            latest_telemetry={
                "cpu_utilization": latest_tele.cpu_utilization,
                "memory_utilization": latest_tele.memory_utilization,
                "network_latency": latest_tele.network_latency,
                "error_rate": latest_tele.error_rate,
                "kpi_value": latest_tele.kpi_value
            },
            forecast_value=latest_tele.forecast_value or latest_tele.kpi_value,
            active_alerts=[a.message for a in active_alarms]
        )
        is_mocked = True
    else:
        narrative_text = latest_narrative.narrative
        is_mocked = latest_narrative.is_mocked

    report_payload = {
        "system_status": latest_drift.system_status if latest_drift else "STABLE",
        "feature_psi": latest_drift.feature_psi if latest_drift else {
            col: 0.0 for col in pipeline_manager.feature_cols
        },
        "prediction_drift": latest_drift.prediction_drift if latest_drift else 0.0,
        "total_points": total_points,
        "latest_telemetry": {
            "cpu_utilization": latest_tele.cpu_utilization,
            "memory_utilization": latest_tele.memory_utilization,
            "network_latency": latest_tele.network_latency,
            "error_rate": latest_tele.error_rate,
            "kpi_value": latest_tele.kpi_value
        },
        "active_alerts": [a.message for a in active_alarms],
        "narrative": narrative_text
    }
    
    pdf_buffer = pdf_report_generator.generate_report(report_payload)
    
    filename = f"DataPulse_Executive_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
