import logging
import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler

from backend.app.db.database import SessionLocal, engine
from backend.app.db.models import TelemetryData, DriftHistory, SystemAlert, Base
from backend.app.ml.pipeline import pipeline_manager
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Ensure tables are created
Base.metadata.create_all(bind=engine)

class TelemetryStreamScheduler:
    """
    Background job scheduler that simulates incoming data streams,
    runs them through the ML pipeline, logs predictions, audits drift,
    and creates system alerts.
    """
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.is_streaming = False
        
        # State variables for simulation behavior (modifiable via REST endpoints)
        self.drift_active = False
        self.anomaly_active = False
        self.noise_level = 0.1  # baseline noise ratio

    def start_stream(self):
        """Starts the background simulation job."""
        if not self.is_streaming:
            self.scheduler.add_job(
                self._simulate_telemetry_tick,
                'interval',
                seconds=settings.SIMULATION_INTERVAL_SECONDS,
                id='telemetry_stream'
            )
            self.scheduler.start()
            self.is_streaming = True
            logger.info("Background telemetry stream scheduler started.")

    def stop_stream(self):
        """Stops the background simulation job."""
        if self.is_streaming:
            self.scheduler.remove_job('telemetry_stream')
            self.is_streaming = False
            logger.info("Background telemetry stream scheduler stopped.")

    def configure_simulation(self, drift: bool = None, anomaly: bool = None, noise: float = None):
        """Configures the telemetry generator profile on the fly."""
        if drift is not None:
            self.drift_active = drift
        if anomaly is not None:
            self.anomaly_active = anomaly
        if noise is not None:
            self.noise_level = max(0.0, min(1.0, noise))
        logger.info(f"Telemetry simulation profile updated: drift={self.drift_active}, anomaly={self.anomaly_active}, noise={self.noise_level}")

    def _generate_point(self, last_point: TelemetryData = None) -> dict:
        """
        Generates next telemetry data point.
        Incorporate drift/anomaly states if active.
        """
        hour = datetime.utcnow().hour
        
        # Base Diurnal KPI trend (diurnal seasonality + noise)
        base_kpi = 500.0 + 200.0 * float(random.uniform(0.9, 1.1)) * (1.0 + 0.3 * (hour % 6 - 3) / 3.0)
        
        # Default normal parameters
        cpu_mean, cpu_std = 42.0, 5.0
        mem_mean, mem_std = 58.0, 3.0
        latency_mean, latency_std = 115.0, 10.0
        error_mean, error_std = 0.4, 0.1
        kpi_mean, kpi_std = base_kpi, 25.0
        
        # Shift distributions if DRIFT is active
        if self.drift_active:
            cpu_mean = 84.0       # severe CPU jump
            mem_mean = 88.0       # memory spike
            latency_mean = 410.0  # high latency shift
            error_mean = 4.2      # error rate drift
            kpi_mean = base_kpi * 0.5  # drop transactional capacity by half
            kpi_std = 75.0        # higher variance
            
        # Add random anomalies if ANOMALY is active
        if self.anomaly_active or (random.random() < 0.04): # 4% random anomalies in normal stream
            # Anomaly is a brief, massive spike in error rate or latency
            if random.random() < 0.5:
                error_mean += 15.0  # catastrophic error rate spike
            else:
                latency_mean += 800.0 # system locking up

        # Apply noise levels
        noise_factor = self.noise_level * 10
        
        cpu = max(0.0, min(100.0, random.normalvariate(cpu_mean, cpu_std * (1 + noise_factor))))
        mem = max(0.0, min(100.0, random.normalvariate(mem_mean, mem_std * (1 + noise_factor))))
        latency = max(5.0, random.normalvariate(latency_mean, latency_std * (1 + noise_factor)))
        error = max(0.0, min(100.0, random.normalvariate(error_mean, error_std * (1 + noise_factor))))
        kpi = max(10.0, random.normalvariate(kpi_mean, kpi_std * (1 + noise_factor)))
        
        return {
            "timestamp": datetime.utcnow(),
            "cpu_utilization": cpu,
            "memory_utilization": mem,
            "network_latency": latency,
            "error_rate": error,
            "kpi_value": kpi,
            "source": "stream"
        }

    def _simulate_telemetry_tick(self):
        """Background execution unit executed every tick."""
        db: Session = SessionLocal()
        try:
            # 1. Fetch last record to maintain state if necessary
            last_record = db.query(TelemetryData).order_by(TelemetryData.timestamp.desc()).first()
            
            # 2. Generate new raw point
            raw_point = self._generate_point(last_record)
            
            # 3. Pull recent historical window for forecaster lag requirements
            historical_records = db.query(TelemetryData).order_by(TelemetryData.timestamp.desc()).limit(10).all()
            historical_buffer = []
            for rec in reversed(historical_records):
                historical_buffer.append({
                    "timestamp": rec.timestamp,
                    "cpu_utilization": rec.cpu_utilization,
                    "memory_utilization": rec.memory_utilization,
                    "network_latency": rec.network_latency,
                    "error_rate": rec.error_rate,
                    "kpi_value": rec.kpi_value
                })
                
            # 4. Process models (Inference)
            inference = pipeline_manager.run_inference(raw_point, historical_buffer)
            
            # 5. Save Telemetry Point to DB
            telemetry_entry = TelemetryData(
                timestamp=raw_point["timestamp"],
                cpu_utilization=raw_point["cpu_utilization"],
                memory_utilization=raw_point["memory_utilization"],
                network_latency=raw_point["network_latency"],
                error_rate=raw_point["error_rate"],
                kpi_value=raw_point["kpi_value"],
                source=raw_point["source"],
                is_anomaly=inference["is_anomaly"],
                forecast_value=inference["forecast_next_kpi"],
                failure_probability=inference["system_failure_probability"]
            )
            db.add(telemetry_entry)
            db.flush() # get telemetry_entry.id populated
            
            # 6. Log system alerts for model anomaly detections
            if telemetry_entry.is_anomaly:
                alert = SystemAlert(
                    timestamp=datetime.utcnow(),
                    type="anomaly",
                    message=f"Anomaly detected! Latency: {telemetry_entry.network_latency:.1f}ms, Error Rate: {telemetry_entry.error_rate:.2f}%, CPU: {telemetry_entry.cpu_utilization:.1f}%",
                    severity="CRITICAL"
                )
                db.add(alert)
                
            # Log alert for extreme system failure probabilities
            if telemetry_entry.failure_probability > 0.50:
                alert = SystemAlert(
                    timestamp=datetime.utcnow(),
                    type="health",
                    message=f"Critical health failure probability predicted: {telemetry_entry.failure_probability*100:.1f}%",
                    severity="CRITICAL" if telemetry_entry.failure_probability > 0.80 else "WARNING"
                )
                db.add(alert)
                
            # 7. Check model and feature drift
            # We fetch recent production stream logs (e.g. the last 50 events) to perform statistical comparisons
            recent_logs = db.query(TelemetryData).order_by(TelemetryData.timestamp.desc()).limit(50).all()
            if len(recent_logs) >= 10:
                prod_data = []
                for log in recent_logs:
                    prod_data.append({
                        "cpu_utilization": log.cpu_utilization,
                        "memory_utilization": log.memory_utilization,
                        "network_latency": log.network_latency,
                        "error_rate": log.error_rate,
                        "kpi_value": log.kpi_value
                    })
                    
                drift_report = pipeline_manager.compute_drift_report(prod_data)
                
                # Check if we should log a new drift report (e.g., every 5 ticks or if status changes to avoid SQLite bloat)
                # For demo purposes, we will compute and save drift every tick
                drift_entry = DriftHistory(
                    timestamp=datetime.utcnow(),
                    system_status=drift_report["system_status"],
                    feature_psi={k: v["psi"] for k, v in drift_report["features"].items()},
                    prediction_drift=drift_report["prediction_drift"],
                    data_quality_report=drift_report["data_quality"]
                )
                db.add(drift_entry)
                
                # If drift status has changed or is alert worthy, create alarms
                if drift_report["alerts"]:
                    for alert_msg in drift_report["alerts"]:
                        # Check if duplicate alert already exists in last 2 minutes
                        recent_alert = db.query(SystemAlert).filter(
                            SystemAlert.message == alert_msg,
                            SystemAlert.timestamp > datetime.utcnow() - timedelta(minutes=2)
                        ).first()
                        if not recent_alert:
                            alert = SystemAlert(
                                timestamp=datetime.utcnow(),
                                type="drift",
                                message=alert_msg,
                                severity="CRITICAL" if "significant" in alert_msg.lower() else "WARNING"
                            )
                            db.add(alert)
                            
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Error in background simulation tick: {e}")
        finally:
            db.close()

# Global scheduler instance
telemetry_scheduler = TelemetryStreamScheduler()
