import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Any, List

from backend.app.ml.models import AnomalyDetector, KPIForecaster, HealthClassifier
from backend.app.ml.drift import monitor_drift

logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class MLPipelineManager:
    """
    Manages loading, training, saving, and executing the 3 ML models:
    - Anomaly Detector (Isolation Forest)
    - KPI Forecaster (XGBoost/Linear Regression)
    - Health Classifier (Logistic Regression)
    Also manages the baseline dataset used for drift detection.
    """
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.forecaster = KPIForecaster()
        self.health_classifier = HealthClassifier()
        
        self.baseline_path = os.path.join(DATA_DIR, "baseline.csv")
        self.baseline_df = None
        self.feature_cols = ['cpu_utilization', 'memory_utilization', 'network_latency', 'error_rate', 'kpi_value']
        
        # Load models and baseline if available
        models_loaded = self.load_models()
        self.load_baseline()
        
        # If no baseline exists, generate a default one (which trains the models)
        if self.baseline_df is None:
            self.generate_default_baseline()
        elif not models_loaded:
            logger.info("Baseline dataset found, but models could not be loaded. Retraining models on existing baseline...")
            self.train_all(self.baseline_df)

    def generate_default_baseline(self, num_points=500) -> pd.DataFrame:
        """
        Generates synthetic standard baseline telemetry representing a healthy system.
        """
        logger.info(f"Generating synthetic baseline dataset ({num_points} points)...")
        np.random.seed(42)
        
        start_time = datetime.now() - timedelta(hours=num_points)
        timestamps = [start_time + timedelta(hours=i) for i in range(num_points)]
        
        # Safe, normal operating ranges
        cpu = np.clip(np.random.normal(loc=40.0, scale=8.0, size=num_points), 5, 95)
        mem = np.clip(np.random.normal(loc=55.0, scale=5.0, size=num_points), 10, 95)
        latency = np.clip(np.random.normal(loc=110.0, scale=15.0, size=num_points), 30, 800)
        error = np.clip(np.random.normal(loc=0.4, scale=0.15, size=num_points), 0, 10)
        
        # KPI is transaction count, follows diurnal pattern with noise
        kpi = []
        for i in range(num_points):
            hour = timestamps[i].hour
            diurnal = 500 + 200 * np.sin(2 * np.pi * hour / 24)
            noise = np.random.normal(loc=0, scale=30)
            kpi.append(max(50.0, diurnal + noise))
            
        kpi = np.array(kpi)
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "cpu_utilization": cpu,
            "memory_utilization": mem,
            "network_latency": latency,
            "error_rate": error,
            "kpi_value": kpi
        })
        
        df.to_csv(self.baseline_path, index=False)
        self.baseline_df = df
        
        # Train models on this baseline
        self.train_all(df)
        return df

    def set_custom_baseline(self, df: pd.DataFrame) -> bool:
        """
        Updates baseline dataset and retrains models on the new baseline.
        """
        # Ensure correct columns exist
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Cannot upload baseline. Missing columns: {missing_cols}")
            return False
            
        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # Generate timestamps if missing
            start_time = datetime.now() - timedelta(hours=len(df))
            df['timestamp'] = [start_time + timedelta(hours=i) for i in range(len(df))]
            
        # Sort and save
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.to_csv(self.baseline_path, index=False)
        self.baseline_df = df
        
        logger.info(f"Custom baseline set. Retraining models on {len(df)} points...")
        return self.train_all(df)

    def train_all(self, df: pd.DataFrame) -> bool:
        """
        Trains all models on the provided dataframe.
        """
        success = True
        success &= self.anomaly_detector.train(df)
        success &= self.forecaster.train(df)
        success &= self.health_classifier.train(df)
        
        if success:
            self.save_models()
            logger.info("All ML pipeline models successfully trained and serialized.")
        else:
            logger.error("Failed to train some ML pipeline models.")
        return success

    def save_models(self):
        self.anomaly_detector.save(os.path.join(MODELS_DIR, "anomaly_detector.pkl"))
        self.forecaster.save(os.path.join(MODELS_DIR, "forecaster.pkl"))
        self.health_classifier.save(os.path.join(MODELS_DIR, "health_classifier.pkl"))

    def load_models(self):
        loaded = True
        loaded &= self.anomaly_detector.load(os.path.join(MODELS_DIR, "anomaly_detector.pkl"))
        loaded &= self.forecaster.load(os.path.join(MODELS_DIR, "forecaster.pkl"))
        loaded &= self.health_classifier.load(os.path.join(MODELS_DIR, "health_classifier.pkl"))
        if loaded:
            logger.info("All ML models successfully loaded from disk.")
        else:
            logger.info("Some ML models were not found on disk, training will be required.")
        return loaded

    def load_baseline(self):
        if os.path.exists(self.baseline_path):
            try:
                self.baseline_df = pd.read_csv(self.baseline_path)
                self.baseline_df['timestamp'] = pd.to_datetime(self.baseline_df['timestamp'])
                logger.info(f"Loaded baseline dataset from disk. Size: {len(self.baseline_df)} rows.")
            except Exception as e:
                logger.error(f"Error loading baseline file: {e}")
                self.baseline_df = None

    def run_inference(self, single_row: Dict[str, Any], historical_buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs the full inference pipeline for an incoming telemetry point.
        - single_row: dict representing the current telemetry point.
        - historical_buffer: list of dicts representing recent telemetry points (needed for forecasting lags).
        
        Returns:
        - anomaly: True/False
        - anomaly_score: Float
        - forecast_next_kpi: Float
        - system_failure_probability: Float
        """
        # Convert inputs to pandas dataframes
        df_single = pd.DataFrame([single_row])
        df_hist = pd.DataFrame(historical_buffer) if historical_buffer else df_single
        
        # Combine if necessary to get enough history for lags
        if len(df_hist) < 6:
            # Pad history if insufficient
            df_hist = pd.concat([self.baseline_df.tail(6), df_hist], ignore_index=True)
            
        # 1. Anomaly Detection
        # Isolation Forest returns -1 for anomaly, 1 for normal
        anomaly_pred = self.anomaly_detector.predict(df_single)[0]
        is_anomaly = bool(anomaly_pred == -1)
        
        # 2. KPI Forecasting (predicts next t+1)
        forecast_val = self.forecaster.predict_next(df_hist)
        
        # 3. System Failure Classification probability
        failure_prob = float(self.health_classifier.predict_probability(df_single)[0])
        
        return {
            "is_anomaly": is_anomaly,
            "forecast_next_kpi": forecast_val,
            "system_failure_probability": failure_prob
        }

    def compute_drift_report(self, production_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes Population Stability Index (PSI) and data quality of recent production data vs baseline.
        """
        if self.baseline_df is None or len(production_data) < 10:
            return {
                "system_status": "STABLE",
                "features": {},
                "prediction_drift": 0.0,
                "data_quality": {},
                "alerts": ["Awaiting production stream points to check drift (Min 10)."]
            }
            
        df_prod = pd.DataFrame(production_data)
        
        # Predict on baseline features using health classifier for baseline probability distribution
        base_preds = self.health_classifier.predict_probability(self.baseline_df)
        prod_preds = self.health_classifier.predict_probability(df_prod)
        
        return monitor_drift(
            df_baseline=self.baseline_df,
            df_production=df_prod,
            feature_cols=self.feature_cols,
            predictions_baseline=base_preds,
            predictions_production=prod_preds
        )

# Global singleton
pipeline_manager = MLPipelineManager()
