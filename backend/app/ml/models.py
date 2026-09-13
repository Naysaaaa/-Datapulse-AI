import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
import pickle
import os

try:
    import xgboost as xgb
except Exception as e:
    logging.getLogger(__name__).warning(f"XGBoost library import/load failed: {e}. A fallback regressor will be used.")
    xgb = None

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Isolation Forest anomaly detector for streaming telemetry data.
    Detects unusual patterns across CPU, memory, latency, error rate, and KPI.
    """
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.feature_cols = ['cpu_utilization', 'memory_utilization', 'network_latency', 'error_rate', 'kpi_value']
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        if len(df) < 10:
            logger.warning("Not enough data to train Isolation Forest. Minimum 10 rows required.")
            return False
        
        try:
            X = df[self.feature_cols].ffill().fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
            self.is_trained = True
            logger.info("Anomaly Detector (Isolation Forest) trained successfully.")
            return True
        except Exception as e:
            logger.error(f"Error training Anomaly Detector: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns -1 for anomalies, 1 for normal data.
        If not trained, returns 1 (normal) for all points.
        """
        if not self.is_trained or len(df) == 0:
            return np.ones(len(df))
        try:
            X = df[self.feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"Error predicting anomalies: {e}")
            return np.ones(len(df))

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump((self.model, self.scaler, self.is_trained), f)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.model, self.scaler, self.is_trained = pickle.load(f)
            return True
        return False


class KPIForecaster:
    """
    XGBoost Regressor for KPI forecasting.
    Predicts the next kpi_value (t+1) using historical lags and rolling window features.
    """
    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        self.model = None
        if xgb is not None:
            try:
                self.model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
            except Exception as e:
                logger.warning(f"Failed to instantiate XGBRegressor: {e}. Fallback will be used.")
        
        # Fallback in case xgboost has issues (e.g. library load errors on some platforms)
        self.fallback_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def create_features(self, df: pd.DataFrame, is_training=True) -> tuple:
        """
        Creates lag and rolling window features from the kpi_value column.
        """
        df_feats = df.copy().sort_values('timestamp')
        
        # Lags
        for i in range(1, self.n_lags + 1):
            df_feats[f'kpi_lag_{i}'] = df_feats['kpi_value'].shift(i)
            
        # Rolling stats
        df_feats['kpi_rolling_mean'] = df_feats['kpi_value'].shift(1).rolling(window=3, min_periods=1).mean()
        df_feats['kpi_rolling_std'] = df_feats['kpi_value'].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
        
        feature_cols = [f'kpi_lag_{i}' for i in range(1, self.n_lags + 1)] + ['kpi_rolling_mean', 'kpi_rolling_std']
        
        # Target is the next step's KPI value (shifted by -1)
        if is_training:
            df_feats['target'] = df_feats['kpi_value'].shift(-1)
            # Drop NaN rows due to shifting and lag creation
            df_clean = df_feats.dropna()
            if len(df_clean) == 0:
                return pd.DataFrame(), pd.Series(), feature_cols
            return df_clean[feature_cols], df_clean['target'], feature_cols
        else:
            # For inference on the last point, we don't drop NaNs since the target is unknown
            # We return features for the very last row
            last_row = df_feats.tail(1)
            return last_row[feature_cols], None, feature_cols

    def train(self, df: pd.DataFrame):
        if len(df) < self.n_lags + 10:
            logger.warning("Not enough data to train KPI Forecaster.")
            return False
        
        try:
            X, y, _ = self.create_features(df, is_training=True)
            if len(X) < 5:
                return False
                
            X_scaled = self.scaler.fit_transform(X)
            
            if self.model is not None:
                try:
                    self.model.fit(X_scaled, y)
                    self.fallback_model = None
                except Exception as xgb_err:
                    logger.warning(f"XGBoost training failed ({xgb_err}), falling back to Linear Regression.")
                    from sklearn.linear_model import LinearRegression
                    self.fallback_model = LinearRegression()
                    self.fallback_model.fit(X_scaled, y)
            else:
                from sklearn.linear_model import LinearRegression
                self.fallback_model = LinearRegression()
                self.fallback_model.fit(X_scaled, y)
                
            self.is_trained = True
            logger.info("KPI Forecaster trained successfully.")
            return True
        except Exception as e:
            logger.error(f"Error training KPI Forecaster: {e}")
            return False

    def predict_next(self, df: pd.DataFrame) -> float:
        """
        Predicts the kpi_value for the next time step.
        """
        if not self.is_trained or len(df) < self.n_lags:
            # Fallback prediction is just the last known value
            return float(df['kpi_value'].iloc[-1]) if len(df) > 0 else 0.0
            
        try:
            X_infer, _, _ = self.create_features(df, is_training=False)
            if X_infer.isnull().values.any():
                X_infer = X_infer.ffill().fillna(0)
            X_scaled = self.scaler.transform(X_infer)
            
            if self.fallback_model is not None:
                pred = self.fallback_model.predict(X_scaled)
            else:
                pred = self.model.predict(X_scaled)
                
            return float(pred[0])
        except Exception as e:
            logger.error(f"Error forecasting KPI: {e}")
            return float(df['kpi_value'].iloc[-1]) if len(df) > 0 else 0.0

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump((self.model, self.fallback_model, self.scaler, self.is_trained), f)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.model, self.fallback_model, self.scaler, self.is_trained = pickle.load(f)
            return True
        return False


class HealthClassifier:
    """
    Logistic Regression Classifier to predict the probability of system failure.
    Tracks prediction confidence over time.
    """
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.feature_cols = ['cpu_utilization', 'memory_utilization', 'network_latency', 'error_rate']
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """
        Trains the classifier. Ground truth 'system_failure' label is 1 if error_rate > 5 or 
        (cpu_utilization > 90 and network_latency > 350), else 0 (synthesized for training/baseline).
        """
        if len(df) < 15:
            logger.warning("Not enough data to train Health Classifier.")
            return False
            
        try:
            X = df[self.feature_cols].fillna(0)
            # Create synthetic failure label based on typical alert criteria
            y = ((df['error_rate'] > 5.0) | ((df['cpu_utilization'] > 90.0) & (df['network_latency'] > 350.0))).astype(int)
            
            # Check if we have both classes
            if len(np.unique(y)) < 2:
                # Add one forced negative and one forced positive row to make it trainable
                X = pd.concat([X, pd.DataFrame([[10.0, 15.0, 50.0, 0.1], [95.0, 95.0, 500.0, 8.0]], columns=self.feature_cols)], ignore_index=True)
                y = pd.concat([y, pd.Series([0, 1])], ignore_index=True)
                
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logger.info("Health Classifier (Logistic Regression) trained successfully.")
            return True
        except Exception as e:
            logger.error(f"Error training Health Classifier: {e}")
            return False

    def predict_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns the probability of system failure [0.0, 1.0].
        If not trained, returns standard rule-based heuristic.
        """
        if not self.is_trained or len(df) == 0:
            # Fallback heuristic
            probs = []
            for _, row in df.iterrows():
                base_prob = 0.05
                if row.get('cpu_utilization', 0) > 90: base_prob += 0.3
                if row.get('error_rate', 0) > 5.0: base_prob += 0.5
                if row.get('network_latency', 0) > 350: base_prob += 0.15
                probs.append(min(base_prob, 0.99))
            return np.array(probs)
            
        try:
            X = df[self.feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)
            # Get probability for class 1 (failure)
            return self.model.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            logger.error(f"Error predicting system health: {e}")
            return np.zeros(len(df))

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump((self.model, self.scaler, self.is_trained), f)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.model, self.scaler, self.is_trained = pickle.load(f)
            return True
        return False
