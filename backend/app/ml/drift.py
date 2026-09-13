import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 5) -> float:
    """
    Calculate the Population Stability Index (PSI) between expected (baseline) and actual (production) data.
    
    PSI Formula:
    PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
    
    Interpretations:
    - PSI < 0.1: No significant change (stable)
    - 0.1 <= PSI < 0.25: Moderate change (requires monitoring)
    - PSI >= 0.25: Significant change (requires retraining/alert)
    """
    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # If the column has only one unique value, PSI calculation is trivial (0 if same, otherwise infinite)
    if len(np.unique(expected)) <= 1:
        if np.array_equal(np.unique(expected), np.unique(actual)):
            return 0.0
        return 1.0  # complete drift

    # Create buckets based on quantiles of the expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    
    # Adjust boundaries slightly to avoid bin issues for equal bounds
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5
    
    # In case we have identical quantiles (e.g. highly skewed distribution), make boundaries unique
    buckets = np.unique(buckets)
    if len(buckets) < 2:
        # Fall back to equal-width bins if quantiles collapsed
        min_val = min(np.min(expected), np.min(actual))
        max_val = max(np.max(expected), np.max(actual))
        if min_val == max_val:
            return 0.0
        buckets = np.linspace(min_val - 1e-5, max_val + 1e-5, num_buckets + 1)
        
    # Count frequencies in each bin
    expected_counts, _ = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=buckets)
    
    # Convert to percentages
    expected_pcts = expected_counts / len(expected)
    actual_pcts = actual_counts / len(actual)
    
    # Handle zero percentages with small epsilon to avoid division by zero / log of zero issues
    eps = 1e-4
    expected_pcts = np.where(expected_pcts == 0, eps, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, eps, actual_pcts)
    
    # Normalize again after adding epsilon
    expected_pcts = expected_pcts / np.sum(expected_pcts)
    actual_pcts = actual_pcts / np.sum(actual_pcts)
    
    # Calculate PSI
    psi_value = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    
    return float(psi_value)


def calculate_data_quality(df_baseline: pd.DataFrame, df_production: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Computes data quality metrics of production data compared to baseline:
    - Missing value percentages per column
    - Out of bounds percentage (outside [min - tol, max + tol] of baseline)
    """
    quality_report = {}
    
    for col in feature_cols:
        if col not in df_production.columns:
            continue
            
        prod_col = df_production[col]
        base_col = df_baseline[col]
        
        # Missing values
        missing_count = prod_col.isnull().sum()
        missing_ratio = float(missing_count / len(prod_col)) if len(prod_col) > 0 else 0.0
        
        # Out-of-bounds metrics (baseline min/max with standard deviation buffers)
        base_mean = base_col.mean()
        base_std = base_col.std()
        
        # Outlier defined as outside mean +- 3*std
        lower_bound = base_mean - 3 * base_std if not pd.isna(base_std) else base_col.min()
        upper_bound = base_mean + 3 * base_std if not pd.isna(base_std) else base_col.max()
        
        outliers_count = ((prod_col < lower_bound) | (prod_col > upper_bound)).sum()
        outliers_ratio = float(outliers_count / len(prod_col)) if len(prod_col) > 0 else 0.0
        
        quality_report[col] = {
            "missing_ratio": missing_ratio,
            "outlier_ratio": outliers_ratio,
            "mean": float(prod_col.mean()) if not pd.isna(prod_col.mean()) else 0.0,
            "std": float(prod_col.std()) if not pd.isna(prod_col.std()) else 0.0,
        }
        
    return quality_report


def monitor_drift(df_baseline: pd.DataFrame, df_production: pd.DataFrame, feature_cols: List[str], predictions_baseline: np.ndarray, predictions_production: np.ndarray) -> Dict[str, Any]:
    """
    Compiles full drift metrics report.
    Tracks feature drift (PSI), model prediction confidence drift, and data quality.
    """
    report = {
        "features": {},
        "prediction_drift": 0.0,
        "data_quality": {},
        "system_status": "STABLE", # STABLE, DEGRADED, DRIFTING
        "alerts": []
    }
    
    # 1. Feature Drift (PSI)
    max_feature_psi = 0.0
    for col in feature_cols:
        if col in df_baseline.columns and col in df_production.columns:
            psi = calculate_psi(df_baseline[col].values, df_production[col].values)
            report["features"][col] = {
                "psi": psi,
                "status": "STABLE" if psi < 0.1 else ("WARNING" if psi < 0.25 else "DRIFTING")
            }
            if psi > max_feature_psi:
                max_feature_psi = psi
                
            if psi >= 0.25:
                report["alerts"].append(f"Significant drift detected in feature '{col}' (PSI: {psi:.3f})")
            elif psi >= 0.10:
                report["alerts"].append(f"Moderate drift detected in feature '{col}' (PSI: {psi:.3f})")
                
    # 2. Prediction Drift (PSI of prediction distributions/probabilities)
    if len(predictions_baseline) > 0 and len(predictions_production) > 0:
        pred_psi = calculate_psi(predictions_baseline, predictions_production)
        report["prediction_drift"] = pred_psi
        if pred_psi >= 0.25:
            report["alerts"].append(f"Significant drift in model predictions detected (PSI: {pred_psi:.3f})")
        elif pred_psi >= 0.10:
            report["alerts"].append(f"Moderate drift in model predictions detected (PSI: {pred_psi:.3f})")
            
    # 3. Data Quality
    report["data_quality"] = calculate_data_quality(df_baseline, df_production, feature_cols)
    
    # 4. Global status evaluation
    # If any feature or the predictions show significant drift, or missing values are high (>10%)
    high_missing = any(q["missing_ratio"] > 0.10 for q in report["data_quality"].values())
    
    if max_feature_psi >= 0.25 or report["prediction_drift"] >= 0.25:
        report["system_status"] = "DRIFTING"
    elif max_feature_psi >= 0.10 or report["prediction_drift"] >= 0.10 or high_missing:
        report["system_status"] = "DEGRADED"
        
    if high_missing:
        report["alerts"].append("High ratio of missing values (>10%) detected in production stream.")
        
    return report
