import httpx
import logging
import json
from typing import Dict, Any, List
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class OllamaNarrativeService:
    """
    Connects to local Ollama API to narrate model findings, alerts, and drift metrics.
    Falls back to a structured template-driven narrative generator if Ollama is unreachable.
    """
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    async def generate_narrative(self, report_data: Dict[str, Any]) -> tuple[str, bool]:
        """
        Generates the executive summary.
        Returns: (narrative_text, is_mocked)
        """
        system_status = report_data.get("system_status", "UNKNOWN")
        feature_psi = report_data.get("feature_psi", {})
        prediction_drift = report_data.get("prediction_drift", 0.0)
        anomaly_count = report_data.get("anomaly_count", 0)
        total_points = report_data.get("total_points", 0)
        latest_telemetry = report_data.get("latest_telemetry", {})
        forecast_value = report_data.get("forecast_value", 0.0)
        active_alerts = report_data.get("active_alerts", [])
        
        # Build prompt
        prompt = self._build_prompt(
            system_status, feature_psi, prediction_drift, 
            anomaly_count, total_points, latest_telemetry, 
            forecast_value, active_alerts
        )
        
        try:
            # Query local Ollama service
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": "You are DataPulse Executive Intelligence, a senior Chief Data Officer reporting to executive leadership. Summarize technical telemetry, anomaly reports, and model drift statistics into a concise, professional, business-friendly executive narrative. Focus on trends, risks, and recommendations. Avoid conversational fillers, introductions, or pleasantries. Output direct markdown text.",
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    narrative = result.get("response", "").strip()
                    if narrative:
                        logger.info("Executive narrative successfully generated using local Ollama.")
                        return narrative, False
                    
                logger.warning(f"Ollama returned empty response or non-200 code: {response.status_code}. Using fallback.")
        except (httpx.ConnectError, httpx.TimeoutException) as conn_err:
            logger.warning(f"Ollama is unreachable at {self.base_url} ({conn_err}). Falling back to rule-based generator.")
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}. Falling back.")
            
        # Fallback to simulated intelligence narrator
        narrative = self._generate_simulated_narrative(
            system_status, feature_psi, prediction_drift, 
            anomaly_count, total_points, latest_telemetry, 
            forecast_value, active_alerts
        )
        return narrative, True

    def _build_prompt(self, system_status: str, feature_psi: Dict[str, float], prediction_drift: float, 
                      anomaly_count: int, total_points: int, latest_telemetry: Dict[str, Any], 
                      forecast_value: float, active_alerts: List[str]) -> str:
        
        psi_details = ", ".join([f"{k}: {v:.3f}" for k, v in feature_psi.items()])
        alerts_list = "\n".join([f"- {alert}" for alert in active_alerts]) if active_alerts else "- No active alerts"
        
        prompt = f"""
        Analyze the following telemetry and MLOps status report:
        
        1. Operational Health Status: {system_status}
        2. Ingestion Summary: Ingested {total_points} events. Identified {anomaly_count} anomalies using Isolation Forest.
        3. Latest Telemetry:
           - CPU: {latest_telemetry.get('cpu_utilization', 0.0):.1f}%
           - Memory: {latest_telemetry.get('memory_utilization', 0.0):.1f}%
           - Latency: {latest_telemetry.get('network_latency', 0.0):.1f}ms
           - Error Rate: {latest_telemetry.get('error_rate', 0.0):.2f}%
           - Current KPI: {latest_telemetry.get('kpi_value', 0.0):.1f}
        4. KPI Forecast: XGBoost forecasts the next KPI value to be {forecast_value:.2f}.
        5. MLOps Drift Analytics:
           - Feature Drift (PSI): {psi_details}
           - Prediction Drift (PSI): {prediction_drift:.3f}
        6. Operational Alerts:
        {alerts_list}
        
        Provide a 3-paragraph executive-ready report:
        Paragraph 1: Overall System Health and Operational Summary. Focus on recent load, failures, and latency.
        Paragraph 2: Machine Learning & MLOps performance. Critique model drift (PSI) and explain if the models require retraining or if the current forecasts are reliable.
        Paragraph 3: Business Recommendations. Give 2-3 specific, actionable recommendations for system administrators and business managers based on the findings.
        """
        return prompt

    def _generate_simulated_narrative(self, system_status: str, feature_psi: Dict[str, float], prediction_drift: float, 
                                      anomaly_count: int, total_points: int, latest_telemetry: Dict[str, Any], 
                                      forecast_value: float, active_alerts: List[str]) -> str:
        """
        A rule-based generator that crafts a highly professional, CDO-level narrative.
        This ensures the application is completely interactive and functional even without Ollama.
        """
        # Determine health summary
        current_kpi = latest_telemetry.get('kpi_value', 0.0)
        current_error = latest_telemetry.get('error_rate', 0.0)
        current_cpu = latest_telemetry.get('cpu_utilization', 0.0)
        current_lat = latest_telemetry.get('network_latency', 0.0)
        
        kpi_trend = "increasing" if forecast_value > current_kpi else "decreasing"
        kpi_diff_pct = abs(forecast_value - current_kpi) / (current_kpi + 1e-5) * 100
        
        # 1. Operational summary
        if system_status == "STABLE":
            health_para = (
                f"### 1. Executive Summary & Operational Status\n"
                f"DataPulse AI has evaluated recent system telemetry and confirms that the core infrastructure is currently **stable and operating within nominal parameters**. "
                f"Over the current monitoring window, {total_points} events were ingested, with only {anomaly_count} minor anomalies flagged by the Isolation Forest pipeline. "
                f"Current CPU load sits at {current_cpu:.1f}% with latency averaging {current_lat:.1f}ms and an error rate of {current_error:.2f}%. "
                f"These statistics indicate a healthy system supporting business operations without immediate resource bottlenecks."
            )
        elif system_status == "DEGRADED":
            health_para = (
                f"### 1. Executive Summary & Operational Status\n"
                f"The platform has transitioned to a **DEGRADED status** due to emerging performance bottlenecks and minor structural changes in the incoming stream. "
                f"While total transaction volume remains stable (KPI: {current_kpi:.1f}), we are experiencing elevated resource usage. CPU utilization is at {current_cpu:.1f}%, "
                f"and system latency has reached {current_lat:.1f}ms. Ingestion pipelines captured {anomaly_count} anomalies. While operations are functional, "
                f"the system is showing early indicators of stress that could impact downstream client transactions if left unaddressed."
            )
        else: # DRIFTING
            health_para = (
                f"### 1. Executive Summary & Operational Status\n"
                f"The system has been flagged as **DRIFTING**. There is a significant divergence between the production telemetry stream and the established baseline. "
                f"We are observing anomalies ({anomaly_count} instances detected) coupled with irregular error spikes (current error rate: {current_error:.2f}%). "
                f"Average CPU utilization has escalated to {current_cpu:.1f}% and latency has spiked to {current_lat:.1f}ms. "
                f"Operational stability is threatened, indicating either a shift in client usage profiles or a serious performance regression."
            )

        # 2. MLOps summary
        max_psi_feature = max(feature_psi, key=feature_psi.get) if feature_psi else "None"
        max_psi_val = feature_psi[max_psi_feature] if feature_psi else 0.0
        
        if max_psi_val < 0.10:
            ml_para = (
                f"### 2. MLOps & Machine Learning Performance\n"
                f"The machine learning infrastructure exhibits high model stability. The **Population Stability Index (PSI) remains low** across all key features, "
                f"indicating that the training baseline reflects active production data. The maximum feature drift is observed in `{max_psi_feature}` (PSI: {max_psi_val:.3f}), "
                f"which is well below the drift threshold of 0.10. "
                f"Model prediction drift (PSI: {prediction_drift:.3f}) is negligible, confirming that the Logistic Regression classification boundary and the Isolation Forest boundaries are highly reliable. "
                f"The XGBoost forecasting pipeline predicts the next step KPI value at **{forecast_value:.2f}** (a change of {kpi_diff_pct:.1f}% from the current {current_kpi:.1f}). "
                f"Given the statistical stability, model predictions can be trusted for operational planning."
            )
        elif max_psi_val < 0.25:
            ml_para = (
                f"### 2. MLOps & Machine Learning Performance\n"
                f"The model monitoring layer has detected **moderate feature drift** in the production pipeline, particularly in the `{max_psi_feature}` feature (PSI: {max_psi_val:.3f}). "
                f"This indicates a noticeable, though not critical, shift in the distribution of incoming metrics compared to the baseline configuration. "
                f"Prediction drift is currently at {prediction_drift:.3f}. The XGBoost KPI forecaster predicts a next-step KPI of **{forecast_value:.2f}**. "
                f"While the models do not require immediate manual retraining, prediction confidence is slightly reduced. "
                f"We recommend scheduling an automated model retraining cycle within the next maintenance window to align with the new data distribution."
            )
        else: # DRIFTING
            ml_para = (
                f"### 2. MLOps & Machine Learning Performance\n"
                f"**CRITICAL MLOps ALERT:** The monitoring layer has identified **significant model drift**. "
                f"The Population Stability Index for `{max_psi_feature}` has breached the critical threshold at **{max_psi_val:.3f}** (PSI >= 0.25). "
                f"Additionally, the prediction drift index has spiked to {prediction_drift:.3f}. This indicates a fundamental divergence in the underlying data generating process "
                f"compared to the training baseline (likely due to a new system state, traffic surge, or data quality degradation). "
                f"The current XGBoost forecast of **{forecast_value:.2f}** and the system health classification probabilities may be compromised due to covariate shift. "
                f"**Immediate model retraining is highly recommended** to update the model boundaries."
            )

        # 3. Recommendations
        rec1 = "Ensure system monitors remain active and baseline configurations are reviewed monthly."
        rec2 = "Continue tracking the XGBoost forecasting error to validate forecast confidence."
        
        if system_status != "STABLE":
            rec1 = "Investigate the root cause of recent anomaly clusters. Specifically inspect system logs matching timestamps of high latency."
        if max_psi_val >= 0.10:
            rec2 = f"Initiate training pipeline using the latest production window to generate an updated model file (.pkl) for `{max_psi_feature}`."
            
        rec_para = (
            f"### 3. Actionable Recommendations & Interventions\n"
            f"- **MLOps Action:** {rec2}\n"
            f"- **Infrastructure Action:** {rec1}\n"
            f"- **Business Action:** Use the forecasted KPI ({forecast_value:.2f}) to adjust capacity allocations and review user SLAs in relation to current latency ({current_lat:.1f}ms)."
        )

        full_narrative = f"{health_para}\n\n{ml_para}\n\n{rec_para}"
        return full_narrative
