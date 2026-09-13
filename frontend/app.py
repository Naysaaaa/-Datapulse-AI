import streamlit as st
import httpx
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

# Import visualization components
from components import (
    get_custom_css,
    plot_metric_gauge,
    plot_telemetry_history,
    plot_anomaly_timeline,
    plot_forecast,
    plot_prediction_confidence
)

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="DataPulse AI - CDO Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Helper functions for API calls
def get_auth_headers():
    if "token" in st.session_state and st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

def test_backend_connection() -> bool:
    try:
        response = httpx.get(f"{os.getenv('API_URL', 'http://localhost:8000')}/", timeout=1.5)
        return response.status_code == 200
    except Exception:
        return False

# Session State Initializations
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

# --- APP LAYOUT HEADER ---
col_h_text, col_h_img = st.columns([5, 2])
with col_h_text:
    st.markdown("<h1 style='margin-top: 15px; margin-bottom: 0px; font-size: 2.8rem; background: linear-gradient(135deg, #0f172a 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>⚡ DataPulse AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.15rem; margin-top: 5px; margin-bottom: 25px;'>End-to-End Analytics, MLOps Monitoring, & Ollama Executive Narration</p>", unsafe_allow_html=True)
with col_h_img:
    st.image("frontend/datapulse_hero_graphic.png", use_container_width=True)

# Connection Check
if not test_backend_connection():
    st.error("🚨 **Cannot connect to the FastAPI Backend Service.**")
    st.warning("Please ensure the FastAPI backend is running locally on port 8000. Run the command:\n`python -m backend.app.main` or execute the launch script.")
    st.info("Tip: If you're building/running the app, click the 'Refresh Connection' button once the backend starts.")
    if st.button("🔄 Retry Connection"):
        st.rerun()
    st.stop()


# --- SIDEBAR: CONTROLS & AUTHENTICATION ---
with st.sidebar:
    st.markdown("### 🗺️ Control Navigation")
    selected_tab = st.radio(
        "Select Dashboard View:",
        [
            "📊 Executive Narrative Report", 
            "📈 Real-Time Data Streams", 
            "🧠 MLOps & Model Drift", 
            "🚨 System Alarms"
        ]
    )
    st.markdown("---")
    st.markdown("### 🔒 Platform Authentication")
    if not st.session_state.token:
        with st.form("login_form"):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="admin123")
            submitted = st.form_submit_button("Log In")
            if submitted:
                try:
                    response = httpx.post(
                        f"{API_BASE_URL}/auth/token",
                        data={"username": username, "password": password},
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        st.session_state.token = response.json()["access_token"]
                        st.session_state.username = username
                        st.success("Successfully logged in!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except Exception as e:
                    st.error(f"Login failed: {e}")
    else:
        st.markdown(f"<span class='badge badge-info'>Logged in as: {st.session_state.username}</span>", unsafe_allow_html=True)
        if st.button("Log Out"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
            
    st.markdown("---")
    
    # Ingestion Stream Control Panel
    st.markdown("### 📡 Ingestion Simulation")
    try:
        status_resp = httpx.get(f"{API_BASE_URL}/telemetry/stream/status", timeout=2.0)
        stream_status = status_resp.json()
        
        is_streaming = stream_status.get("is_streaming", False)
        drift_active = stream_status.get("drift_active", False)
        anomaly_active = stream_status.get("anomaly_active", False)
        noise_level = stream_status.get("noise_level", 0.1)
        
        # Toggle streaming (requires auth)
        if not st.session_state.token:
            st.info("💡 Logging in unlocks streaming/drift injection controls.")
            st.checkbox("Stream Active", value=is_streaming, disabled=True)
            st.checkbox("Inject Drift (MLOps Shift)", value=drift_active, disabled=True)
            st.checkbox("Force Anomalies", value=anomaly_active, disabled=True)
        else:
            # Interactive controls
            new_streaming = st.checkbox("Stream Active", value=is_streaming)
            if new_streaming != is_streaming:
                endpoint = "start" if new_streaming else "stop"
                httpx.post(f"{API_BASE_URL}/telemetry/stream/{endpoint}", headers=get_auth_headers(), timeout=5.0)
                st.rerun()
                
            new_drift = st.checkbox("Inject Drift (MLOps Shift)", value=drift_active)
            new_anomaly = st.checkbox("Force Anomalies", value=anomaly_active)
            new_noise = st.slider("Signal Noise Ratio", min_value=0.0, max_value=1.0, value=noise_level, step=0.05)
            
            if new_drift != drift_active or new_anomaly != anomaly_active or new_noise != noise_level:
                httpx.post(
                    f"{API_BASE_URL}/telemetry/stream/config",
                    json={"drift": new_drift, "anomaly": new_anomaly, "noise": new_noise},
                    headers=get_auth_headers(),
                    timeout=5.0
                )
                st.rerun()
    except Exception as e:
        st.error(f"Failed to fetch streaming status: {e}")

    st.markdown("---")

    # CSV Upload Baseline retrain
    st.markdown("### 📥 Baseline Upload (Retrain)")
    if not st.session_state.token:
        st.info("💡 Logging in unlocks custom CSV baseline training.")
    else:
        uploaded_file = st.file_uploader("Upload Telemetry CSV", type=["csv"])
        if uploaded_file is not None:
            if st.button("🚀 Train Models on Baseline"):
                with st.spinner("Processing data & retraining models..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        response = httpx.post(
                            f"{API_BASE_URL}/telemetry/upload",
                            files=files,
                            headers=get_auth_headers(),
                            timeout=30.0
                        )
                        if response.status_code == 200:
                            st.success("Models retrained! Baseline reset successfully.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Failed to upload CSV: {e}")

    st.markdown("---")
    
    # Auto refresh configuration
    st.markdown("### ⏱️ Settings")
    auto_refresh = st.checkbox("Auto-Refresh Dashboard (5s)", value=False)
    if st.button("🔄 Force Refresh"):
        st.rerun()


# --- DATABASE QUERIES & DATA SYNCHRONIZATION ---
try:
    telemetry_resp = httpx.get(f"{API_BASE_URL}/telemetry?limit=80", timeout=3.0)
    df_telemetry = pd.DataFrame(telemetry_resp.json())
    if not df_telemetry.empty:
        df_telemetry['timestamp'] = pd.to_datetime(df_telemetry['timestamp'])
        df_telemetry = df_telemetry.sort_values('timestamp')
except Exception as e:
    st.error(f"Error fetching telemetry data: {e}")
    df_telemetry = pd.DataFrame()

try:
    drift_resp = httpx.get(f"{API_BASE_URL}/drift/latest", timeout=3.0)
    latest_drift = drift_resp.json() if drift_resp.status_code == 200 else None
except Exception:
    latest_drift = None

try:
    alert_resp = httpx.get(f"{API_BASE_URL}/alerts?unresolved_only=true", timeout=3.0)
    active_alerts = alert_resp.json()
except Exception:
    active_alerts = []

try:
    narrative_resp = httpx.get(f"{API_BASE_URL}/narrative/latest", timeout=3.0)
    latest_narrative = narrative_resp.json() if narrative_resp.status_code == 200 else None
except Exception:
    latest_narrative = None


# --- HEADER STATUS INFO BAR ---
if not df_telemetry.empty:
    latest_point = df_telemetry.iloc[-1]
    
    # Compute active systems status
    sys_status_str = latest_drift.get("system_status", "STABLE") if latest_drift else "STABLE"
    status_badge_class = "badge-stable"
    if sys_status_str == "DEGRADED":
        status_badge_class = "badge-degraded"
    elif sys_status_str == "DRIFTING":
        status_badge_class = "badge-drifting"
        
    narrator_str = "Simulated Fallback"
    narrator_badge = "badge-degraded"
    if latest_narrative and not latest_narrative.get("is_mocked", True):
        narrator_str = "Ollama Local (Mistral)"
        narrator_badge = "badge-stable"
        
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f"**MLOps Status:** <span class='badge {status_badge_class}'>{sys_status_str}</span>", unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"**Active Alerts:** <span class='badge {'badge-drifting' if active_alerts else 'badge-stable'}'>{len(active_alerts)} Alerts</span>", unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f"**AI Narrative Agent:** <span class='badge {narrator_badge}'>{narrator_str}</span>", unsafe_allow_html=True)
    with col_stat4:
        st.markdown(f"**Engine Active:** <span class='badge {'badge-stable' if is_streaming else 'badge-degraded'}'>{'ACTIVE' if is_streaming else 'INACTIVE'}</span>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ==================== MAIN VIEW RENDERING (USER-CONTROLLED) ====================
    if selected_tab == "📊 Executive Narrative Report":
        col_exec_l, col_exec_r = st.columns([2, 1])
        
        with col_exec_l:
            st.markdown("### Executive Analyst Insights")
            
            # Narrative panel
            if latest_narrative:
                narrative_time = pd.to_datetime(latest_narrative.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"Last updated: {narrative_time} UTC")
                
                # Custom CSS styled container
                st.markdown(f"""
                <div class='narrative-container'>
                    {latest_narrative.get("narrative", "")}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No executive narrative has been synthesized yet. Click the 'Synthesize Narrative' button to query Ollama.")

            # Trigger narrative generation
            col_actions_1, col_actions_2 = st.columns(2)
            with col_actions_1:
                if st.button("🧠 Synthesize Executive Narrative (Ollama/Fallback)", use_container_width=True):
                    with st.spinner("Ollama analyzing data trends and writing narrative..."):
                        try:
                            narr_gen_resp = httpx.post(f"{API_BASE_URL}/narrative/generate", timeout=25.0)
                            if narr_gen_resp.status_code == 200:
                                st.success("Narrative generated successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to generate narrative.")
                        except Exception as e:
                            st.error(f"Error contacting Ollama narrative service: {e}")
            with col_actions_2:
                # PDF report download trigger
                pdf_url = f"{API_BASE_URL}/reports/pdf"
                try:
                    pdf_data_resp = httpx.get(pdf_url, timeout=10.0)
                    if pdf_data_resp.status_code == 200:
                        st.download_button(
                            label="📄 Export Executive Report (PDF)",
                            data=pdf_data_resp.content,
                            file_name=f"DataPulse_Executive_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.button("📄 PDF Export Unavailable", disabled=True, use_container_width=True)
                except Exception:
                    st.button("📄 PDF Export Unreachable", disabled=True, use_container_width=True)
                    
        with col_exec_r:
            st.markdown("### Latest KPI State")
            
            # Telemetry Overview Grid
            st.markdown(f"""
            <div class='premium-card'>
                <div class='metric-label'>Transaction Count (KPI Target)</div>
                <div class='metric-value'>{latest_point.get('kpi_value', 0.0):.1f}</div>
                <div class='metric-label' style='margin-top:10px;'>XGBoost Forecast (t+1)</div>
                <div class='metric-value' style='color:#F59E0B;'>{f"{latest_point.get('forecast_value'):.1f}" if latest_point.get('forecast_value') is not None and not pd.isna(latest_point.get('forecast_value')) else 'N/A'}</div>
            </div>
            <div class='premium-card'>
                <div class='metric-label'>System Failure Prob</div>
                <div class='metric-value' style='color:{'#EF4444' if latest_point.get('failure_probability', 0) > 0.5 else '#10B981'};'>
                    {latest_point.get('failure_probability', 0.0)*100:.1f}%
                </div>
            </div>
            <div class='premium-card'>
                <div class='metric-label'>Ingestion Database Rows</div>
                <div class='metric-value'>{len(df_telemetry)} / 80 Cached</div>
            </div>
            """, unsafe_allow_html=True)


    # Define Streamlit tabs
    tab_telemetry, tab_mlops, tab_alerts = st.tabs(["Telemetry", "MLOps", "System Alerts"]) 
    # ==================== TAB 2: DATA STREAMS ====================

    with tab_telemetry:
        st.markdown("### Telemetry Stream Insights")
        
        # Grid of current numeric values
        col_grid1, col_grid2, col_grid3, col_grid4 = st.columns(4)
        with col_grid1:
            st.metric("CPU Utilization", f"{latest_point.cpu_utilization:.1f}%")
        with col_grid2:
            st.metric("Memory Utilization", f"{latest_point.memory_utilization:.1f}%")
        with col_grid3:
            st.metric("Network Latency", f"{latest_point.network_latency:.1f} ms")
        with col_grid4:
            st.metric("Telemetry Error Rate", f"{latest_point.error_rate:.2f}%")
            
        st.markdown("---")
        
        # Charts Grid
        col_chart_l, col_chart_r = st.columns(2)
        with col_chart_l:
            st.plotly_chart(plot_telemetry_history(df_telemetry), use_container_width=True)
            st.plotly_chart(plot_forecast(df_telemetry), use_container_width=True)
        with col_chart_r:
            st.plotly_chart(plot_anomaly_timeline(df_telemetry), use_container_width=True)
            st.plotly_chart(plot_prediction_confidence(df_telemetry), use_container_width=True)


    # ==================== TAB 3: MLOps MONITORING ====================
    with tab_mlops:
        st.markdown("### MLOps Model Performance & Drift Auditing")
        
        if latest_drift:
            # Top row showing Prediction Drift Gauge and Data Quality Report
            col_drift_l, col_drift_r = st.columns(2)
            with col_drift_l:
                pred_drift_val = latest_drift.get("prediction_drift", 0.0)
                st.plotly_chart(plot_metric_gauge(pred_drift_val, "System Health Model Prediction Drift (PSI)"), use_container_width=True)
                
                # Check status
                if pred_drift_val >= 0.25:
                    st.error("🚨 **Critical Prediction Drift:** The classification output distributions has drifted significantly. Model retraining recommended.")
                elif pred_drift_val >= 0.10:
                    st.warning("⚠️ **Moderate Prediction Drift:** Prediction characteristics are changing. Monitor performance.")
                else:
                    st.success("✅ **Classification model remains stable.**")
                    
            with col_drift_r:
                st.markdown("#### Baseline Calibration vs Production Stream Data Quality")
                dq_dict = latest_drift.get("data_quality_report", {})
                
                if dq_dict:
                    dq_rows = []
                    for col, metrics in dq_dict.items():
                        dq_rows.append({
                            "Metric Column": col,
                            "Missing Ratio": f"{metrics.get('missing_ratio', 0.0)*100:.2f}%",
                            "Outlier Ratio": f"{metrics.get('outlier_ratio', 0.0)*100:.2f}%",
                            "Production Mean": f"{metrics.get('mean', 0.0):.2f}",
                            "Production StdDev": f"{metrics.get('std', 0.0):.2f}"
                        })
                    st.table(pd.DataFrame(dq_rows))
                else:
                    st.info("No data quality checks recorded.")
                    
            st.markdown("---")
            
            # Feature Drift Gauges
            st.markdown("#### Individual Feature Population Stability Index (PSI)")
            feat_psi_dict = latest_drift.get("feature_psi", {})
            
            if feat_psi_dict:
                # Plot in 3 columns
                cols = st.columns(3)
                for index, (feature_name, psi_val) in enumerate(feat_psi_dict.items()):
                    with cols[index % 3]:
                        st.plotly_chart(plot_metric_gauge(psi_val, f"{feature_name} (PSI)"), use_container_width=True)
            else:
                st.info("No feature PSI metrics recorded yet.")
        else:
            st.info("Awaiting MLOps background calibration reports. Ensure the stream is active and has generated at least 10 events.")


    # ==================== TAB 4: SYSTEM ALARMS ====================
    with tab_alerts:
        st.markdown("### Real-time Platform Alerts & Model Faults")
        
        if active_alerts:
            # Alert list
            for alert in active_alerts:
                severity = alert.get("severity", "WARNING")
                alert_id = alert.get("id")
                alert_type = alert.get("type", "health").upper()
                msg = alert.get("message", "")
                timestamp = pd.to_datetime(alert.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
                
                # Display alert
                col_alert_text, col_alert_btn = st.columns([4, 1])
                with col_alert_text:
                    if severity == "CRITICAL":
                        st.error(f"🔴 **[{alert_type}] {msg}** (Logged: {timestamp} UTC)")
                    else:
                        st.warning(f"⚠️ **[{alert_type}] {msg}** (Logged: {timestamp} UTC)")
                        
                with col_alert_btn:
                    if not st.session_state.token:
                        st.button("Resolve", key=f"btn_dis_{alert_id}", disabled=True)
                    else:
                        if st.button("Resolve ✔️", key=f"btn_res_{alert_id}"):
                            try:
                                resolve_resp = httpx.post(
                                    f"{API_BASE_URL}/alerts/{alert_id}/resolve",
                                    json={"is_resolved": True},
                                    headers=get_auth_headers(),
                                    timeout=5.0
                                )
                                if resolve_resp.status_code == 200:
                                    st.success(f"Alert {alert_id} resolved!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to resolve alert.")
                            except Exception as e:
                                st.error(f"Error: {e}")
        else:
            st.success("🎉 **Zero active alerts triggered.** All systems operating within nominal baseline bounds.")
else:
    st.info("Initializing database and streaming pipelines. Refreshing in a moment...")
    time.sleep(2)
    st.rerun()

# Auto Refresh loop
if auto_refresh:
    time.sleep(5)
    st.rerun()
