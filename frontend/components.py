import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def get_custom_css() -> str:
    """Returns custom CSS to make Streamlit dashboard look premium and polished."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global layout */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #FFFAF0, #FFF5E5);
        color: #1F2937;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #111827;
    }
    
    /* Premium card with vibrant gradient */
    .premium-card {
        background: linear-gradient(135deg, #FFFAF0, #FFE4E1);
        border: 1px solid #FCD34D;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px -2px rgba(255, 99, 71, 0.12);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 16px;
    }
    .premium-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px -5px rgba(255, 99, 71, 0.2);
    }
    
    /* Badge variants with bright pastel hues */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-stable { background-color: #D1FAE5; color: #065F46; }
    .badge-degraded { background-color: #FEF3C7; color: #92400E; }
    .badge-drifting { background-color: #FEE2E2; color: #991B1B; }
    .badge-info { background-color: #DBEAFE; color: #1E40AF; }
    
    /* Buttons with bright accent */
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B, #FFEA00);
        color: #1F2937;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: transform 0.1s ease;
    }
    .stButton>button:hover {
        transform: scale(1.03);
    }
    
    </style>
    """

def plot_metric_gauge(val: float, title: str, system_status: str = "STABLE") -> go.Figure:
    """Creates a sleek, modern half-gauge for model drift/PSI indicators."""
    max_val = max(1.0, val * 1.5)
    
    # Determine indicator color based on value
    if val < 0.10:
        indicator_color = "#10B981" # Emerald Green
    elif val < 0.25:
        indicator_color = "#F59E0B" # Amber Yellow
    else:
        indicator_color = "#EF4444" # Rose Red
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>{title}</b>", 'font': {'size': 14, 'family': 'Outfit'}},
        gauge = {
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': indicator_color},
            'bgcolor': "#F1F5F9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 0.10], 'color': '#E2E8F0'},
                {'range': [0.10, 0.25], 'color': '#CBD5E1'},
                {'range': [0.25, max_val], 'color': '#94A3B8'}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.75,
                'value': 0.25
            }
        }
    ))
    
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def plot_telemetry_history(df: pd.DataFrame) -> go.Figure:
    """Plots CPU and Memory trends in a beautiful double y-axis plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cpu_utilization'],
        name="CPU Utilization",
        line=dict(color='#0EA5E9', width=2),
        fill='tozeroy',
        fillcolor='rgba(14, 165, 233, 0.05)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['memory_utilization'],
        name="Memory Utilization",
        line=dict(color='#8B5CF6', width=2),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.05)'
    ))
    
    fig.update_layout(
        title=dict(text="System Load History", font=dict(family='Outfit', size=16)),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0', linecolor='#CBD5E1'),
        yaxis=dict(title="Utilization (%)", showgrid=True, gridcolor='#E2E8F0', linecolor='#CBD5E1', range=[0, 105]),
    )
    return fig

def plot_anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Plots network latency and wraps flagged anomalies with red scatter points.
    Isolation Forest anomalies visualized interactively.
    """
    fig = go.Figure()
    
    # Latency line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['network_latency'],
        name="Network Latency",
        line=dict(color='#64748B', width=1.5, shape='spline'),
    ))
    
    # Filter anomalies
    anomalies = df[df['is_anomaly'] == True]
    
    # Anomaly marker scatter
    fig.add_trace(go.Scatter(
        x=anomalies['timestamp'],
        y=anomalies['network_latency'],
        name="Anomaly Trigger",
        mode='markers',
        marker=dict(color='#EF4444', size=8, symbol='circle-open', line=dict(width=2)),
        hovertemplate='<b>Anomaly Detected</b><br>Time: %{x}<br>Latency: %{y:.1f}ms<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Network Latency & Anomalies (Isolation Forest)", font=dict(family='Outfit', size=16)),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
        yaxis=dict(title="Latency (ms)", showgrid=True, gridcolor='#E2E8F0'),
    )
    return fig

def plot_forecast(df: pd.DataFrame) -> go.Figure:
    """
    Plots historical KPI (e.g. transaction count) alongside forecasted value
    (predicted t+1 in step-style projection).
    """
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp')
    
    fig = go.Figure()
    
    # Historical KPI line
    fig.add_trace(go.Scatter(
        x=df_sorted['timestamp'],
        y=df_sorted['kpi_value'],
        name="Actual KPI",
        line=dict(color='#10B981', width=2),
    ))
    
    # Forecasted points (shifted by 1 step forward to align prediction target)
    forecast_df = df_sorted.dropna(subset=['forecast_value'])
    if not forecast_df.empty:
        # Create t+1 timestamps for visualization
        # Find mean time delta between rows
        time_deltas = forecast_df['timestamp'].diff().dropna()
        avg_delta = time_deltas.mean() if not time_deltas.empty else pd.Timedelta(seconds=5)
        
        forecast_times = forecast_df['timestamp'] + avg_delta
        
        fig.add_trace(go.Scatter(
            x=forecast_times,
            y=forecast_df['forecast_value'],
            name="XGBoost Forecast (t+1)",
            line=dict(color='#F59E0B', width=1.5, dash='dash'),
            mode='lines+markers',
            marker=dict(size=4)
        ))
        
    fig.update_layout(
        title=dict(text="KPI Forecasting (XGBoost)", font=dict(family='Outfit', size=16)),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
        yaxis=dict(title="Transaction Volume (KPI)", showgrid=True, gridcolor='#E2E8F0'),
    )
    return fig

def plot_prediction_confidence(df: pd.DataFrame) -> go.Figure:
    """Plots probability of system health failure over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['failure_probability'],
        name="Failure Probability",
        line=dict(color='#EF4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.05)'
    ))
    
    # Add alarm line at 50%
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=0.50,
        x1=df['timestamp'].max(),
        y1=0.50,
        line=dict(color="orange", width=1.5, dash="dot"),
        name="Warning Threshold"
    )
    
    fig.update_layout(
        title=dict(text="System Health Risk Probability (Logistic Regression)", font=dict(family='Outfit', size=16)),
        hovermode="x",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
        yaxis=dict(title="Failure Prob [0, 1]", showgrid=True, gridcolor='#E2E8F0', range=[0, 1.05]),
    )
    return fig
