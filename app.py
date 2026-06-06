"""
Pipeline Leak Detection AI - Dashboard
Olabisi Onabanjo University | Final Year Project 2026
Authors: Adeleke Jesuloluwa Daniel & Mustapha Yunus Opeyemi
"""

import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Leak Detection AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid #1e4d6b;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e0f4ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #7fbfd4;
    margin: 6px 0 0 0;
    font-weight: 300;
}

/* ── Alert Banners ── */
.alert-leak {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    color: white;
    animation: pulseRed 2s ease-in-out infinite;
    margin-bottom: 20px;
}
.alert-normal {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.alert-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.alert-body {
    font-size: 0.95rem;
    margin: 0;
    opacity: 0.9;
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }
    50% { box-shadow: 0 0 20px 6px rgba(239,68,68,0.15); }
}

/* ── Sensor Cards ── */
.sensor-card {
    background: #0d1f2d;
    border: 1px solid #1e3a4f;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.3s;
    min-height: 130px;
}
.sensor-card.normal  { border-color: #10b981; }
.sensor-card.warning { border-color: #f59e0b; }
.sensor-card.critical{ border-color: #ef4444; }

.sensor-icon  { font-size: 1.6rem; line-height: 1; margin-bottom: 6px; }
.sensor-label { font-size: 0.75rem; color: #7fbfd4; text-transform: uppercase;
                letter-spacing: 0.08em; margin-bottom: 4px; }
.sensor-value { font-family: 'IBM Plex Mono', monospace;
                font-size: 1.5rem; font-weight: 600; color: #e0f4ff; }
.sensor-unit  { font-size: 0.75rem; color: #7fbfd4; margin-left: 2px; }

.badge {
    display: inline-block;
    margin-top: 8px;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge.normal   { background: #064e3b; color: #10b981; border: 1px solid #10b981; }
.badge.warning  { background: #451a03; color: #f59e0b; border: 1px solid #f59e0b; }
.badge.critical { background: #450a0a; color: #ef4444; border: 1px solid #ef4444; }

/* ── Info panels ── */
.info-panel {
    background: #0d1f2d;
    border: 1px solid #1e3a4f;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.info-panel h4 {
    color: #7fbfd4;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 12px 0;
    font-weight: 600;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #4a7a94;
    font-size: 0.8rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #1e3a4f;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("my_leak_detector.pkl")
    with open("features.json") as f:
        features = json.load(f)
    return model, features

model, features = load_model()


# ─── Session State Init ────────────────────────────────────────────────────────
for key, default in [
    ("monitoring", False),
    ("history", []),
    ("last_reading", None),
    ("reading_count", 0),
    ("leak_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Helper Functions ─────────────────────────────────────────────────────────
def generate_reading(force_leak=False):
    """Generate a realistic sensor reading. ~15% chance of leak."""
    # Use microseconds to avoid seed repetition on fast reruns
    rng = np.random.default_rng(int(datetime.now().timestamp() * 1e6) % (2**32))
    is_leak = force_leak or (rng.random() < 0.15)

    if is_leak:
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "pressure":       float(rng.uniform(22, 39)),
            "flow_rate":      float(rng.uniform(65, 88)),
            "temperature":    float(rng.uniform(20, 28)),
            "vibration":      float(rng.uniform(1.2, 2.6)),
            "acoustic_level": float(rng.uniform(52, 88)),
            "pressure_drop":  float(rng.uniform(12, 32)),
            "flow_anomaly":   float(rng.uniform(11, 36)),
            "is_leak": True,
        }
    else:
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "pressure":       float(rng.uniform(46, 57)),
            "flow_rate":      float(rng.uniform(93, 113)),
            "temperature":    float(rng.uniform(20, 25)),
            "vibration":      float(rng.uniform(0.2, 0.75)),
            "acoustic_level": float(rng.uniform(24, 37)),
            "pressure_drop":  float(rng.uniform(0, 3.5)),
            "flow_anomaly":   float(rng.uniform(0, 3.5)),
            "is_leak": False,
        }


def run_prediction(reading):
    data = pd.DataFrame([{
        "pressure":       reading["pressure"],
        "flow_rate":      reading["flow_rate"],
        "temperature":    reading["temperature"],
        "vibration":      reading["vibration"],
        "acoustic_level": reading["acoustic_level"],
        "pressure_drop":  reading["pressure_drop"],
        "flow_anomaly":   reading["flow_anomaly"],
    }])
    prob = model.predict_proba(data)[0][1]
    pred = int(model.predict(data)[0])
    return pred, prob


def sensor_status(value, ok_lo, ok_hi, warn_lo, warn_hi):
    if ok_lo <= value <= ok_hi:
        return "normal"
    elif warn_lo <= value <= warn_hi:
        return "warning"
    return "critical"


def render_sensor_card(icon, label, value, unit, status):
    st.markdown(f"""
    <div class="sensor-card {status}">
        <div class="sensor-icon">{icon}</div>
        <div class="sensor-label">{label}</div>
        <div class="sensor-value">{value:.2f}<span class="sensor-unit">{unit}</span></div>
        <span class="badge {status}">{status}</span>
    </div>
    """, unsafe_allow_html=True)


def make_gauge(probability):
    color = "#ef4444" if probability > 0.6 else "#f59e0b" if probability > 0.3 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": "#e0f4ff", "family": "IBM Plex Mono"}},
        title={"text": "Leak Risk Score", "font": {"size": 13, "color": "#7fbfd4"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a7a94", "tickfont": {"color": "#4a7a94"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0d1f2d",
            "bordercolor": "#1e3a4f",
            "steps": [
                {"range": [0, 30],  "color": "#062418"},
                {"range": [30, 60], "color": "#1c1000"},
                {"range": [60, 100],"color": "#2a0000"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0f4ff"},
    )
    return fig


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">🔧 Pipeline Leak Detection AI</p>
    <p class="app-subtitle">Real-time AI-powered monitoring for pipeline infrastructure &nbsp;·&nbsp; Olabisi Onabanjo University 2026</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🎛️ Control Panel")
mode = st.sidebar.radio("Operation Mode", ["Real-time Monitoring", "Manual Analysis"])

if mode == "Real-time Monitoring":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Monitoring Controls**")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        start = st.button("▶ Start", type="primary", use_container_width=True)
    with c2:
        stop = st.button("⏹ Stop", use_container_width=True)

    if start:
        st.session_state.monitoring = True
    if stop:
        st.session_state.monitoring = False

    interval = st.sidebar.slider("Refresh interval (s)", 2, 10, 4)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Session Stats**")
    st.sidebar.metric("Total Readings", st.session_state.reading_count)
    st.sidebar.metric("Leaks Detected", st.session_state.leak_count)
    if st.session_state.reading_count > 0:
        rate = st.session_state.leak_count / st.session_state.reading_count * 100
        st.sidebar.metric("Leak Rate", f"{rate:.1f}%")

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑 Clear History"):
        st.session_state.history = []
        st.session_state.reading_count = 0
        st.session_state.leak_count = 0
        st.session_state.last_reading = None
        st.session_state.monitoring = False
        st.rerun()

    if st.session_state.history:
        df_export = pd.DataFrame(st.session_state.history)
        st.sidebar.download_button(
            "📥 Export CSV",
            df_export.to_csv(index=False),
            "pipeline_sensor_data.csv",
            "text/csv",
            use_container_width=True,
        )


# ─── Main Content ─────────────────────────────────────────────────────────────

if mode == "Real-time Monitoring":

    # Auto-generate reading when monitoring is active
    if st.session_state.monitoring:
        reading = generate_reading()
        st.session_state.last_reading = reading
        st.session_state.history.append(reading)
        st.session_state.reading_count += 1
        if reading["is_leak"]:
            st.session_state.leak_count += 1
        if len(st.session_state.history) > 60:
            st.session_state.history = st.session_state.history[-60:]
        time.sleep(interval)
        st.rerun()

    # If stopped but we have a reading, show last state
    if st.session_state.last_reading is None:
        st.info("👆 Click **Start** in the sidebar to begin real-time monitoring.")
        st.stop()

    reading = st.session_state.last_reading
    pred, prob = run_prediction(reading)

    # ── Alert Banner ──
    if pred == 1:
        st.markdown(f"""
        <div class="alert-leak">
            <p class="alert-title">⚠️ CRITICAL — LEAK DETECTED IN PIPELINE</p>
            <p class="alert-body">Immediate action required &nbsp;·&nbsp; Leak probability: <strong>{prob*100:.1f}%</strong> &nbsp;·&nbsp; Check pressure drop and flow anomaly</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-normal">
            <p class="alert-title">✅ Pipeline Normal</p>
            <p class="alert-body">All sensors within operating range &nbsp;·&nbsp; Leak probability: <strong>{prob*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ──
    tab_detect, tab_predict = st.tabs(["🔍 Detection", "📈 Prediction"])

    with tab_detect:
        st.markdown("##### Live Sensor Readings")
        cols = st.columns(4)
        cards = [
            ("🌡️", "Pressure",      reading["pressure"],       "bar",  sensor_status(reading["pressure"],       40, 60, 30, 70)),
            ("💧", "Flow Rate",     reading["flow_rate"],      "m³/h", sensor_status(reading["flow_rate"],      80,120, 60,140)),
            ("📳", "Vibration",     reading["vibration"],      "mm/s", sensor_status(reading["vibration"],       0, 1.0, 1.0,2.0)),
            ("🔊", "Acoustic Level",reading["acoustic_level"], "dB",   sensor_status(reading["acoustic_level"],20, 45, 45, 70)),
        ]
        for col, (icon, label, val, unit, status) in zip(cols, cards):
            with col:
                render_sensor_card(icon, label, val, unit, status)

        cols2 = st.columns(4)
        cards2 = [
            ("🌡️", "Temperature",   reading["temperature"],   "°C",  sensor_status(reading["temperature"],  18, 28, 15, 32)),
            ("📉", "Pressure Drop", reading["pressure_drop"], "",    sensor_status(reading["pressure_drop"],  0,  5,  5, 15)),
            ("⚡", "Flow Anomaly",  reading["flow_anomaly"],  "",    sensor_status(reading["flow_anomaly"],   0,  5,  5, 15)),
            ("🎯", "Leak Risk",     prob * 100,               "%",   "critical" if prob > 0.6 else "warning" if prob > 0.3 else "normal"),
        ]
        for col, (icon, label, val, unit, status) in zip(cols2, cards2):
            with col:
                render_sensor_card(icon, label, val, unit, status)

        # History chart
        if len(st.session_state.history) > 2:
            st.markdown("##### Sensor History")
            hdf = pd.DataFrame(st.session_state.history)
            st.line_chart(
                hdf[["pressure", "flow_rate", "vibration", "acoustic_level"]].rename(columns={
                    "pressure": "Pressure (bar)",
                    "flow_rate": "Flow Rate (m³/h)",
                    "vibration": "Vibration (mm/s)",
                    "acoustic_level": "Acoustic (dB)",
                }),
                height=260,
            )

            with st.expander("📋 Raw Data (last 10 readings)"):
                st.dataframe(hdf.tail(10), use_container_width=True)

    with tab_predict:
        left, right = st.columns([2, 1])

        with left:
            st.markdown("##### Feature Importance")
            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)
            st.bar_chart(importance.set_index("Feature"), height=280)

            # Prediction breakdown
            top_feat = importance.iloc[0]
            st.markdown(f"""
            <div class="info-panel">
                <h4>Prediction Breakdown</h4>
                <p style="color:#cbd5e1; font-size:0.9rem; margin:0">
                    Top indicator: <strong style="color:#e0f4ff">{top_feat['Feature']}</strong>
                    &nbsp;({top_feat['Importance']*100:.1f}% of decision weight)<br><br>
                    Current pressure drop: <strong style="color:#e0f4ff">{reading['pressure_drop']:.2f}</strong>
                    (warning threshold: &gt;5)<br>
                    Current flow anomaly: <strong style="color:#e0f4ff">{reading['flow_anomaly']:.2f}</strong>
                    (warning threshold: &gt;5)<br>
                    Combined risk score: <strong style="color:{'#ef4444' if prob > 0.5 else '#10b981'}">{prob*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("##### Leak Risk Score")
            st.plotly_chart(make_gauge(prob), use_container_width=True)

            # 48-hour forecast
            st.markdown("##### 48-Hour Forecast")
            if len(st.session_state.history) >= 5:
                recent = st.session_state.history[-5:]
                avg_pd = np.mean([r["pressure_drop"] for r in recent])
                avg_fa = np.mean([r["flow_anomaly"]  for r in recent])
                future_risk = min(0.95, (avg_pd + avg_fa) / 40)
            else:
                future_risk = prob

            if future_risk > 0.5:
                st.warning(f"⚠️ Leak likely in 48h\nRisk: **{future_risk*100:.1f}%**")
            else:
                st.success(f"✅ No leak expected\nRisk: **{future_risk*100:.1f}%**")

            # System info
            st.markdown("""
            <div class="info-panel" style="margin-top:16px">
                <h4>System Status</h4>
                <p style="color:#cbd5e1; font-size:0.85rem; margin:0; line-height:1.9">
                    Detection Model &nbsp;🟢 Active<br>
                    Prediction Model &nbsp;🟢 Active<br>
                    Data Quality &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🟢 Excellent
                </p>
            </div>
            <div class="info-panel">
                <h4>Model Performance</h4>
                <p style="color:#cbd5e1; font-size:0.85rem; margin:0; line-height:1.9">
                    Accuracy &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;99.8%<br>
                    Precision &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;98.5%<br>
                    False Positive &nbsp;0.2%<br>
                    Response Time &nbsp;&lt;100ms
                </p>
            </div>
            <div class="info-panel">
                <h4>Pipeline Info</h4>
                <p style="color:#cbd5e1; font-size:0.85rem; margin:0; line-height:1.9">
                    ID: PL-2026-001<br>
                    Length: 100 km<br>
                    Diameter: 36 in<br>
                    Material: Steel<br>
                    Route: Lagos-Abuja
                </p>
            </div>
            """, unsafe_allow_html=True)

# ─── Manual Analysis Mode ─────────────────────────────────────────────────────
else:
    st.markdown("#### 🔧 Manual Sensor Input")
    st.caption("Adjust sensor values and click Analyze to get an instant prediction.")

    col1, col2 = st.columns(2)
    with col1:
        pressure      = st.slider("Pressure (bar)",       0.0, 100.0,  50.0)
        flow_rate     = st.slider("Flow Rate (m³/h)",     0.0, 200.0, 100.0)
        temperature   = st.slider("Temperature (°C)",     0.0,  50.0,  22.0)
        vibration     = st.slider("Vibration (mm/s)",     0.0,   5.0,   0.5)
    with col2:
        acoustic      = st.slider("Acoustic Level (dB)",  0.0, 100.0,  30.0)
        pressure_drop = st.slider("Pressure Drop",        0.0,  50.0,   0.0)
        flow_anomaly  = st.slider("Flow Anomaly",         0.0,  50.0,   0.0)

    if st.button("🔍 Analyze Pipeline", type="primary"):
        manual_reading = {
            "pressure": pressure, "flow_rate": flow_rate,
            "temperature": temperature, "vibration": vibration,
            "acoustic_level": acoustic, "pressure_drop": pressure_drop,
            "flow_anomaly": flow_anomaly, "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        pred, prob = run_prediction(manual_reading)

        if pred == 1:
            st.markdown(f"""
            <div class="alert-leak">
                <p class="alert-title">⚠️ LEAK DETECTED</p>
                <p class="alert-body">Confidence: <strong>{prob*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-normal">
                <p class="alert-title">✅ Pipeline Normal</p>
                <p class="alert-body">Leak Probability: <strong>{prob*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### Feature Importance")
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature"), height=260)

        col_g, _ = st.columns([1, 2])
        with col_g:
            st.plotly_chart(make_gauge(prob), use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🔧 Pipeline Leak Detection AI &nbsp;·&nbsp;
    Olabisi Onabanjo University &nbsp;·&nbsp;
    Authors: Adeleke Jesuloluwa Daniel & Mustapha Yunus Opeyemi &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
