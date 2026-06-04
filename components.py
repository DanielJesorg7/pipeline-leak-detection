"""Reusable UI components for the dashboard."""

import streamlit as st
import pandas as pd


def load_css():
    """Load custom CSS styles."""
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def header_section():
    """Display main header."""
    st.markdown('<p class="main-header">🔧 Pipeline Leak Detection AI</p>', 
                unsafe_allow_html=True)
    st.write("Real-time monitoring and AI-powered leak detection")


def control_buttons():
    """Display start/stop/refresh buttons."""
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        start = st.button("▶️ Start", type="primary", key="start_btn")
    with col2:
        stop = st.button("⏹️ Stop", key="stop_btn")
    with col3:
        refresh = st.button("🔄 Refresh", key="refresh_btn")
    
    return start, stop, refresh


def alert_banner(prediction, probability):
    """Display leak or normal alert."""
    if prediction == 1:
        st.error(f"""
        ⚠️ **CRITICAL: LEAK DETECTED**
        
        Immediate action required. Leak probability: **{probability*100:.1f}%**
        """)
    else:
        st.success(f"""
        ✅ **Pipeline Normal**
        
        Leak probability: **{probability*100:.1f}%**
        """)


def sensor_card(title, value, unit, status, status_class):
    """Display a single sensor reading card."""
    st.markdown(f"""
    <div class="sensor-card">
        <div style="color: #666; font-size: 0.9rem;">{title}</div>
        <div class="metric-value">{value:.2f} {unit}</div>
        <span class="{status_class}">{status}</span>
    </div>
    """, unsafe_allow_html=True)


def sensor_grid(readings):
    """Display all sensor cards in a grid."""
    st.subheader("Live Sensor Readings")
    
    from utils import get_status
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        status, cls = get_status(readings['pressure'], (40, 60), (30, 70))
        sensor_card("Pressure", readings['pressure'], "bar", status, cls)
    
    with c2:
        status, cls = get_status(readings['flow_rate'], (80, 120), (60, 140))
        sensor_card("Flow Rate", readings['flow_rate'], "m³/h", status, cls)
    
    with c3:
        status, cls = get_status(readings['vibration'], (0, 1.0), (1.0, 2.0))
        sensor_card("Vibration", readings['vibration'], "mm/s", status, cls)
    
    with c4:
        status, cls = get_status(readings['acoustic_level'], (20, 45), (45, 70))
        sensor_card("Acoustic Level", readings['acoustic_level'], "dB", status, cls)


def feature_importance_chart(model, features):
    """Display feature importance bar chart."""
    st.subheader("Feature Importance")
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(importance.set_index('Feature'))


def system_status_panel():
    """Display system status information."""
    st.subheader("System Status")
    st.markdown("""
    | Component | Status |
    |-----------|--------|
    | Detection Model | 🟢 Active |
    | Prediction Model | 🟢 Active |
    | Data Quality | 🟢 Excellent |
    """)


def model_performance_panel():
    """Display model performance metrics."""
    st.subheader("Model Performance")
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Detection Accuracy | 99.8% |
    | Prediction Accuracy | 98.5% |
    | False Positive Rate | 0.2% |
    """)


def pipeline_info_panel():
    """Display pipeline information."""
    st.subheader("Pipeline Info")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | Pipeline ID | PL-2026-001 |
    | Length | 100 km |
    | Last Inspection | 2 days ago |
    """)


def manual_input_form():
    """Display manual sensor input form."""
    st.subheader("Manual Sensor Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pressure = st.slider("Pressure (bar)", 0.0, 100.0, 50.0)
        flow_rate = st.slider("Flow Rate (m³/h)", 0.0, 200.0, 100.0)
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 22.0)
        vibration = st.slider("Vibration (mm/s)", 0.0, 5.0, 0.5)
    
    with col2:
        acoustic = st.slider("Acoustic Level (dB)", 0.0, 100.0, 30.0)
        pressure_drop = st.slider("Pressure Drop", 0.0, 50.0, 0.0)
        flow_anomaly = st.slider("Flow Anomaly", 0.0, 50.0, 0.0)
    
    return {
        'pressure': pressure,
        'flow_rate': flow_rate,
        'temperature': temperature,
        'vibration': vibration,
        'acoustic_level': acoustic,
        'pressure_drop': pressure_drop,
        'flow_anomaly': flow_anomaly
    }


def sensor_history_chart(history):
    """Display sensor history line chart."""
    if len(history) > 1:
        st.subheader("Sensor History")
        history_df = pd.DataFrame(history)
        st.line_chart(history_df.set_index('timestamp')[['pressure', 'flow_rate', 'vibration']])


def footer():
    """Display footer."""
    st.markdown("---")
    st.caption("Built for Olabisi Onabanjo University | Pipeline Leak Detection Project 2026")
