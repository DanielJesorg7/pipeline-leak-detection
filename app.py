"""Main Streamlit application for Pipeline Leak Detection."""

import streamlit as st
import joblib
import json

from components import (
    load_css, header_section, control_buttons, alert_banner,
    sensor_grid, feature_importance_chart, system_status_panel,
    model_performance_panel, pipeline_info_panel, manual_input_form,
    sensor_history_chart, footer
)
from utils import generate_sensor_data, create_prediction_data


# Page config
st.set_page_config(page_title="Pipeline Leak Detection AI", layout="wide")

# Load styles
load_css()

# Load model
@st.cache_resource
def load_model():
    return joblib.load('my_leak_detector.pkl')

model = load_model()

with open('features.json') as f:
    features = json.load(f)

# Initialize session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'sensor_history' not in st.session_state:
    st.session_state.sensor_history = []

# Header
header_section()

# Controls
start, stop, refresh = control_buttons()

# Handle button clicks
if start:
    st.session_state.monitoring = True
if stop:
    st.session_state.monitoring = False
if refresh:
    st.session_state.sensor_history = []

# Main display
if st.session_state.monitoring:
    # Generate data
    readings = generate_sensor_data(leak_probability=0.1)
    
    # Store history
    st.session_state.sensor_history.append(readings)
    if len(st.session_state.sensor_history) > 20:
        st.session_state.sensor_history = st.session_state.sensor_history[-20:]
    
    # Predict
    data = create_prediction_data(readings, features)
    prob = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]
    
    # Display
    alert_banner(pred, prob)
    sensor_grid(readings)
    
    # Analysis section
    col_left, col_right = st.columns(2)
    with col_left:
        feature_importance_chart(model, features)
        sensor_history_chart(st.session_state.sensor_history)
    
    with col_right:
        system_status_panel()
        model_performance_panel()
        pipeline_info_panel()

else:
    # Manual mode
    st.info("Click **Start** to begin real-time monitoring, or use manual input below")
    
    readings = manual_input_form()
    
    if st.button("🔍 Analyze Pipeline", type="primary"):
        data = create_prediction_data(readings, features)
        prob = model.predict_proba(data)[0][1]
        pred = model.predict(data)[0]
        
        alert_banner(pred, prob)

# Footer
footer()
