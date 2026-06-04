"""Helper functions for the leak detection dashboard."""

import numpy as np
import pandas as pd
from datetime import datetime


def generate_sensor_data(leak_probability=0.1):
    """Generate realistic pipeline sensor readings."""
    np.random.seed(int(datetime.now().timestamp()))
    
    # Base normal readings
    base = {
        'pressure': 50 + np.random.normal(0, 3),
        'flow_rate': 100 + np.random.normal(0, 5),
        'vibration': 0.5 + np.random.normal(0, 0.1),
        'acoustic_level': 30 + np.random.normal(0, 3),
        'temperature': 22 + np.random.normal(0, 1),
    }
    
    # Random leak scenario
    is_leak = np.random.random() < leak_probability
    
    if is_leak:
        readings = {
            'pressure': base['pressure'] - np.random.uniform(10, 20),
            'flow_rate': base['flow_rate'] - np.random.uniform(10, 30),
            'vibration': base['vibration'] + np.random.uniform(0.5, 1.5),
            'acoustic_level': base['acoustic_level'] + np.random.uniform(15, 40),
            'temperature': base['temperature'],
            'pressure_drop': np.random.uniform(10, 25),
            'flow_anomaly': np.random.uniform(10, 30),
        }
    else:
        readings = {
            'pressure': base['pressure'],
            'flow_rate': base['flow_rate'],
            'vibration': base['vibration'],
            'acoustic_level': base['acoustic_level'],
            'temperature': base['temperature'],
            'pressure_drop': np.random.uniform(0, 3),
            'flow_anomaly': np.random.uniform(0, 3),
        }
    
    readings['timestamp'] = datetime.now().strftime("%H:%M:%S")
    readings['is_leak'] = int(is_leak)
    
    return readings


def get_status(value, normal_range, warning_range):
    """Determine sensor status based on thresholds."""
    if normal_range[0] <= value <= normal_range[1]:
        return "Normal", "status-normal"
    elif warning_range[0] <= value <= warning_range[1]:
        return "Warning", "status-warning"
    else:
        return "Critical", "status-critical"


def create_prediction_data(readings, features):
    """Format readings for model prediction."""
    return pd.DataFrame([{
        'pressure': readings['pressure'],
        'flow_rate': readings['flow_rate'],
        'temperature': readings['temperature'],
        'vibration': readings['vibration'],
        'acoustic_level': readings['acoustic_level'],
        'pressure_drop': readings['pressure_drop'],
        'flow_anomaly': readings['flow_anomaly']
    }])
