# Pipeline Leak Detection System

AI-powered leak detection for pipeline infrastructure using sensor data analysis.

## Overview
This project uses machine learning to detect pipeline leaks in real-time by analyzing sensor data including pressure, flow rate, temperature, vibration, and acoustic levels.

## Features
- Real-time leak detection from 7 sensor inputs
- Feature importance analysis identifies key leak indicators
- Interactive demo with adjustable sensor values

## Dataset
- 10,000 pipeline sensor readings
- 7 features: pressure, flow_rate, temperature, vibration, acoustic_level, pressure_drop, flow_anomaly
- Target: is_leak (0 = normal, 1 = leak detected)

## Model
- Algorithm: Random Forest Classifier
- Training: 80/20 train-test split with stratification
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score

## Files
| File | Description |
|------|-------------|
| `my_leak_detector.pkl` | Trained model file |
| `features.json` | Feature names for prediction |
| `leak_prediction_and_detection_model.ipynb` | Training notebook with full code |

## How to Use

### Quick Prediction
```python
import joblib
import pandas as pd

model = joblib.load('my_leak_detector.pkl')

# Example sensor reading
data = pd.DataFrame([{
    'pressure': 50.0,
    'flow_rate': 100.0,
    'temperature': 22.0,
    'vibration': 0.5,
    'acoustic_level': 30.0,
    'pressure_drop': 0.0,
    'flow_anomaly': 0.0
}])
Requirements
Python 3.8+
scikit-learn
pandas
joblib
Author
Adeleke Jesuloluwa Daniel and Mustapha Yunus Opeyemi
Olabisi  Onabanjo University
04,June ,2026
prediction = model.predict(data)
probability = model.predict_proba(data)[0][1]
print(f"Leak Probability: {probability*100:.1f}%")
