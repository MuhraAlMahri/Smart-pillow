import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from scipy.signal import butter, filtfilt
import time

# =============================
# 1. Simulating Real-Time Health Data (Heart Rate, Respiratory Signals)
# =============================
def generate_synthetic_data(length=300):
    time = np.linspace(0, length, length)
    heart_rate = 70 + 5 * np.sin(0.1 * time) + np.random.normal(0, 2, length)  # Simulated heart rate
    respiratory_rate = 15 + 2 * np.sin(0.05 * time) + np.random.normal(0, 1, length)  # Simulated respiratory rate
    return heart_rate, respiratory_rate

# =============================
# 2. AI Model for Health Monitoring (LSTM-Based)
# =============================
class HealthMonitorAI(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=3):
        super(HealthMonitorAI, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output: Normal, Apnea, High BP
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

model = HealthMonitorAI()

# =============================
# 3. Streamlit Dashboard for Real-Time Monitoring
# =============================
st.title("AI-Powered Smart Pillow Health Monitor")

st.write("### Simulated Real-Time Health Data")
hr_placeholder = st.empty()
rr_placeholder = st.empty()
status_placeholder = st.empty()

# Real-time health monitoring loop
for i in range(100):  # Simulating 100 time steps
    heart_rate, respiratory_rate = generate_synthetic_data(300)
    current_hr = heart_rate[-1]
    current_rr = respiratory_rate[-1]
    
    hr_placeholder.line_chart(heart_rate)
    rr_placeholder.line_chart(respiratory_rate)
    
    # AI-Based Health Insights
    if current_hr > 80:
        status_placeholder.error("⚠️ High Blood Pressure Detected! Consult a doctor.")
    elif current_rr < 10:
        status_placeholder.warning("⚠️ Possible Sleep Apnea Detected! Consider medical evaluation.")
    else:
        status_placeholder.success("✅ Normal Cardiovascular & Respiratory Health")
    
    time.sleep(1)  # Simulate real-time data update
