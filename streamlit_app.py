import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import time
import matplotlib.pyplot as plt

# =============================
# 1. AI Model for Health Monitoring (LSTM-Based)
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

# =============================
# 2. Streamlit Dashboard
# =============================
st.title("🛏️ AI-Powered Smart Pillow Health Monitor")

st.write("### 📊 Real-Time Health Data Monitoring")

# **Initialize session state variables for real-time updates**
if "heart_rate" not in st.session_state:
    st.session_state.heart_rate = 75
if "respiratory_rate" not in st.session_state:
    st.session_state.respiratory_rate = 15
if "hr_data" not in st.session_state:
    st.session_state.hr_data = []
if "rr_data" not in st.session_state:
    st.session_state.rr_data = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# **Start real-time simulation button**
if st.button("🚀 Start Real-Time Monitoring"):
    st.session_state.monitoring = True

if st.button("⏹️ Stop Monitoring"):
    st.session_state.monitoring = False

# **Real-time Chart Area**
chart_placeholder = st.empty()
alert_placeholder = st.empty()

# **Loop for Real-Time Updates**
while st.session_state.get("monitoring", False):
    # Simulate real-time data updates
    new_hr = st.session_state.heart_rate + np.random.randint(-3, 3)
    new_rr = st.session_state.respiratory_rate + np.random.randint(-1, 2)
    
    # Store only the last 50 data points (moving window)
    if len(st.session_state.hr_data) > 50:
        st.session_state.hr_data.pop(0)
        st.session_state.rr_data.pop(0)
        st.session_state.timestamps.pop(0)

    # Append new data
    st.session_state.hr_data.append(new_hr)
    st.session_state.rr_data.append(new_rr)
    st.session_state.timestamps.append(time.time())

    # **Plot live data**
    fig, ax = plt.subplots()
    ax.plot(st.session_state.timestamps, st.session_state.hr_data, label="Heart Rate (BPM)", color="red")
    ax.plot(st.session_state.timestamps, st.session_state.rr_data, label="Respiratory Rate (Breaths/min)", color="blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend()
    ax.grid()

    # **Update live chart**
    chart_placeholder.pyplot(fig)

    # **AI Health Alerts**
    if new_hr > 90:
        alert_placeholder.error("⚠️ High Blood Pressure Detected! Consult a doctor.")
    elif new_rr < 10:
        alert_placeholder.warning("⚠️ Possible Sleep Apnea Detected! Consider medical evaluation.")
    else:
        alert_placeholder.success("✅ Normal Cardiovascular & Respiratory Health")

    # **Wait for next update (1 second)**
    time.sleep(1)
