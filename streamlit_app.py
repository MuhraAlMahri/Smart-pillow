import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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
# 2. Streamlit Dashboard Setup
# =============================
st.title("🛏️ AI-Powered Smart Pillow Health Monitor")
st.write("### 📊 Real-Time Sleep & Blood Pressure Monitoring")

# **Initialize session state variables**
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "hr_data" not in st.session_state:
    st.session_state.hr_data = []
if "rr_data" not in st.session_state:
    st.session_state.rr_data = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []
if "sleep_stages" not in st.session_state:
    st.session_state.sleep_stages = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# **Start Monitoring Button**
if st.button("🚀 Start Sleep Monitoring"):
    st.session_state.monitoring = True
    st.session_state.start_time = datetime.now()
    st.session_state.hr_data = []
    st.session_state.rr_data = []
    st.session_state.timestamps = []
    st.session_state.sleep_stages = []
    st.session_state.alerts = []

# **Stop Monitoring Button**
if st.button("⏹️ Stop Monitoring"):
    st.session_state.monitoring = False

# **Real-Time Chart & Alerts**
chart_placeholder = st.empty()
sleep_stage_placeholder = st.empty()
alert_placeholder = st.empty()
report_placeholder = st.empty()

# =============================
# 3. Real-Time Data Simulation (Moving Graph)
# =============================
def determine_sleep_stage(last_stage, time_elapsed):
    """Simulate a full sleep cycle in just 5 minutes for the demo"""
    if time_elapsed < 1 * 60:  # First 1 min → Light Sleep
        return "Light Sleep"
    elif time_elapsed < 2.5 * 60:  # 1-2.5 min → Deep Sleep
        return "Deep Sleep"
    elif time_elapsed < 4 * 60:  # 2.5-4 min → REM Sleep
        return "REM Sleep"
    else:  # 4-5 min → Awake (cycle restarts)
        return "Awake"

# **Monitoring Process**
while st.session_state.monitoring:
    elapsed_time = (datetime.now() - st.session_state.start_time).total_seconds()

    if elapsed_time >= 5 * 60:  # Stop after 5 minutes
        st.session_state.monitoring = False
        break

    # Simulated heart rate (HR) and respiratory rate (RR)
    new_hr = np.random.randint(60, 100)
    new_rr = np.random.randint(10, 20)

    # Determine sleep stage
    sleep_stage = determine_sleep_stage(
        st.session_state.sleep_stages[-1] if st.session_state.sleep_stages else "Awake",
        elapsed_time
    )

    # Store data
    st.session_state.hr_data.append(new_hr)
    st.session_state.rr_data.append(new_rr)
    st.session_state.timestamps.append(datetime.now())
    st.session_state.sleep_stages.append(sleep_stage)

    # Real-time graph updates...
    time.sleep(1)  # 1 second = 1 minute of sleep
# =============================
# 4. Sleep Report Summary (Anytime Monitoring Stops)
# =============================
if not st.session_state.monitoring and st.session_state.start_time:
    if st.session_state.hr_data:
        elapsed_time = (datetime.now() - st.session_state.start_time).total_seconds() / 3600  # Convert to hours
        avg_hr = np.mean(st.session_state.hr_data)
        min_hr = np.min(st.session_state.hr_data)
        max_hr = np.max(st.session_state.hr_data)
        hr_variability = np.std(st.session_state.hr_data)  # Heart Rate Variability

        avg_rr = np.mean(st.session_state.rr_data)
        min_rr = np.min(st.session_state.rr_data)
        max_rr = np.max(st.session_state.rr_data)

        sleep_stage_counts = pd.Series(st.session_state.sleep_stages).value_counts(normalize=True) * 100  # Percentage

        # Display summary
        report_placeholder.write("## 💤 Sleep & Blood Pressure Report")
        report_placeholder.write(f"**📅 Sleep Duration:** {elapsed_time:.2f} hours")
        report_placeholder.write(f"**❤️ Avg Heart Rate:** {avg_hr:.1f} BPM (Min: {min_hr} | Max: {max_hr})")
        report_placeholder.write(f"**📊 Heart Rate Variability (HRV):** {hr_variability:.2f}")
        report_placeholder.write(f"**💨 Avg Respiratory Rate:** {avg_rr:.1f} Breaths/min (Min: {min_rr} | Max: {max_rr})")
        report_placeholder.write("### 🛏️ Sleep Stages Breakdown:")
        report_placeholder.write(sleep_stage_counts.to_string())

        # Show health alerts if any
        if st.session_state.alerts:
            alert_df = pd.DataFrame(st.session_state.alerts, columns=["Time", "Alert"])
            report_placeholder.write("### 🚨 Health Alerts During Sleep:")
            report_placeholder.dataframe(alert_df)

        st.success("✅ Sleep monitoring completed! Summary report generated.")
