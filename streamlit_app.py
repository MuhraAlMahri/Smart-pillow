import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

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
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# **Start Monitoring Button**
if st.button("🚀 Start Sleep Monitoring"):
    st.session_state.monitoring = True
    st.session_state.start_time = datetime.now()
    st.session_state.hr_data = []
    st.session_state.rr_data = []
    st.session_state.timestamps = []
    st.session_state.alerts = []
    st.rerun()  # Forces app to refresh

# **Stop Monitoring Button**
if st.button("⏹️ Stop Monitoring"):
    st.session_state.monitoring = False
    st.rerun()  # Refresh to display final report

# **Real-Time Chart & Alerts**
chart_placeholder = st.empty()
alert_placeholder = st.empty()
report_placeholder = st.empty()

# =============================
# 3. Real-Time Data Simulation (Moving Graph)
# =============================
if st.session_state.monitoring:
    elapsed_time = (datetime.now() - st.session_state.start_time).total_seconds()

    if elapsed_time < 8 * 3600:
        # Simulated heart rate (HR) and respiratory rate (RR)
        new_hr = np.random.randint(60, 100)
        new_rr = np.random.randint(10, 20)

        # Store only the last 60 minutes of data (moving window)
        if len(st.session_state.hr_data) > 300:
            st.session_state.hr_data.pop(0)
            st.session_state.rr_data.pop(0)
            st.session_state.timestamps.pop(0)

        # Append new data
        st.session_state.hr_data.append(new_hr)
        st.session_state.rr_data.append(new_rr)
        st.session_state.timestamps.append(datetime.now())

        # **Plot real-time graph**
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(st.session_state.timestamps, st.session_state.hr_data, label="Heart Rate (BPM)", color="red", linewidth=2)
        ax.plot(st.session_state.timestamps, st.session_state.rr_data, label="Respiratory Rate (Breaths/min)", color="blue", linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Values")
        ax.legend()
        ax.grid()
        chart_placeholder.pyplot(fig)

        # **Health Alerts**
        alert_msg = None
        if new_hr > 90:
            alert_msg = "⚠️ High Blood Pressure Detected! Consult a doctor."
        elif new_rr < 10:
            alert_msg = "⚠️ Possible Sleep Apnea Detected! Consider medical evaluation."

        if alert_msg:
            st.session_state.alerts.append((datetime.now().strftime("%H:%M:%S"), alert_msg))
        
        # **Show latest alert**
        if st.session_state.alerts:
            latest_alert = st.session_state.alerts[-1]
            alert_placeholder.error(f"{latest_alert[1]} (Time: {latest_alert[0]})")
        else:
            alert_placeholder.success("✅ Normal Sleep & Cardiovascular Health")

        # **Wait 1 second before rerunning**
        time.sleep(1)
        st.rerun()

    else:
        # **Stop Monitoring after 8 Hours**
        st.session_state.monitoring = False
        st.rerun()

# =============================
# 4. Sleep Report Summary (After 8 Hours)
# =============================
if not st.session_state.monitoring and st.session_state.start_time:
    if st.session_state.hr_data:
        # Calculate statistics
        avg_hr = np.mean(st.session_state.hr_data)
        avg_rr = np.mean(st.session_state.rr_data)
        max_hr = np.max(st.session_state.hr_data)
        min_hr = np.min(st.session_state.hr_data)
        max_rr = np.max(st.session_state.rr_data)
        min_rr = np.min(st.session_state.rr_data)

        # Create DataFrame for Alerts
        alert_df = pd.DataFrame(st.session_state.alerts, columns=["Time", "Alert"])

        # Display summary
        report_placeholder.write("## 💤 Sleep & Blood Pressure Report")
        report_placeholder.write(f"**📅 Sleep Duration:** 8 Hours")
        report_placeholder.write(f"**❤️ Avg Heart Rate:** {avg_hr:.1f} BPM (Min: {min_hr} | Max: {max_hr})")
        report_placeholder.write(f"**💨 Avg Respiratory Rate:** {avg_rr:.1f} Breaths/min (Min: {min_rr} | Max: {max_rr})")
        
        if not alert_df.empty:
            report_placeholder.write("### 🚨 Health Alerts During Sleep:")
            report_placeholder.dataframe(alert_df)

        st.success("✅ Sleep monitoring completed! Summary report generated.")
