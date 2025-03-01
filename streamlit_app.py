import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# =============================
# 1. Streamlit Dashboard Setup
# =============================
st.title("🛏️ AI-Powered Smart Pillow Health Monitor")
st.write("### 📊 Real-Time Sleep & Blood Pressure Monitoring")

# **Simulation Settings**
REAL_SLEEP_HOURS = 8  # Real sleep time (8 hours)
SIMULATED_SECONDS = 30  # Simulated time for demo (30 seconds)
TIME_MULTIPLIER = (REAL_SLEEP_HOURS * 3600) / SIMULATED_SECONDS  # Speed up time

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
# 2. Real-Time Data Simulation
# =============================
def determine_sleep_stage(hr, rr):
    """Simulate sleep stages based on HR and RR"""
    if hr > 90 or rr > 18:
        return "Awake"
    elif 70 <= hr <= 85 and 14 <= rr <= 18:
        return "Light Sleep"
    elif 60 <= hr < 70 and 12 <= rr < 14:
        return "Deep Sleep"
    elif 60 <= hr <= 85 and 14 <= rr <= 16:
        return "REM Sleep"
    return "Awake"

# **Run for 30 seconds (fast-forwarding 8 hours of sleep)**
start_demo_time = time.time()
while st.session_state.monitoring and (time.time() - start_demo_time) < SIMULATED_SECONDS:
    # Simulated heart rate (HR) and respiratory rate (RR)
    new_hr = np.random.randint(60, 100)
    new_rr = np.random.randint(10, 20)

    # Determine sleep stage
    sleep_stage = determine_sleep_stage(new_hr, new_rr)

    # Append new data
    sim_time = datetime.now() + timedelta(seconds=len(st.session_state.timestamps) * TIME_MULTIPLIER)
    st.session_state.hr_data.append(new_hr)
    st.session_state.rr_data.append(new_rr)
    st.session_state.timestamps.append(sim_time)
    st.session_state.sleep_stages.append(sleep_stage)

    # **Plot Heart Rate & Respiratory Rate**
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(st.session_state.timestamps, st.session_state.hr_data, label="Heart Rate (BPM)", color="red", linewidth=2)
    ax.plot(st.session_state.timestamps, st.session_state.rr_data, label="Respiratory Rate (Breaths/min)", color="blue", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend()
    ax.grid()
    chart_placeholder.pyplot(fig)

    # **Plot Sleep Stage Graph**
    stage_order = {"Awake": 0, "Light Sleep": 1, "Deep Sleep": 2, "REM Sleep": 3}
    stage_colors = {0: "red", 1: "orange", 2: "blue", 3: "purple"}

    sleep_stage_values = [stage_order[stage] for stage in st.session_state.sleep_stages]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.step(st.session_state.timestamps, sleep_stage_values, where="post", linewidth=2, color="black")

    for stage, value in stage_order.items():
        ax2.fill_between(st.session_state.timestamps, value - 0.5, value + 0.5, color=stage_colors[value], alpha=0.3, label=stage)

    ax2.set_yticks(list(stage_order.values()))
    ax2.set_yticklabels(list(stage_order.keys()))
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Sleep Stage")
    ax2.legend()
    ax2.grid()

    sleep_stage_placeholder.pyplot(fig2)

    # **Health Alerts**
    alert_msg = None
    if new_hr > 90:
        alert_msg = "⚠️ High Blood Pressure Detected! Consult a doctor."
    elif new_rr < 10:
        alert_msg = "⚠️ Possible Sleep Apnea Detected! Consider medical evaluation."

    if alert_msg:
        st.session_state.alerts.append((sim_time.strftime("%H:%M:%S"), alert_msg))

    if st.session_state.alerts:
        latest_alert = st.session_state.alerts[-1]
        alert_placeholder.error(f"{latest_alert[1]} (Time: {latest_alert[0]})")
    else:
        alert_placeholder.success("✅ Normal Sleep & Cardiovascular Health")

    time.sleep(1)

# Stop Monitoring after 30 seconds
st.session_state.monitoring = False

# =============================
# 3. Sleep Report Summary
# =============================
if not st.session_state.monitoring and st.session_state.start_time:
    if st.session_state.hr_data:
        elapsed_time = REAL_SLEEP_HOURS  # Representing 8 hours of sleep
        avg_hr = np.mean(st.session_state.hr_data)
        max_hr = np.max(st.session_state.hr_data)
        min_hr = np.min(st.session_state.hr_data)
        avg_rr = np.mean(st.session_state.rr_data)
        sleep_stage_counts = pd.Series(st.session_state.sleep_stages).value_counts()

        alert_logs = pd.DataFrame(st.session_state.alerts, columns=["Time", "Alert"]) if st.session_state.alerts else None

        report_placeholder.write("## 💤 Sleep & Blood Pressure Report")
        report_placeholder.write(f"**📅 Simulated Sleep Duration:** {elapsed_time:.1f} hours")
        report_placeholder.write(f"**❤️ Avg Heart Rate:** {avg_hr:.1f} BPM (Min: {min_hr} | Max: {max_hr})")
        report_placeholder.write(f"**💨 Avg Respiratory Rate:** {avg_rr:.1f} Breaths/min")
        report_placeholder.write("### 🛏️ Sleep Stages Summary:")
        report_placeholder.write(sleep_stage_counts.to_string())

        if alert_logs is not None:
            report_placeholder.write("### 🚨 Health Alerts During Sleep:")
            report_placeholder.dataframe(alert_logs)

        st.success("✅ Sleep monitoring completed! Summary report generated.")
