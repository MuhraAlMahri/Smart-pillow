import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import time

# =============================
# 1. Simulating Health Data (Heart Rate, Respiratory Rate)
# =============================
def generate_synthetic_data(length=300):
    time = np.linspace(0, length, length)
    heart_rate = 70 + 5 * np.sin(0.1 * time)  # Simulated heart rate
    respiratory_rate = 15 + 2 * np.sin(0.05 * time)  # Simulated respiratory rate
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

# =============================
# 3. Streamlit Interactive Dashboard
# =============================
st.title("🛏️ AI-Powered Smart Pillow Health Monitor")

st.write("### 📊 Real-Time Health Data Monitoring")

# **Initialize session state variables for real-time updates**
if "heart_rate" not in st.session_state:
    st.session_state.heart_rate = 75
if "respiratory_rate" not in st.session_state:
    st.session_state.respiratory_rate = 15

# **User-controlled sliders**
st.session_state.heart_rate = st.slider("💓 Heart Rate (BPM)", min_value=50, max_value=120, value=st.session_state.heart_rate)
st.session_state.respiratory_rate = st.slider("🌬️ Respiratory Rate (Breaths per Min)", min_value=5, max_value=30, value=st.session_state.respiratory_rate)

# **Real-time Update Simulation**
if st.button("🔄 Simulate Real-Time Updates"):
    st.session_state.heart_rate += np.random.randint(-2, 3)
    st.session_state.respiratory_rate += np.random.randint(-1, 2)

# **Display Live Data Charts**
st.line_chart(np.random.randint(60, 100, size=50))  # Simulated HR chart
st.line_chart(np.random.randint(10, 25, size=50))   # Simulated RR chart

# **AI Health Warnings**
st.write("### 🏥 AI-Based Health Insights")
if st.session_state.heart_rate > 90:
    st.error("⚠️ High Blood Pressure Detected! Consult a doctor.")
elif st.session_state.respiratory_rate < 10:
    st.warning("⚠️ Possible Sleep Apnea Detected! Consider medical evaluation.")
else:
    st.success("✅ Normal Cardiovascular & Respiratory Health")
