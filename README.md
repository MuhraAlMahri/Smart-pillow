# 📌 AI-Powered Smart Pillow Health Monitor

🚀 Real-Time Sleep & Blood Pressure Monitoring with AI

📖 Overview

This project is an AI-powered smart pillow health monitor built using Streamlit, PyTorch, and Matplotlib. It provides real-time tracking of heart rate (HR), respiratory rate (RR), and sleep stages to detect potential health issues like sleep apnea and high blood pressure.

🛠 Features

✅ Real-time health monitoring of heart rate and breathing rate.
✅ Live updating charts for HR and RR.
✅ Sleep stage tracking (Awake, Light Sleep, Deep Sleep, REM).
✅ Health alerts for high blood pressure and sleep apnea detection.
✅ Automatic sleep summary report after stopping the monitoring.
✅ Fast sleep cycle demo (5 minutes) for testing instead of 8 hours.


📦 Installation

🔹 Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries.

🔹 Clone the Repository

git clone https://github.com/yourusername/smart-pillow-monitor.git
cd smart-pillow-monitor

🔹 Install Dependencies
pip install -r requirements.txt

🚀 Usage

🔹 Running the App Locally
streamlit run streamlit_app.py

This will open the Streamlit dashboard in your browser.

🔹 Start Monitoring
	1.	Click “🚀 Start Sleep Monitoring” to begin real-time tracking.
	2.	The heart rate and breathing rate graph will update live.
	3.	Sleep stages will transition (Awake → Light → Deep → REM).
	4.	Alerts will appear if any health risks are detected.

🔹 Stop Monitoring & Get Sleep Report
	•	Click “⏹ Stop Monitoring” at any time.
	•	A summary report of sleep duration, heart rate trends, and sleep stages will be generated.
 
📊 Demo: Fast Sleep Simulation (5 Minutes)

For demo purposes, this version speeds up sleep monitoring:
	•	1 second = 1 minute of real sleep.
	•	A full sleep cycle (Light → Deep → REM → Awake) completes in 5 minutes.
	•	Auto-stops after 5 minutes instead of 8 hours.

 🖥️ Technologies Used

🔹 Python
🔹 Streamlit (for the dashboard UI)
🔹 PyTorch (for AI-based sleep stage detection)
🔹 Matplotlib (for live data visualization)
🔹 NumPy & Pandas (for data processing)

📜 License

This project is licensed under the MIT License. Feel free to modify and contribute!


🤝 Contributing

Want to improve this project? Contributions are welcome! 🚀
	1.	Fork the repository.
	2.	Create a new branch: git checkout -b feature-name
	3.	Commit changes: git commit -m "Add new feature"
	4.	Push to the branch: git push origin feature-name
	5.	Open a pull request.
