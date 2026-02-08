# 😴 Sleep Quality Estimation Using Wearable Sensors

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io)

A **Cyber-Physical System (CPS)** and **Machine Learning** project designed to provide real-time, privacy-focused sleep quality estimation. This system integrates hardware sensing with a Random Forest model to process physiological data locally (Fog Computing), delivering an interpretable sleep score from 0-100.

---

## 🚀 Project Overview

Traditional sleep monitoring often relies on cloud-based processing, leading to latency issues and privacy concerns. This project addresses these gaps by:

| Feature | Description |
|---------|-------------|
| 🔒 **Local Processing (Fog Node)** | Uses a laptop as a processing hub to ensure data privacy and low-latency feedback |
| ⚡ **Real-Time Sensing** | Captures raw PPG (Photoplethysmogram) data via an Arduino-integrated pulse sensor |
| 🧠 **Interpretable AI** | Implements a Random Forest Classifier to categorize sleep quality and identify key physiological drivers |

---

## 🏗️ System Architecture

The project is structured into three distinct layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CYBER/FOG LAYER                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │
│  │ Preprocessing│ → │  ML Model   │ → │  Streamlit Dashboard    │   │
│  │ (HeartPy)   │   │(RandomForest)│   │  (Real-time Visuals)   │   │
│  └─────────────┘   └─────────────┘   └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              ↑
                    USB Serial (PySerial)
                              ↑
┌─────────────────────────────────────────────────────────────────────┐
│                      PHYSICAL LAYER                                 │
│           ┌──────────────────────────────────────┐                 │
│           │  Arduino UNO + PPG Pulse Sensor      │                 │
│           │  (Raw Physiological Data Acquisition)│                 │
│           └──────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer Details:**
1. **Physical Layer:** PPG Pulse Sensor + Arduino UNO for raw physiological data acquisition
2. **Communication Layer:** USB Serial bridge using `PySerial` for high-speed data transmission
3. **Cyber/Fog Layer:**
   - **Preprocessing:** Feature extraction (BPM, HRV) from raw signals
   - **Inference:** A pre-trained Random Forest model for classification and scoring
   - **Visualization:** A Streamlit-based real-time dashboard

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Hardware** | Arduino UNO R3, PPG Pulse Sensor, LEDs, Resistors |
| **Languages** | Python 3.9+, C++ (Arduino Sketch) |
| **ML Framework** | Scikit-learn (Random Forest) |
| **Visualization** | Streamlit |
| **Key Libraries** | `pandas`, `joblib`, `pyserial`, `heartpy`, `numpy`, `matplotlib` |

---

## 📂 Project Structure

```
Sleep_Quality_Project/
├── 📁 data/                    # Training datasets
│   └── Sleep_health_and_lifestyle_dataset.csv
├── 📁 firmware/                # Arduino source code (.ino)
├── 📁 models/                  # Saved Machine Learning models
│   └── sleep_quality_model.pkl
├── 📁 scripts/                 # Python scripts for ML and Dashboard
│   ├── train.py               # Model training and feature importance
│   └── dashboard.py           # Streamlit real-time interface
├── 📄 requirements.txt        # Python dependencies
├── 📄 .gitignore              # Git ignore rules
├── 📄 LICENSE                 # MIT License
└── 📄 README.md               # Project documentation (you are here!)
```

---

## ⚙️ Installation & Setup

### 1. Hardware Connection

1. Connect the **PPG Pulse Sensor** to the Arduino:
   - `Signal` → `A0`
   - `VCC` → `5V`
   - `GND` → `GND`

2. Connect the Arduino to your laptop via USB

### 2. Software Installation

**Clone the repository:**
```bash
git clone https://github.com/GuruMohith24/Sleep_Quality_Project.git
cd Sleep_Quality_Project
```

**Set up Virtual Environment:**
```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Mac/Linux:
source .venv/bin/activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📈 Usage

### 1. Upload Firmware
Upload the `firmware/ppg_sensor.ino` to your Arduino using the Arduino IDE.

### 2. Train Model (Optional)
Run the training script to see feature importance and save the model:
```bash
python scripts/train.py
```

**Expected Output:**
```
Loaded data from: .../data/Sleep_health_and_lifestyle_dataset.csv

--- Feature Importance ---
Sleep Duration              0.35
Heart Rate                  0.30
Physical Activity Level     0.20
Daily Steps                 0.10
Age                         0.05

Model saved to: .../models/sleep_quality_model.pkl
```

### 3. Run Dashboard
Launch the real-time Streamlit interface:
```bash
streamlit run scripts/dashboard.py
```

---

## 🧠 Machine Learning Insights

The model uses a **Random Forest Classifier** which provides high robustness against noisy sensor data.

### Feature Importance

| Feature | Weight | Description |
|---------|--------|-------------|
| 🛏️ **Sleep Duration** | ~35% | Primary predictor of sleep quality |
| ❤️ **Heart Rate** | ~30% | Key physiological indicator |
| 🏃 **Physical Activity** | ~20% | Daily exercise impact |
| 👣 **Daily Steps** | ~10% | Movement patterns |
| 📅 **Age** | ~5% | Demographic factor |

### Model Output
- **Classification:** Binary (Good/Poor Sleep Quality)
- **Score:** Heuristic score from 0-100
- **Threshold:** Sleep quality ≥ 7 classified as "Good"

---

## 🔬 Why This Approach?

| Challenge | Our Solution |
|-----------|--------------|
| **Privacy Concerns** | All processing happens locally on the Fog Node |
| **Latency Issues** | Edge computing eliminates cloud round-trips |
| **Interpretability** | Random Forest provides feature importance insights |
| **Cost** | Low-cost Arduino + open-source software stack |

---

## 🎯 Future Enhancements

- [ ] Add LSTM/RNN for temporal pattern recognition
- [ ] Integrate SpO2 sensor for oxygen saturation monitoring
- [ ] Mobile app for remote monitoring
- [ ] Sleep stage classification (REM, Deep, Light)

---

## 👨‍💻 Author

**Guru Mohith**
- GitHub: [@GuruMohith24](https://github.com/GuruMohith24)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for better sleep quality monitoring
</p>
