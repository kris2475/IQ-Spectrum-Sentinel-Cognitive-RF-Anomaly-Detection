# Spectrum Sentinel: Cognitive RF Anomaly Detection

## 🛡️ Project Overview

**Spectrum Sentinel** is a hobby project that mixes **Software-Defined Radio (SDR)** with **Unsupervised Deep Learning** to experiment with real-time, autonomous spectrum awareness.

The system uses a **Nooelec NESDR Smart** SDR to capture wideband RF data, converts it into spectrograms, and trains an **autoencoder** to spot signals that do not match the "normal" spectrum. These anomalies might represent **unknown transmitters**, **interference**, or other unusual signals.

### Layman’s Summary

Imagine “listening” to all the radio signals around you. Most of the time, the airwaves are predictable—Wi-Fi, TV, radio, etc. Spectrum Sentinel learns what’s normal, so when something unusual appears—like a new signal—it can flag it.  

**Spectrograms** are images that represent **frequency over time**. Bright spots show strong signals; dark areas are quiet. Essentially, it turns invisible radio signals into pictures that a computer can understand.

---

## 🚀 Key Capabilities

* **Zero-Day Threat Detection:** Flags unknown signals using anomaly detection.
* **Blind Signal Classification:** Clusters unknown signals into groups automatically.
* **Cognitive Architecture:** Can adapt its monitoring based on what it learns in real-time.
* **High-Fidelity Feature Extraction:** Uses deep learning to extract meaningful signal features.

---

## 🛠️ Technical Architecture

Three layers: Acquisition → Feature Engineering → Cognitive Processing.

### 1. Acquisition Layer (SDR Front-End)

| Component | Technology | Function |
| :--- | :--- | :--- |
| RF Front-End | Nooelec NESDR Smart | Captures raw I/Q time-series RF data. |
| Digitization | ADC | Converts analog RF to digital for processing. |

### 2. Feature Engineering Layer

| Component | Method | Function |
| :--- | :--- | :--- |
| Time-Frequency Transform | STFT | Converts raw I/Q data into 2D **Spectrograms**. |
| Normalization | Data Scaling | Ensures stable neural network training. |

### 3. Cognitive Processing Layer

#### Autoencoder for Anomaly Detection
* **Model:** Convolutional Autoencoder (CAE)
* **Training:** On “normal” RF data only.
* **Detection:** High reconstruction loss flags anomalies.

#### Clustering for Blind Classification
* **Input Feature:** Latent vector from CAE.
* **Algorithm:** DBSCAN or K-Means.
* **Function:** Groups similar signals automatically.

---

## 💻 Installation and Setup

### Prerequisites

* Python 3.8+
* SDR Hardware (Nooelec NESDR Smart or similar)
* GPU recommended for training

### Environment Setup

```bash
git clone https://github.com/YourOrg/spectrum-sentinel.git
cd spectrum-sentinel

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```

---

## ⚙️ Usage

### 1. Training the Autoencoder

```bash
python scripts/01_capture_normal_data.py --duration 3600 --freq 2400e6
python scripts/02_train_autoencoder.py --dataset data/normal_spectrograms/ --epochs 50
```

### 2. Real-Time Anomaly Monitoring

```bash
python run_sentinel.py --model_path models/cae_final.h5 --threshold 0.5
```

Output example:

| Timestamp | Anomaly Score | Freq (MHz) | Duration (s) | Status | Cluster ID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10:45:01 | 1.21 | 902.1 | 0.8 | HIGH ANOMALY | NEW (A) |
| 10:45:03 | 0.04 | 450.5 | 1.2 | NORMAL | FRIENDLY (C) |

---

## 📊 Spectrogram Visualisation

Spectrograms are **2D images**:  
* **X-axis:** Time  
* **Y-axis:** Frequency  
* **Color intensity:** Signal strength  

Bright spots = strong signals. Dark = weak or no signal. This allows humans and ML models to “see” RF activity.

---

## 🗓 Notes

* Only strong signals can optionally be saved as images.
* Filenames include timestamps for traceability.
* Can be adapted to different SDR hardware or frequency ranges.
* Suitable for training deep learning models on real-world spectrum data.
