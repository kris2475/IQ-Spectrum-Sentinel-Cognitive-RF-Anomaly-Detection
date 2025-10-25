# Spectrum Sentinel: Cognitive RF Anomaly Detection

## 🛡️ Project Overview

**Spectrum Sentinel** is a hobby project that combines **Software-Defined Radio (SDR)** with **Unsupervised Deep Learning** to explore real-time, autonomous spectrum awareness.

The system employs a **Nooelec NESDR Smart** SDR to capture wideband RF data, converts it into spectrograms, and trains an **autoencoder** to identify signals that deviate from the “normal” spectrum. These anomalies may represent **unknown transmitters**, **interference**, or other unusual emissions.

### Layman’s Summary

Imagine being able to “listen” to all the radio signals around you. Most of the time, the airwaves are predictable — Wi-Fi, television, radio, and so on. Spectrum Sentinel learns what’s normal, so when something unusual appears — such as a new signal — it can flag it.  

**Spectrograms** are images that represent **frequency over time**. Bright areas indicate strong signals, while darker areas are quieter. In essence, this process converts invisible radio signals into pictures a computer can understand.

---

## 🚀 Key Capabilities

* **Zero-Day Threat Detection:** Identifies previously unseen signals through anomaly detection.  
* **Blind Signal Classification:** Automatically groups unknown signals into clusters.  
* **Cognitive Architecture:** Adapts its monitoring behaviour based on what it learns in real time.  
* **High-Fidelity Feature Extraction:** Utilises deep learning to extract meaningful signal characteristics.  

---

## 🛠️ Technical Architecture

Three layers: Acquisition → Feature Engineering → Cognitive Processing.

### 1. Acquisition Layer (SDR Front End)

| Component | Technology | Function |
| :--- | :--- | :--- |
| RF Front End | Nooelec NESDR Smart | Captures raw I/Q time-series RF data. |
| Digitisation | ADC | Converts analogue RF signals into digital form for processing. |

### 2. Feature Engineering Layer

| Component | Method | Function |
| :--- | :--- | :--- |
| Time-Frequency Transform | STFT | Converts raw I/Q data into 2D **Spectrograms**. |
| Normalisation | Data Scaling | Ensures stable neural network training. |

### 3. Cognitive Processing Layer

#### Autoencoder for Anomaly Detection
* **Model:** Convolutional Autoencoder (CAE)  
* **Training:** Trained solely on “normal” RF data.  
* **Detection:** High reconstruction loss indicates anomalies.

#### Clustering for Blind Classification
* **Input Feature:** Latent vector from the CAE.  
* **Algorithm:** DBSCAN or K-Means.  
* **Function:** Automatically groups similar signals.

---

## 💻 Installation and Setup

### Prerequisites

* Python 3.8+  
* SDR hardware (Nooelec NESDR Smart or equivalent)  
* GPU recommended for model training  

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

Example output:

| Timestamp | Anomaly Score | Freq (MHz) | Duration (s) | Status | Cluster ID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10:45:01 | 1.21 | 902.1 | 0.8 | HIGH ANOMALY | NEW (A) |
| 10:45:03 | 0.04 | 450.5 | 1.2 | NORMAL | FRIENDLY (C) |

---

## 📊 Spectrogram Visualisation

Spectrograms are **2D images**:  
* **X-axis:** Time  
* **Y-axis:** Frequency  
* **Colour intensity:** Signal strength  

Bright areas indicate strong signals; dark areas represent weak or absent signals. This enables both humans and machine learning models to “see” RF activity.

---

## 🗓 Notes

* Only strong signals may optionally be saved as images.  
* Filenames include timestamps for traceability.  
* Can be adapted for different SDR hardware or frequency ranges.  
* Suitable for training deep learning models on real-world spectrum data.  
