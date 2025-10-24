# Spectrum Sentinel: Cognitive RF Anomaly Detection

## 🛡️ Project Overview

**Spectrum Sentinel** is a next-generation Electronic Warfare (EW) and Signals Intelligence (SIGINT) framework leveraging **Software-Defined Radio (SDR)** and **Unsupervised Deep Learning** to achieve real-time, autonomous spectrum awareness.

The core mission is to rapidly detect and classify **"zero-day" or novel threat emitters**—such as adaptive jammers, clandestine communication links, and dynamic protocols—that would bypass traditional, library-dependent EW systems. By using unsupervised techniques like Autoencoders and Clustering, the system can automatically build a comprehensive intelligence picture of an unknown electromagnetic environment.

---

## 🚀 Key Capabilities

* **Zero-Day Threat Detection:** Instantly flags unknown and novel signals by identifying anomalies in the Radio Frequency (RF) spectrum using Reconstruction Loss.
* **Blind Signal Classification:** Automatically clusters similar, uncataloged signals into logical groups, enabling rapid Pattern-of-Life (PoL) analysis and threat prioritization.
* **Cognitive Architecture:** Built on SDR principles, allowing for dynamic, software-driven re-configuration of the acquisition and processing chain based on real-time ML inference.
* **High-Fidelity Feature Extraction:** Utilizes the latent space of a Deep Autoencoder to generate robust, low-dimensional feature vectors that represent the signal's core characteristics, making clustering highly effective.

---

## 🛠️ Technical Architecture

The architecture follows a three-layer pipeline: Acquisition, Feature Engineering, and Cognitive Processing.

### 1. Acquisition Layer (SDR Front-End)

| Component | Technology | Function |
| :--- | :--- | :--- |
| **RF Front-End** | Commercial Off-The-Shelf (COTS) SDR (e.g., USRP, BladeRF) | Wideband, high-speed capture of raw In-phase and Quadrature (I/Q) time-series data. |
| **Digitization** | High-speed ADC | Converts analog RF to digital I/Q streams for processing. |

### 2. Feature Engineering Layer (Pre-Processing)

| Component | Method | Function |
| :--- | :--- | :--- |
| **Time-Frequency Transform** | Short-Time Fourier Transform (STFT) | Converts raw I/Q time-series data into 2D **Spectrograms** (Time vs. Frequency vs. Power), preparing the data for image-based deep learning. |
| **Normalization** | Data Scaling | Normalizes spectrogram pixels to ensure stable training of the neural network. |

### 3. Cognitive Processing Layer (Unsupervised Deep Learning)

#### **A. Autoencoder for Anomaly Detection**

* **Model:** Convolutional Autoencoder (CAE)
* **Training:** Trained **only** on a large corpus of **known-normal** spectrum activity (friendly comms, expected noise, civilian traffic).
* **Mechanism:** An input spectrogram ($X$) is compressed into a latent vector ($Z$) by the **Encoder ($E$)** and then reconstructed ($\hat{X}$) by the **Decoder ($D$)**.
* **Detection:** Anomaly Score is the **Reconstruction Loss ($L_{\text{recon}}$)**. A novel threat signal, which the model has not learned, results in a significantly higher $L_{\text{recon}}$ score, instantly flagging it as an unknown threat.
    $$L_{\text{recon}} = ||X - \hat{X}||^2 \approx \text{MSE}$$

#### **B. Clustering for Blind Classification**

* **Input Feature:** The latent vector ($Z$) output from the trained Encoder ($E$).
* **Algorithm:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) or K-Means.
* **Function:** Groups the latent vectors ($Z$) of incoming signals based on their proximity in the feature space. Each resulting cluster represents a distinct, automatically-identified signal type, forming a real-time catalog of the environment.

---

## 💻 Installation and Setup

### Prerequisites

* Python 3.8+
* SDR Hardware (with drivers installed)
* GPU recommended for rapid deep learning inference.

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/YourOrg/spectrum-sentinel.git
cd spectrum-sentinel

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required Python packages
pip install -r requirements.txt
```

### Key Dependencies

| Library | Purpose |
| :--- | :--- |
| TensorFlow / PyTorch | Deep Learning framework for the CAE. |
| NumPy / SciPy | Core numerical and signal processing (STFT). |
| scikit-learn | DBSCAN and K-Means clustering implementation. |
| pysdr / gnuradio | SDR hardware interfacing and data capture utility. |

---

## ⚙️ Usage

### 1. Training the Autoencoder

First, you must train the CAE on your specific environment's "normal" data.

```bash
# Acquire a large dataset of normal I/Q data and store as spectrograms
python scripts/01_capture_normal_data.py --duration 3600 --freq 2400e6

# Train the CAE model 
# (This step is computationally intensive and should use a GPU)
python scripts/02_train_autoencoder.py --dataset data/normal_spectrograms/ --epochs 50 
```

### 2. Real-Time Anomaly Monitoring

Execute the Sentinel module for continuous spectrum monitoring.

```bash
# The system will continuously monitor the spectrum and log anomalies
python run_sentinel.py --model_path models/cae_final.h5 --threshold 0.5 
```

#### Output Log Example:

| Timestamp | Anomaly Score ($L_{recon}$) | Freq (MHz) | Duration (s) | Status | Cluster ID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10:45:01 | 1.21 | 902.1 | 0.8 | HIGH ANOMALY | NEW (A) |
| 10:45:03 | 0.04 | 450.5 | 1.2 | NORMAL | FRIENDLY (C) |
| 10:45:05 | 0.95 | 902.1 | 0.9 | HIGH ANOMALY | NEW (A) |

---
