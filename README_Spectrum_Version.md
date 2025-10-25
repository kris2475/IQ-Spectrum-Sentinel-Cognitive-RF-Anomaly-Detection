# 🛰️ Spectrum Sentinel: Cognitive RF Anomaly Detection 🛡️

**Spectrum Sentinel** is a hobby project that combines **Software-Defined Radio (SDR)** with **Unsupervised Deep Learning** to explore real-time, autonomous spectrum awareness.

The system employs a **Nooelec NESDR Smart SDR** to capture wideband RF data, converts it into **spectrograms**, and trains an **autoencoder** to identify signals that deviate from the “normal” spectrum.  
These anomalies may represent unknown transmitters, interference, or other unusual emissions.

---

## 📈 Current Development Focus: Spectrogram-First Data Collection

The project has transitioned to a **Spectrogram-First** approach for data capture and model training.

### 🧠 Background
Initially, Spectrum Sentinel stored and processed raw **I/Q time-series** data.  
While comprehensive, this method was inefficient and complex for training deep learning models.

The new **Spectrogram-First** pipeline saves the **2D spectrogram matrix** for every sweep, which serves as the optimal feature set for deep learning.

---

## 🎯 Why the Spectrogram-First Approach?

### ⚙️ Computational Efficiency
Training deep learning models (**2D CNNs**) on spectrogram matrices is far more efficient than using high-dimensional raw **1D I/Q** data — reducing **GPU memory usage** and **training time**.

### 🎨 Optimised Feature Extraction
The **Short-Time Fourier Transform (STFT)** generates spectrograms that:
- Reduce noise  
- Standardise input dimensions  
- Highlight key features (frequency, time, and power)

### ⚖️ Unbiased Training Data
The autoencoder learns from the *entire* spectrum of "normal" data (including quiet noise).  
All spectrograms are saved — ensuring a **complete, unbiased** dataset.

---

## 👂 Layman’s Summary

Imagine being able to “listen” to all the radio signals around you.  
Most of the time, the airwaves are predictable — **Wi-Fi**, **television**, **radio**, and so on.

Spectrum Sentinel learns what’s *normal*. When something unusual appears — like a new or unknown signal — it can **flag it** automatically.

Spectrograms are **images** representing **frequency over time**:
- **Bright** areas → strong signals  
- **Dark** areas → quieter regions  

This converts invisible radio energy into pictures that a computer (and you) can understand.

---

## 🚀 Key Capabilities

| Capability | Description |
|-------------|-------------|
| 🧩 Zero-Day Threat Detection | Identifies previously unseen or unknown signals. |
| 🔍 Blind Signal Classification | Automatically groups unknown signals using clustering. |
| 🧠 Cognitive Architecture | Adapts monitoring based on learned patterns. |
| 🎯 High-Fidelity Feature Extraction | Uses deep learning to extract meaningful RF features. |

---

## 🧱 Technical Architecture

The system is composed of **three layers**:

### 1. Acquisition Layer (SDR Front End)

| Component | Technology | Function |
|------------|-------------|-----------|
| RF Front End | Nooelec NESDR Smart | Captures raw I/Q RF data. |
| Digitisation | ADC | Converts analogue RF to digital data. |

### 2. Feature Engineering Layer

| Component | Method | Function |
|------------|---------|-----------|
| Time-Frequency Transform | STFT (Spectrogram Generation) | Converts I/Q data into 2D spectrograms. |
| Normalisation | Data Scaling | Ensures stable neural network training. |

### 3. Cognitive Processing Layer

#### Autoencoder for Anomaly Detection
- **Model:** Convolutional Autoencoder (CAE)  
- **Training:** Only on “normal” RF data  
- **Detection:** High reconstruction loss ⇒ anomaly detected  

#### Clustering for Blind Classification
- **Input Feature:** Latent vector from CAE  
- **Algorithm:** DBSCAN / K-Means  
- **Function:** Groups similar signals for categorisation  

---

## 🧩 Technical Deep Dive: Conv2D Autoencoder Workflow

### 🧱 1. Data Loading and Preparation
- **Google Drive Mounting:** Accessed compressed spectrogram datasets directly from Drive (`/content/drive/My Drive/ZIP/`).  
- **Training vs. Testing Data:**  
  - Training: All `.zip` files except `spectrograms_anomolous.zip`  
  - Testing: `spectrograms_anomolous.zip` (anomalous samples)  
- **Extraction Directories:**  
  - `/tmp/extracted_training_data`  
  - `/tmp/extracted_testing_data`  
  Cleaned and recreated to ensure no data overlap.  
- **Verification:** Confirmed that all `.npy` spectrograms were correctly extracted and separated.

---

### ⚙️ 2. Data Handling for Training
To prevent memory overflow, a **custom data generator** (`SpectrogramDataGenerator`) was implemented:
- **On-the-fly loading:** Streams `.npy` spectrograms in mini-batches instead of loading all at once.  
- **Shape Consistency:** Automatically infers input dimensions (e.g. `(1024, 292, 1)`) from the dataset.  
- **Memory Efficiency:** Reduced RAM usage drastically — stable even with thousands of spectrograms.  
- **Generators:** Created for both training and testing datasets, with shuffling enabled for training.

---

### 🧠 3. Autoencoder Model Definition (Conv2D)
- **Architecture:** A **2D Convolutional Autoencoder** designed for spectrogram data.  
- **Encoder:** Stacked `Conv2D` and `MaxPooling2D` layers for hierarchical compression.  
- **Decoder:** Uses `UpSampling2D` and `Cropping2D` to reconstruct input size exactly.  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Input Shape:** `(1024, 292, 1)` — a single-channel (grayscale) spectrogram image.  

---

### 🧪 4. Model Training
- **Training Mode:** Utilised the generator to stream batches to the model.  
- **Validation:** The anomalous dataset was used for validation to monitor reconstruction accuracy on unseen patterns.  
- **Performance (CPU vs TPU):**  
  - CPU runtime: ~62 sec/step  
  - TPU runtime: ~544 ms/step  
  → **>100× speed improvement** using TPU acceleration.  

---

### 🔍 5. Model Evaluation and Anomaly Detection
- **Reconstruction Error:** Calculated MSE between input and reconstructed spectrograms.  
- **Histogram Visualization:** Showed distribution of reconstruction errors to identify natural thresholds.  
- **Error Metrics:** Computed mean, median, std, max, and min to summarize error spread.  
- **Anomaly Threshold:** Set as `mean + 2×std`. Samples above threshold were marked as anomalies.  
- **Anomaly Mapping:** Each anomaly’s index was mapped back to its original `.npy` file for identification.  
- **Ranking:** Anomalies were ranked by reconstruction error (descending).  
- **Visualization:** Plotted original vs reconstructed spectrograms to qualitatively assess model behavior.  

---

### 📊 Summary of Findings
- Successfully trained a **Conv2D Autoencoder** for anomaly detection in spectrogram data.  
- **Data Generators** enabled large-scale training without RAM overload.  
- **TPU acceleration** provided a massive runtime improvement.  
- Detected and ranked **anomalous spectrograms** with traceability to original files.  
- Future improvements may include **frequency localization** of anomaly sources and **latent space clustering** for deeper insight.  

---

## 💻 Installation and Setup

### 🧩 Prerequisites
- Python **3.8+**
- SDR hardware (**Nooelec NESDR Smart** or equivalent)
- **GPU or TPU** (recommended for model training)

### 🧱 Environment Setup

```bash
git clone https://github.com/YourOrg/spectrum-sentinel.git
cd spectrum-sentinel

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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

#### Example Output

| Timestamp | Anomaly Score | Freq (MHz) | Duration (s) | Status | Cluster ID |
|------------|----------------|-------------|----------------|----------|-------------|
| 10:45:01 | **1.21** | 902.1 | 0.8 | 🔴 HIGH ANOMALY | NEW (A) |
| 10:45:03 | 0.04 | 450.5 | 1.2 | 🟢 NORMAL | FRIENDLY (C) |

---

## 📊 Spectrogram Visualisation

Spectrograms are **2D images**:
- **X-axis:** Time  
- **Y-axis:** Frequency  
- **Color intensity:** Signal strength  

Bright regions = strong signals  
Dark regions = weak or absent signals  

Both humans and AI models can “see” RF activity visually.

---

## 🗓 Notes

- All spectrogram data is saved as **NumPy arrays** for training.  
- Spectrogram **images** (e.g., `.png`) are optionally saved for strong signals.  
- **Timestamps** are embedded in filenames for traceability.  
- Supports different **SDR hardware** and **frequency ranges**.  
- Suitable for real-world **deep learning** experiments on spectrum data.

---

## 📚 License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Pull requests and improvements are welcome!  
Feel free to open an issue or contribute enhancements to:
- Data preprocessing  
- Model performance  
- Real-time inference  

---

## 🧭 Roadmap

- [ ] Real-time dashboard visualisation  
- [ ] Extended clustering analysis  
- [ ] Multi-SDR distributed monitoring  
- [ ] Edge inference optimisation  

---

> **Spectrum Sentinel** — Cognitive RF awareness through deep learning and signal intelligence.
