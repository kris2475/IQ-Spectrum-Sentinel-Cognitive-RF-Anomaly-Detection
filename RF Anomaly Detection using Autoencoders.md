# RF Anomaly Detection using Autoencoders

## Overview

This project demonstrates how to identify unusual patterns — or **anomalies** — within radio frequency (RF) spectrum data using an **unsupervised deep learning model** known as an **autoencoder**.

The goal is straightforward:  
Train a model to understand what *normal* RF data looks like, and then use that understanding to flag anything that significantly deviates from the norm.

---

## How It Works

### 1. Data Preparation: Extraction, Loading, and Preprocessing

1. **Extraction**  
   The process begins with unpacking your compressed RF data archive (`scan4.zip`).  
   Inside are multiple `.npy` files, each representing a single RF spectrum sample.

2. **Loading**  
   Each file is loaded as a one-dimensional NumPy array containing complex numbers (`complex128`), representing both the amplitude and phase of the RF signal.

3. **Preprocessing**  
   Since most machine learning models only work with real numbers, we convert the complex values to their **magnitudes**.  
   The magnitude represents the signal strength at each point, creating a clean, real-valued dataset ready for modelling.

---

### 2. The Autoencoder Model

An **autoencoder** is a neural network designed to **reconstruct its input** after compressing it through a smaller internal representation.

- **Encoder:** Compresses the input data into a lower-dimensional “bottleneck” representation.  
- **Decoder:** Attempts to reconstruct the original input from this compressed form.

During training, the autoencoder learns to minimise the difference between the original input and the reconstructed output.  
By training exclusively on *normal* RF data, it effectively learns what “normal” looks like in your spectrum.

---

### 3. Detecting Anomalies with Reconstruction Loss

After training, we run all samples through the autoencoder and calculate the **reconstruction loss** for each one — typically using **Mean Squared Error (MSE)**:

- **Low reconstruction loss:** The model recreated the input accurately → likely *normal*.  
- **High reconstruction loss:** The model struggled to recreate it → likely *anomalous*.

A **threshold** is then set (for example, based on the mean and standard deviation of the losses).  
Any sample whose reconstruction loss exceeds this threshold is flagged as a potential anomaly.

---

### 4. Visualising the Results

Visualisation helps confirm how well the model distinguishes normal from abnormal data:

- **Histogram of Reconstruction Losses:**  
  Shows the distribution of loss values. Most samples cluster near low losses (normal), while a few stand out in the high-loss “tail” (anomalies).

- **Scatter Plot of Reconstruction Loss per Sample:**  
  Displays loss values sequentially, with high-loss points clearly separated from the rest. A threshold line marks the boundary between normal and anomalous data.

- **Original vs. Reconstructed Signal Comparison:**  
  For normal samples, the reconstructed signal closely matches the original.  
  For anomalous samples, the reconstruction shows noticeable distortion, confirming that the pattern was unfamiliar to the model.

---

## Summary of the Process

| Step | Description | Purpose |
|------|--------------|----------|
| **1. Extract & Preprocess** | Load `.npy` files and convert complex data to real magnitudes | Prepare clean input data |
| **2. Train Autoencoder** | Learn a compressed representation of normal RF signals | Build a baseline understanding |
| **3. Compute Reconstruction Loss** | Measure how well each sample is reconstructed | Quantify deviations from normal |
| **4. Apply Threshold** | Flag samples with unusually high losses | Identify potential anomalies |
| **5. Visualise Results** | Inspect distributions and signal comparisons | Verify and interpret findings |

---

## Intuitive Analogy

Think of the autoencoder as a **security guard trained to recognise familiar faces** (normal RF patterns):

- When it sees someone familiar, it easily recalls and reconstructs the face (low loss).  
- When it encounters a stranger, it struggles to recall the details (high loss) — that’s your anomaly.

---

## In Plain English

The model “learns” the rhythm of your usual radio activity.  
When something unexpected happens — such as interference, a fault, or a new transmitter — the model notices immediately because it doesn’t fit the patterns it has seen before.

---

## Example Outputs

- **Histogram:** Highlights normal versus anomalous samples based on reconstruction loss distribution.  
- **Scatter Plot:** Visualises which samples exceed the anomaly threshold.  
- **Comparison Plots:** Show how reconstruction quality differs between normal and anomalous signals.

---

## Key Takeaways

- The system requires **no labelled data** — it learns normal behaviour automatically.  
- **Reconstruction loss** provides a reliable measure of “normality.”  
- Visualisations offer clear insight into what’s normal and what’s not.  
- This approach adapts to various RF environments and can detect subtle or emerging anomalies over time.



