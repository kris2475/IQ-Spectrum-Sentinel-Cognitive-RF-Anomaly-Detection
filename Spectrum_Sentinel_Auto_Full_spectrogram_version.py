"""
Spectrum Sentinel Auto Capture Script
------------------------------------

Description:
This script is optimized for training an anomaly detection autoencoder using 
spectrogram data.

- **Spectrogram (Sxx_dB) data is saved unconditionally for ALL sweeps** as a NumPy array 
  (using float32 for size reduction).
- Raw I/Q data is **discarded** to prioritize spectrograms and save disk space.
- Spectrogram images are saved **conditionally** when a strong signal is detected 
  (for visual inspection).

Prerequisites:
- Python 3.8+ installed
- NESDR Smart or compatible SDR connected
- Virtual environment activated with all dependencies installed
    (numpy, scipy, matplotlib, tensorflow, pyrtlsdr, scikit-learn)

Notes:
- **Raw I/Q data is NOT saved in this version.**
- **Spectrogram data (Sxx_dB) is saved as numpy.float32 for ALL sweeps.**
- Adjust SIGNAL_THRESHOLD for desired image/log recording balance.

"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy.signal import spectrogram
from datetime import datetime
import csv
from tqdm import tqdm # Recommended for long sweeps

# Configuration
SWEEP_COUNT = 726            # Number of sweeps across SDR band
SAMPLES_PER_CAPTURE = 262144 # I/Q samples per capture
SAMPLE_RATE = 2.4e6          # 2.4 MHz sample rate
SIGNAL_THRESHOLD = -40.0     # dB threshold for saving spectrogram images
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
SPECTROGRAM_DIR = os.path.join(DATA_DIR, "spectrograms") # New directory for NumPy spectro data
LOG_FILE = os.path.join(DATA_DIR, "sdr_log.csv")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True) # Ensure spectrogram directory exists

# Open CSV log
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Center_Freq_MHz", "Max_dB", "Spectrogram_File", "Image_File"])

# Initialize SDR
sdr = RtlSdr()
# SDR frequency range for Nooelec NESDR Smart
freq_min = 24e6    # 24 MHz
freq_max = 1766e6  # 1766 MHz
freq_step = SAMPLE_RATE # sweep by sample rate

sdr.sample_rate = SAMPLE_RATE
sdr.gain = 'auto'

print(f"[INFO] SDR range: {freq_min/1e6:.1f} MHz to {freq_max/1e6:.1f} MHz")
print(f"[INFO] Samples per capture: {SAMPLES_PER_CAPTURE}")
print(f"[INFO] Sample rate: {SAMPLE_RATE/1e6:.2f} MHz")
print(f"[INFO] Raw I/Q data is **DISCARDED**.")
print(f"[INFO] **Spectrogram data** is saved for ALL sweeps as float32 NumPy arrays.")
print(f"[INFO] Images are saved ONLY if max power > {SIGNAL_THRESHOLD} dB.")

# Sweep loop
sweep_number = 0
for center_freq in tqdm(np.linspace(freq_min, freq_max, SWEEP_COUNT), desc="RF Sweep Progress"):
    sweep_number += 1
    sdr.center_freq = center_freq
    
    # Capture samples (returns complex floating point numbers)
    samples = sdr.read_samples(SAMPLES_PER_CAPTURE)
    
    # Generate spectrogram
    f, t, Sxx = spectrogram(samples, fs=SAMPLE_RATE, nperseg=1024)
    # Convert to dB. Add 1e-12 to avoid log(0)
    Sxx_dB = 10 * np.log10(np.abs(Sxx) + 1e-12) 
    max_power = np.max(Sxx_dB)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_file = ""
    spectrogram_file_saved = ""

    # --- UNCONDITIONAL SPECTROGRAM DATA SAVING (REQUIRED FOR TRAINING) ---
    
    # Convert the spectro data to float32 to save space (typical for DL input)
    Sxx_dB_float32 = Sxx_dB.astype(np.float32)
    
    spectrogram_file = os.path.join(SPECTROGRAM_DIR, f"spec_{int(center_freq/1e6)}MHz_{timestamp}.npy")
    np.save(spectrogram_file, Sxx_dB_float32)
    spectrogram_file_saved = spectrogram_file
    
    # --- CONDITIONAL IMAGE SAVING (for visualization) ---
    is_strong_signal = max_power > SIGNAL_THRESHOLD
    
    if is_strong_signal:
        image_file = os.path.join(IMAGE_DIR, f"spec_{int(center_freq/1e6)}MHz_{timestamp}.png")
        
        # Plotting code remains the same
        plt.figure(figsize=(8,4))
        plt.pcolormesh(t, f/1e6, Sxx_dB, shading='gouraud')
        plt.ylabel('Frequency (MHz)')
        plt.xlabel('Time (s)')
        plt.title(f'Sweep {sweep_number} - Centre {center_freq/1e6:.2f} MHz (Max: {max_power:.2f} dB)')
        plt.colorbar(label='Power (dB)')
        plt.tight_layout()
        plt.savefig(image_file)
        plt.close()
        print(f"[INFO] Strong signal detected ({max_power:.2f} dB). Spectrogram image saved.")
            
    # Append log
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, center_freq/1e6, max_power, spectrogram_file_saved, image_file])

print("\n[INFO] Capture complete.")
sdr.close()













