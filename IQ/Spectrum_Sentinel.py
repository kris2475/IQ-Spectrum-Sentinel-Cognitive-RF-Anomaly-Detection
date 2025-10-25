#!/usr/bin/env python3
"""
Spectrum Sentinel - SDR Capture and Spectrogram Generator
A self-contained, user-friendly script for NESDR Smart users.
"""

import sys
import subprocess
import os
import numpy as np
from pathlib import Path

# ------------------------
# 1. Environment check
# ------------------------
MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required.")
    sys.exit(1)

print(f"Python version {sys.version_info.major}.{sys.version_info.minor} OK.")

# ------------------------
# 2. Ensure dependencies
# ------------------------
required_packages = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "scikit-learn",
    "pyrtlsdr",
    "tensorflow"
]

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Package '{pkg}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ------------------------
# 3. Import SDR packages after installing
# ------------------------
from rtlsdr import RtlSdr
from scipy.signal import stft
import matplotlib.pyplot as plt

# ------------------------
# 4. User prompts
# ------------------------
print("\nWelcome to Spectrum Sentinel - SDR Capture Utility!\n")
print("Please ensure your NESDR Smart is connected and antenna attached.\n")

bands = []
while True:
    try:
        freq = float(input("Enter a centre frequency in MHz (or '0' to finish): "))
        if freq == 0:
            break
        duration = float(input(f"Enter capture duration for {freq} MHz in seconds: "))
        bands.append({'freq': freq*1e6, 'duration': duration})
    except ValueError:
        print("Invalid input, please enter numbers only.")

if not bands:
    print("No bands selected. Exiting.")
    sys.exit(0)

sample_rate = float(input("Enter sample rate in Msps (e.g., 2.048): ")) * 1e6
gain = float(input("Enter gain (0–50 recommended, e.g., 40): "))
slice_seconds = float(input("Enter spectrogram slice length in seconds (e.g., 2): "))

out_dir = Path("data/spectrograms")
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------
# 5. SDR capture
# ------------------------
try:
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.gain = gain
except Exception as e:
    print(f"Error initialising SDR: {e}")
    sys.exit(1)

for band in bands:
    print(f"\nCapturing {band['freq']/1e6:.2f} MHz for {band['duration']} seconds...")
    sdr.center_freq = band['freq']
    num_samples = int(band['duration'] * sample_rate)
    iq_samples = sdr.read_samples(num_samples)
    iq_samples = iq_samples.astype(np.complex64)

    band_dir = out_dir / f"{int(band['freq']/1e6)}MHz"
    band_dir.mkdir(exist_ok=True)

    n_slice = int(slice_seconds * sample_rate)
    num_slices = len(iq_samples) // n_slice

    print(f"Total slices to generate: {num_slices}")
    for i in range(num_slices):
        slice_iq = iq_samples[i*n_slice:(i+1)*n_slice]
        f, t, Zxx = stft(slice_iq, fs=sample_rate, nperseg=1024, noverlap=1024//2)
        Sxx = np.abs(Zxx)
        np.save(band_dir / f"spec_{i:04d}.npy", Sxx)

        if i % 10 == 0:
            print(f"Saved slice {i}/{num_slices}")

print("\nCapture complete! Spectrogram slices saved in 'data/spectrograms/'.")

sdr.close()

# ------------------------
# 6. Optional preview
# ------------------------
preview = input("Do you want to preview the first spectrogram slice? (y/n): ")
if preview.lower() == 'y':
    first_band = bands[0]['freq']/1e6
    slice_file = out_dir / f"{int(first_band)}MHz" / "spec_0000.npy"
    if slice_file.exists():
        Sxx = np.load(slice_file)
        plt.imshow(20*np.log10(Sxx + 1e-12), aspect='auto', origin='lower')
        plt.title(f"Spectrogram Preview: {int(first_band)} MHz")
        plt.xlabel("Time bins")
        plt.ylabel("Frequency bins")
        plt.colorbar(label='Amplitude (dB)')
        plt.show()
    else:
        print("Slice not found.")

print("All done. You can now use these slices for autoencoder training.")
