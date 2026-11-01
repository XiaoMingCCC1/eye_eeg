#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Power Spectrum Analysis
Python implementation based on EEGLAB spectopo.m

Reference: EEGLAB spectopo.m and spectcomp functions for power spectrum calculation
Main functions:
1. Load EEG CSV data
2. Select single channel
3. Calculate power spectral density using Welch method
4. Plot power spectrum (in dB units)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os

def load_eeg_data(csv_file):
    """
    Load EEG CSV data
    
    Parameters:
    csv_file: str, path to CSV file
    
    Returns:
    data: numpy array, shape (n_channels, n_timepoints)
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")
    
    # Read CSV file, assuming no header
    # Each row is one channel, each column is one time point
    data = pd.read_csv(csv_file, header=None).values
    print(f"Data shape: {data.shape}")
    print(f"Number of channels: {data.shape[0]}, Number of time points: {data.shape[1]}")
    print(f"Data range: {data.min():.3f} to {data.max():.3f}")
    
    return data

def compute_power_spectrum(channel_data, srate=1000, nperseg=None, noverlap=None):
    """
    Calculate power spectral density for single channel
    Based on EEGLAB spectcomp function implementation
    
    Parameters:
    channel_data: array, single channel time series data
    srate: float, sampling rate (Hz)
    nperseg: int, length of each segment, default is 1/8 of data length
    noverlap: int, overlap length, default is 50% of nperseg
    
    Returns:
    freqs: array, frequency array
    psd_db: array, power spectral density (dB)
    """
    
    # Remove DC component (similar to EEGLAB processing)
    channel_data = channel_data - np.mean(channel_data)
    
    print(f"Channel data length: {len(channel_data)}")
    print(f"Data range after DC removal: {channel_data.min():.3f} to {channel_data.max():.3f}")
    
    # Set default parameters (reference EEGLAB spectcomp window settings)
    if nperseg is None:
        nperseg = len(channel_data) // 8  # Similar to EEGLAB default window length
        nperseg = max(nperseg, 64)  # Ensure minimum window length
        nperseg = min(nperseg, 512)  # Limit maximum window length
    
    if noverlap is None:
        noverlap = nperseg // 2  # 50% overlap
    
    print(f"Window length: {nperseg}, Overlap: {noverlap}")
    
    # Calculate power spectral density using Welch method (similar to EEGLAB using pwelch)
    freqs, psd = signal.welch(
        channel_data, 
        fs=srate,
        window='hann',  # Hanning window, similar to EEGLAB default
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'  # Power spectral density
    )
    
    # Convert to dB (similar to EEGLAB: 10*log10(psd))
    # Avoid log(0) problem
    psd_db = 10 * np.log10(psd + np.finfo(float).eps)
    
    print(f"Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f"Power spectrum range: {psd_db.min():.2f} - {psd_db.max():.2f} dB")
    
    return freqs, psd_db

def plot_power_spectrum(freqs, psd_db, channel_idx, title_suffix="", freq_limit=50, save_path=None):
    freq_mask = freqs <= freq_limit
    freqs_plot = freqs[freq_mask]
    psd_db_plot = psd_db[freq_mask]
    psd_lin = np.power(10.0, psd_db_plot / 10.0)
    psd_log = np.log10(psd_lin + np.finfo(float).eps)
    psd_log_plot = psd_log
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(freqs_plot, psd_log_plot, color='k', linewidth=2.8)
    fit_mask = freqs_plot >= 1.0
    if np.sum(fit_mask) >= 2:
        x = np.log10(freqs_plot[fit_mask])
        y = psd_log_plot[fit_mask]
        A = np.vstack([np.ones_like(x), x]).T
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        y_fit = coef[0] + coef[1] * np.log10(freqs_plot[fit_mask])
        ax.plot(freqs_plot[fit_mask], y_fit, color='royalblue', linestyle='--', linewidth=2.2)
    ax.set_xlabel('Frequency', fontsize=18)
    ax.set_ylabel('log(Power)', fontsize=18)
    ax.set_xlim([0, freq_limit])
    ax.minorticks_on()
    ax.grid(True, which='major', color='#d0d0d0', linestyle='-', alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=16)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n=== Channel {channel_idx+1} Power Spectrum Statistics ===")
    print(f"Mean log-power: {np.mean(psd_log):.2f}")
    print(f"Peak log-power: {np.max(psd_log):.2f} (@ {freqs_plot[np.argmax(psd_log)]:.2f} Hz)")
    print(f"Minimum log-power: {np.min(psd_log):.2f}")

def main():
    """
    Main function: demonstrate power spectrum analysis
    """
    
    # File path
    csv_file = "f:/eye_eeg/A00053375_Video1_EEG.csv"
    
    print("=== EEG Power Spectrum Analysis ===")
    print("Based on EEGLAB spectopo.m implementation\n")
    
    try:
        # 1. Load data
        print("1. Loading EEG data...")
        data = load_eeg_data(csv_file)
        
        # 2. Select channel (default: first channel)
        channel_idx = 93  # Can be modified to other channels
        # IMPORTANT: Each row is one channel, each column is one time point
        channel_data = data[channel_idx, :]  # Select row (channel)
        
        print(f"\n2. Selected channel {channel_idx+1}")
        print(f"Channel data length: {len(channel_data)}")
        print(f"Data range: {channel_data.min():.3f} - {channel_data.max():.3f}")
        
        # 3. Calculate power spectrum
        print(f"\n3. Computing power spectrum...")
        srate = 1000  # Assume sampling rate is 1000Hz, adjust according to actual situation
        freqs, psd_db = compute_power_spectrum(channel_data, srate=srate)
        
        # 4. Plot power spectrum
        print(f"\n4. Plotting power spectrum...")
        out_path = os.path.join(os.path.dirname(csv_file), f"{os.path.splitext(os.path.basename(csv_file))[0]}_power.png")
        plot_power_spectrum(freqs, psd_db, channel_idx, title_suffix=f" (Sampling Rate: {srate}Hz)", save_path=out_path)
        print(f"Saved power spectrum: {out_path}")
        print(f"\n=== Analysis Complete ===")
        print(f"Reference EEGLAB spectopo.m implementation:")
        print(f"- Use Welch method to calculate power spectral density")
        print(f"- Apply Hanning window for data segmentation")
        print(f"- Convert to dB units: 10*log10(PSD)")
        print(f"- Display common EEG frequency band markers")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
