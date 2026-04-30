"""
diagnose_envelope_time_domain.py
Standalone diagnostic script to visualize the time-domain extraction of the Hilbert envelope
for a healthy 100% speed signal.

Usage:
    python diagnose_envelope_time_domain.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
FILE_PATH = r"C:\Project Work\Denoised\Motor-2\100\healthy 1\Electric_Motor-2_100_time-healthy 1-ch1.csv"
OUTPUT_PATH = r"C:\Project Work\Outputs\4class\diagnostics\healthy_1_100_col1_envelope_time_domain.png"
FS = 20_000
# ───────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    # Col index 1 in the 4-class project generally refers to the first signal column
    # after the 'time' column (which is index 0). Hence, df.iloc[:, 2].
    # Wait, the prompt says "column 1", we will use col_index=1, which is index 2.
    signal = df.iloc[:, 2].values.astype(np.float32)
    
    n_samples = len(signal)
    t = np.arange(n_samples) / FS
    print(f"Loaded {n_samples} samples ({n_samples/FS:.1f} seconds).")

    # Step 1: Detrend the raw signal
    x = signal - signal.mean()

    # Step 2: Compute Analytic Signal and Envelope Magnitude
    print("Computing Hilbert Transform...")
    analytic = hilbert(x)
    env_raw = np.abs(analytic)

    # Step 3: Detrend Envelope (Remove DC Offset/Carrier)
    env_detrended = env_raw - env_raw.mean()

    # ─── VISUALIZATION (Interactive) ─────────────────────────────────────────
    print("Generating plot...")
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Envelope Extraction: Healthy 1 (100% Speed, Col 1)", fontsize=16)

    # Panel 1: Full 15s Raw vs Raw Envelope
    axs[0].plot(t, x, color="lightgray", linewidth=0.5, label="Raw Signal (detrended)")
    axs[0].plot(t, env_raw, color="red", linewidth=1.0, alpha=0.8, label="Raw Envelope (with DC offset)")
    axs[0].set_title("Full 15s: Raw Signal & Raw Envelope")
    axs[0].set_ylabel("Current Amplitude")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    # Panel 2: Zoomed in 0.5s
    zoom_sec = 0.5
    zoom_smp = int(zoom_sec * FS)
    t_zoom = t[:zoom_smp]
    x_zoom = x[:zoom_smp]
    env_raw_zoom = env_raw[:zoom_smp]

    axs[1].plot(t_zoom, x_zoom, color="lightgray", linewidth=1.0, label="Raw Signal (detrended)")
    axs[1].plot(t_zoom, env_raw_zoom, color="red", linewidth=2.0, alpha=0.8, label="Raw Envelope (with DC offset)")
    axs[1].set_title(f"Zoomed {zoom_sec}s: How the Envelope Traces the Carrier Peaks")
    axs[1].set_ylabel("Current Amplitude")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    # Panel 3: Zoomed in 0.5s Detrended Envelope
    env_detrended_zoom = env_detrended[:zoom_smp]
    
    axs[2].plot(t_zoom, env_detrended_zoom, color="blue", linewidth=2.0, label="Detrended Envelope")
    axs[2].axhline(0, color="black", linestyle="--", linewidth=1.0, label="Zero Mean")
    axs[2].set_title(f"Zoomed {zoom_sec}s: Detrended Envelope (Carrier Removed)")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_ylabel("Modulation Amplitude")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved diagnostic PNG to: {OUTPUT_PATH}")

    # Show interactive GUI window
    print("Opening interactive plot window. You can use the magnifying glass tool to zoom in/out.")
    plt.show()

if __name__ == "__main__":
    main()
