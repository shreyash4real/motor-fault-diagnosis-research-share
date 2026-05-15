"""
STEP 4 — PRECOMPUTE STFT FEATURES (Robust for nobpfo100)
========================================================
Reads the nobpfo100 splits.csv and produces three .pt files:
  features/stft_robust/train.pt
  features/stft_robust/val.pt
  features/stft_robust/test.pt

CHANGES FOR ROBUSTNESS:
  - FREQ_CUTOFF_HZ lowered from 3000 to 1000. 
  - BPFO modulations are typically low frequency. This crops out 
    the 1000-3000 Hz band to prevent the model from wasting capacity
    on high-frequency noise.
  - New output shape will be roughly (3, 52, 149).
"""

from __future__ import annotations

import os
import re
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import stft

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
RAW_ROOT       = r"C:\Project Work\Dataset\Electric\Motor-2"
DENOISED_ROOT  = r"C:\Project Work\Denoised\Motor-2"
SPLITS_CSV     = r"C:\Project Work\Outputs_nobpfo100\splits.csv"
FEATURES_DIR   = r"C:\Project Work\Outputs_nobpfo100\features\stft_robust"
SAVE_PNGS      = False
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "healthy 1",
    "stator short 1",
    "bearing bpfo 3",
    "broken rotor bar",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Locked STFT parameters
FS              = 20_000
WINDOW_SAMPLES  = 20_000
NPERSEG         = 1024
NOVERLAP        = 896

# CHANGED: Lower frequency cutoff to focus on BPFO modulations
FREQ_CUTOFF_HZ  = 1000

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def rewrite_path(raw_path: str) -> str:
    """Map raw-dataset path → denoised-dataset path."""
    return raw_path.replace(RAW_ROOT, DENOISED_ROOT)

def load_full_three_phases(ch1_path: str, col_idx: int) -> np.ndarray:
    """Load one full 15-second column across all 3 phases."""
    phases = []
    for ch in ("ch1", "ch2", "ch3"):
        path = ch1_path.replace("-ch1.csv", f"-{ch}.csv")
        df = pd.read_csv(path)
        phases.append(df.iloc[:, col_idx + 1].values.astype(np.float32))
    return np.stack(phases)

def compute_spectrogram_db(sig_3phase: np.ndarray) -> np.ndarray:
    """
    (3, 20000) signal → float32 log-magnitude dB spectrogram.
    """
    spec_phases = []
    for ph in range(3):
        f, _, Zxx = stft(sig_3phase[ph], fs=FS, window="hann",
                         nperseg=NPERSEG, noverlap=NOVERLAP,
                         boundary=None, padded=False)
        mag = np.abs(Zxx).astype(np.float32)
        log_mag = 20.0 * np.log10(mag + 1e-10)
        log_mag = log_mag[f <= FREQ_CUTOFF_HZ]
        spec_phases.append(log_mag)
    return np.stack(spec_phases).astype(np.float32)

def zscore_normalize(spec_db: np.ndarray) -> np.ndarray:
    """Per-image z-score across all 3 channels together."""
    mean = spec_db.mean()
    std  = spec_db.std() + 1e-8
    return (spec_db - mean) / std

def fmt_time(seconds: float) -> str:
    if seconds < 60:   return f"{seconds:.0f}s"
    if seconds < 3600: return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"

def fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024 ** 2: return f"{n_bytes / 1024:.0f} KB"
    if n_bytes < 1024 ** 3: return f"{n_bytes / (1024 ** 2):.0f} MB"
    return f"{n_bytes / (1024 ** 3):.2f} GB"

# ──────────────────────────────────────────────────────────────────────────────
#  Per-split processing
# ──────────────────────────────────────────────────────────────────────────────

def process_split(splits_df: pd.DataFrame, split: str,
                  features_dir: Path, spec_shape: tuple | None) -> tuple:
    print(f"\nProcessing: {split.upper()}")
    df = splits_df[
        (splits_df["split"] == split) &
        (splits_df["full_path"].str.contains("-ch1.csv"))
    ].copy()

    groups = defaultdict(list)
    for _, row in df.iterrows():
        ch1_path = rewrite_path(row["full_path"])
        col_idx  = int(row["col_index"])
        n_segs   = int(row["n_segments"])
        for seg_idx in range(n_segs):
            groups[(ch1_path, col_idx)].append({
                "seg_idx":     seg_idx,
                "class_label": row["class_label"],
                "label_idx":   LABEL_TO_IDX[row["class_label"]],
                "n_segments":  n_segs,
                "speed_pct":   int(row["speed_pct"]),
            })

    n_columns = len(groups)
    n_samples = sum(len(v) for v in groups.values())

    if spec_shape is None:
        first_key = next(iter(groups))
        full = load_full_three_phases(*first_key)
        spec_shape = compute_spectrogram_db(full[:, :WINDOW_SAMPLES]).shape
        print(f"Per-sample shape   : {spec_shape}")
    
    print(f"Samples to process : {n_samples:,}")

    features_buf = np.zeros((n_samples, *spec_shape), dtype=np.float32)
    labels_buf   = np.zeros((n_samples,), dtype=np.int64)
    metadata     = [None] * n_samples

    sample_i = 0
    t_start  = time.time()

    for col_i, ((ch1_path, col_idx), entries) in enumerate(groups.items(), 1):
        try:
            full_3phase = load_full_three_phases(ch1_path, col_idx)
        except Exception as e:
            print(f"  ERROR loading col {col_idx} of {os.path.basename(ch1_path)}: {e}")
            sample_i += len(entries)
            continue

        n_segs = entries[0]["n_segments"]
        stride = (300_000 - WINDOW_SAMPLES) // (n_segs - 1) if n_segs > 1 else 0

        for entry in entries:
            start = entry["seg_idx"] * stride
            sig   = full_3phase[:, start:start + WINDOW_SAMPLES]

            try:
                spec_db = compute_spectrogram_db(sig)
            except Exception as e:
                print(f"  ERROR STFT col {col_idx} seg {entry['seg_idx']}: {e}")
                sample_i += 1
                continue

            features_buf[sample_i] = zscore_normalize(spec_db)
            labels_buf[sample_i]   = entry["label_idx"]
            metadata[sample_i] = {
                "sample_idx":  sample_i,
                "ch1_path":    ch1_path,
                "col_index":   col_idx,
                "seg_idx":     entry["seg_idx"],
                "class_label": entry["class_label"],
                "speed_pct":   entry["speed_pct"],
            }
            sample_i += 1

        if col_i % max(1, n_columns // 10) == 0 or col_i == n_columns:
            elapsed = time.time() - t_start
            rate    = sample_i / elapsed if elapsed > 0 else 0
            print(f"  col {col_i:>3}/{n_columns}  {sample_i:>5}/{n_samples} ({100*sample_i/n_samples:5.1f}%)  {rate:5.1f} smp/s")

    elapsed = time.time() - t_start
    print(f"Done in {fmt_time(elapsed)}")

    save_path = features_dir / f"{split}.pt"
    t0 = time.time()
    torch.save({
        "features":    torch.from_numpy(features_buf),
        "labels":      torch.from_numpy(labels_buf),
        "class_names": CLASS_NAMES,
        "metadata":    metadata,
        "stft_params": {
            "fs":              FS,
            "nperseg":         NPERSEG,
            "noverlap":        NOVERLAP,
            "window":          "hann",
            "boundary":        None,
            "padded":          False,
            "freq_cutoff_hz":  FREQ_CUTOFF_HZ,
            "window_samples":  WINDOW_SAMPLES,
            "normalization":   "per_image_zscore",
        },
    }, save_path)
    save_elapsed = time.time() - t0
    file_size = save_path.stat().st_size
    print(f"Saved {split}.pt ({fmt_size(file_size)})")

    return n_samples, file_size, elapsed + save_elapsed, spec_shape


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PRECOMPUTE ROBUST STFT FEATURES — nobpfo100")
    print("=" * 70)
    print(f"splits.csv:     {SPLITS_CSV}")
    print(f"features dir:   {FEATURES_DIR}")
    print(f"freq_cutoff:    {FREQ_CUTOFF_HZ} Hz (ROBUST)")
    print()

    if not os.path.exists(SPLITS_CSV):
        print(f"\nERROR: splits.csv not found at {SPLITS_CSV}")
        sys.exit(1)

    splits_df = pd.read_csv(SPLITS_CSV)
    features_dir = Path(FEATURES_DIR)
    features_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()
    summary = []
    spec_shape = None

    for split in ("train", "val", "test"):
        n, sz, dt, spec_shape = process_split(
            splits_df, split, features_dir, spec_shape)
        summary.append((f"{split}.pt", n, sz, dt))

    total_elapsed = time.time() - t_total_start

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    total_bytes = total_samples = 0
    for name, n, sz, dt in summary:
        print(f"  {name:<18} {n:>6} samples  {fmt_size(sz):>9}  ({fmt_time(dt)})")
        total_bytes   += sz
        total_samples += n
    print(f"  {'TOTAL':<18} {total_samples:>6} samples  {fmt_size(total_bytes):>9}")

if __name__ == "__main__":
    main()
