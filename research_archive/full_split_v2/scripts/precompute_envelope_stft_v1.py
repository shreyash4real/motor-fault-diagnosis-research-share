"""
STEP 4 — PRECOMPUTE ENVELOPE STFT FEATURES + PER-CHANNEL PNG TREE (4-class)
===========================================================================
Reads splits.csv and produces two parallel artifacts for the envelope STFT:

  1. THREE .pt files for training:
        envelope_stft/features/train.pt
        envelope_stft/features/val.pt
        envelope_stft/features/test.pt

  2. PER-CHANNEL PNG TREE for inspection:
        envelope_stft/images/<split>/<class>/<speed>/colXX_segYYY_chN.png

SIGNAL PROCESSING
-----------------
  1. Segment is detrended.
  2. Hilbert analytic signal is computed and magnitude taken to get the envelope.
  3. Envelope is detrended (removes DC carrier).
  4. scipy.signal.stft is applied.
  
LOCKED STFT PARAMETERS
----------------------
  fs             = 20_000
  window_samples = 20_000
  nperseg        = 8192      (~2.44 Hz resolution for crisp narrowband peaks)
  noverlap       = 8000      (dense sliding to preserve image width)
  freq_cutoff    = 500 Hz    (narrowband envelope region)
  
PNG VISUALIZATION
-----------------
  Raw dB values, jet colormap, no axes/labels.
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

from pipeline_paths import legacy_path

import numpy as np
import pandas as pd
import torch
from scipy.signal import stft, hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
RAW_ROOT       = legacy_path(r"C:\Project Work\Dataset\Electric\Motor-2")
DENOISED_ROOT  = legacy_path(r"C:\Project Work\Denoised\Motor-2")
SPLITS_CSV     = legacy_path(r"C:\Project Work\outputs\4class\v2_speed_strat\splits.csv")
FEATURES_DIR   = legacy_path(r"C:\Project Work\outputs\4class\v2_speed_strat\envelope_stft\features")
IMAGES_DIR     = legacy_path(r"C:\Project Work\outputs\4class\v2_speed_strat\envelope_stft\images")
SAVE_PNGS      = True
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "healthy 1",
    "stator short 1",
    "bearing bpfo 3",
    "broken rotor bar",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# STFT and Signal Parameters
FS              = 20_000
WINDOW_SAMPLES  = 20_000
NPERSEG         = 8192
NOVERLAP        = 8000
FREQ_CUTOFF_HZ  = 500

# PNG visualization params (match main STFT aesthetic)
PNG_VMIN_DB, PNG_VMAX_DB = -50.0, 10.0
PNG_CMAP = "jet"


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip())

def rewrite_path(raw_path: str) -> str:
    return raw_path.replace(RAW_ROOT, DENOISED_ROOT)

def load_full_three_phases(ch1_path: str, col_idx: int) -> np.ndarray:
    phases = []
    for ch in ("ch1", "ch2", "ch3"):
        path = ch1_path.replace("-ch1.csv", f"-{ch}.csv")
        df = pd.read_csv(path)
        phases.append(df.iloc[:, col_idx + 1].values.astype(np.float32))
    return np.stack(phases)

def compute_envelope_stft_db(sig_3phase: np.ndarray) -> np.ndarray:
    """
    1. Detrend signal
    2. Hilbert -> Magnitude (Envelope)
    3. Detrend Envelope
    4. STFT -> Magnitude -> dB -> crop <= 500 Hz
    """
    spec_phases = []
    for ph in range(3):
        x = sig_3phase[ph].astype(np.float32)
        # 1. Detrend
        x = x - x.mean()
        # 2. Envelope
        env = np.abs(hilbert(x)).astype(np.float32)
        # 3. Detrend Envelope
        env = env - env.mean()
        
        # 4. STFT
        f, t_ax, Zxx = stft(env, fs=FS, window="hann",
                         nperseg=NPERSEG, noverlap=NOVERLAP,
                         boundary=None, padded=False)
        mag = np.abs(Zxx).astype(np.float32)
        log_mag = 20.0 * np.log10(mag + 1e-10)
        
        # Crop to cutoff
        log_mag = log_mag[f <= FREQ_CUTOFF_HZ]
        spec_phases.append(log_mag)
        
    return np.stack(spec_phases).astype(np.float32)

def zscore_normalize(spec_db: np.ndarray) -> np.ndarray:
    mean = spec_db.mean()
    std  = spec_db.std() + 1e-8
    return (spec_db - mean) / std

def save_channel_pngs(spec_db: np.ndarray, out_dir: Path,
                      col_idx: int, seg_idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch_i in range(3):
        img = spec_db[ch_i]
        fname = f"col{col_idx:02d}_seg{seg_idx:03d}_ch{ch_i + 1}.png"
        plt.imsave(out_dir / fname, img,
                   cmap=PNG_CMAP,
                   vmin=PNG_VMIN_DB, vmax=PNG_VMAX_DB,
                   origin="lower")

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
                  features_dir: Path, images_dir: Path,
                  save_png_flag: bool,
                  spec_shape: tuple | None) -> tuple:
    print()
    print("─" * 70)
    print(f"Processing: {split.upper()}")
    print("─" * 70)

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

    print(f"Expected shape     : {spec_shape}")
    print(f"Samples to process : {n_samples:,}")
    print(f"Unique columns     : {n_columns}")

    features_buf = None
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
                spec_db = compute_envelope_stft_db(sig)
            except Exception as e:
                print(f"  ERROR STFT col {col_idx} seg {entry['seg_idx']}: {e}")
                sample_i += 1
                continue

            if features_buf is None:
                spec_shape = spec_db.shape
                print(f"\nDiscovered spec shape: {spec_shape}")
                features_buf = np.zeros((n_samples, *spec_shape), dtype=np.float32)
                print(f"Pre-allocated      : {fmt_size(features_buf.nbytes)}\n")

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

            if save_png_flag:
                cls_folder   = sanitize(entry["class_label"])
                speed_folder = f"speed_{entry['speed_pct']}"
                out_dir = images_dir / split / cls_folder / speed_folder
                save_channel_pngs(spec_db, out_dir, col_idx, entry["seg_idx"])

            sample_i += 1

        elapsed = time.time() - t_start
        rate    = sample_i / elapsed if elapsed > 0 else 0
        eta     = (n_samples - sample_i) / rate if rate > 0 else 0
        short   = os.path.basename(ch1_path)[:42]
        print(f"  col {col_i:>3}/{n_columns}  {short:<44}  "
              f"{sample_i:>5}/{n_samples} ({100*sample_i/n_samples:5.1f}%)  "
              f"elapsed={fmt_time(elapsed):>5}  "
              f"ETA={fmt_time(eta):>5}  {rate:5.1f} smp/s")

    elapsed = time.time() - t_start
    print(f"\nComputation done in {fmt_time(elapsed)}")

    save_path = features_dir / f"{split}.pt"
    print(f"Saving {split}.pt ...", end=" ", flush=True)
    t0 = time.time()
    torch.save({
        "features":    torch.from_numpy(features_buf),
        "labels":      torch.from_numpy(labels_buf),
        "class_names": CLASS_NAMES,
        "metadata":    metadata,
        "stft_params": {
            "fs":             FS,
            "window_samples": WINDOW_SAMPLES,
            "nperseg":        NPERSEG,
            "noverlap":       NOVERLAP,
            "freq_cutoff":    FREQ_CUTOFF_HZ,
            "normalization":  "per_image_zscore",
        },
    }, save_path)
    save_elapsed = time.time() - t0
    file_size = save_path.stat().st_size
    print(f"done in {fmt_time(save_elapsed)}  ({fmt_size(file_size)})")

    counts = np.bincount(labels_buf, minlength=len(CLASS_NAMES))
    print("Per-class counts in saved file:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<22} {counts[i]:>5}")

    return n_samples, file_size, elapsed + save_elapsed, spec_shape


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PRECOMPUTE ENVELOPE STFT FEATURES + PER-CHANNEL PANELS - 4 classes")
    print("=" * 70)
    print(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"splits.csv:     {SPLITS_CSV}")
    print(f"denoised root:  {DENOISED_ROOT}")
    print(f"features dir:   {FEATURES_DIR}")
    print(f"images dir:     {IMAGES_DIR}")
    print(f"save PNGs:      {SAVE_PNGS}")
    print()
    print("LOCKED STFT PARAMETERS:")
    print(f"  fs                = {FS}")
    print(f"  window_samples    = {WINDOW_SAMPLES} (1.0 s)")
    print(f"  nperseg           = {NPERSEG} (Freq Res: {FS/NPERSEG:.2f} Hz)")
    print(f"  noverlap          = {NOVERLAP}")
    print(f"  freq cutoff       = {FREQ_CUTOFF_HZ} Hz")
    print()

    if not os.path.exists(SPLITS_CSV):
        print(f"\nERROR: splits.csv not found at {SPLITS_CSV}")
        sys.exit(1)

    splits_df = pd.read_csv(SPLITS_CSV)
    features_dir = Path(FEATURES_DIR)
    images_dir   = Path(IMAGES_DIR)
    features_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PNGS:
        images_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()
    summary = []
    discovered_shape = None

    for split in ("train", "val", "test"):
        n, sz, dt, spec_shape = process_split(
            splits_df, split, features_dir, images_dir,
            save_png_flag=SAVE_PNGS, spec_shape=discovered_shape)
        if discovered_shape is None:
            discovered_shape = spec_shape
        summary.append((f"{split}.pt", n, sz, dt))

    total_elapsed = time.time() - t_total_start

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(f"Total elapsed : {fmt_time(total_elapsed)}")
    print(f"Final Tensor Shape: {discovered_shape}")
    print()
    print("Files written:")
    total_bytes = total_samples = 0
    for name, n, sz, dt in summary:
        print(f"  {name:<18} {n:>6} samples  {fmt_size(sz):>9}  ({fmt_time(dt)})")
        total_bytes   += sz
        total_samples += n

if __name__ == "__main__":
    main()
