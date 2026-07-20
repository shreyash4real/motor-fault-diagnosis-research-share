"""
STEP 4 — PRECOMPUTE ENVELOPE-SPECTRUM FEATURES + PER-SEGMENT PNG (4-class)
==========================================================================
Sibling of precompute_stft_v2.py / precompute_dwt_v1.py / precompute_clarke_v2.py.
Reads the locked v2_speed_strat splits and produces two parallel artefacts:

  1. THREE .pt files for training (no augmentation):
        envelope/features/train.pt
        envelope/features/val.pt
        envelope/features/test.pt

  2. PER-PHASE PNG TREE for inspection:
        envelope/images/<split>/<class>/<speed>/colXX_segYYY.png
        Three compact PNGs per segment, one per phase:
        colXX_segYYY_ch1.png, colXX_segYYY_ch2.png, colXX_segYYY_ch3.png.
        No labels, no axes, no title — matches the minimalist convention
        and channel suffix naming of the STFT PNGs.

REPRESENTATION — WHY ENVELOPE SPECTRUM
--------------------------------------
  Outer-race bearing defects modulate the line current via torque ripple.
  The result is a pair of sidebands around the supply fundamental at
  f_line +/- k*f_BPFO. Those sidebands are small compared to the 50 Hz
  line peak, which is why linear STFT + CNN plateaus for bearing bpfo 3
  (test F1 ~ 0.79). A Hilbert envelope strips the carrier and moves the
  modulation (f_BPFO and its harmonics) directly into baseband, where
  BPFO peaks are easy to see.

  Expected BPFO at 50/75/100% speed (9-ball bearing, BPFO/f_shaft ~ 3.5):
    50% speed: ~ 44 Hz
    75% speed: ~ 66 Hz
    100% speed: ~ 87 Hz
  The 0-500 Hz cap covers up to the 5th BPFO harmonic at 100% speed.

LOCKED PARAMETERS (shared with other precompute scripts)
--------------------------------------------------------
  fs             = 20000
  window_samples = 20000           (matches STFT / DWT / Clarke)
  n_segments     = 57 per column   (via splits.csv)
  input          = denoised (5-5000 Hz Butterworth, filtfilt)

ENVELOPE-SPECIFIC PARAMETERS
----------------------------
  detrend         = subtract mean of raw segment
  analytic        = scipy.signal.hilbert (FFT-based, length 20000)
  envelope        = |analytic|; then subtract mean (remove env DC)
  fft_window      = Hann on envelope before FFT
  fft_length      = 20000          (native; 1 Hz frequency resolution)
  freq_cap_hz     = 500            -> 501 bins (0, 1, 2, ..., 500 Hz)
  magnitude_form  = log1p(|rfft|)  (compresses 4+ decade dynamic range)
  normalization   = per-image z-score (across all 3 phases), same as STFT

  per-sample shape = (3, 501) float32
  per-sample bytes = ~6 KB

Usage
-----
    python precompute_envelope_v1.py

Requires: pandas, numpy, scipy, torch, matplotlib
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
from scipy.signal import hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
RAW_ROOT       = legacy_path(r"C:\Project Work\Dataset\Electric\Motor-2")
DENOISED_ROOT  = legacy_path(r"C:\Project Work\Denoised\Motor-2")
SPLITS_CSV     = legacy_path(r"C:\Project Work\outputs\4class\v2_speed_strat\splits.csv")
FEATURES_DIR   = legacy_path(r"C:\Project Work\outputs\4class\v2_speed_strat\envelope\features")
IMAGES_DIR     = legacy_path(r"C:\Project Work\outputs\4class\v2_speed_strat\envelope\images")
SAVE_PNGS      = True
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "healthy 1",
    "stator short 1",
    "bearing bpfo 3",
    "broken rotor bar",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ─── LOCKED ENVELOPE PARAMETERS ───────────────────────────────────────────────
FS              = 20_000
WINDOW_SAMPLES  = 20_000
FREQ_CAP_HZ     = 500
N_FREQ_BINS     = FREQ_CAP_HZ + 1         # 501 bins at 1 Hz resolution
# ──────────────────────────────────────────────────────────────────────────────

# Pre-computed Hann window for envelope (applied before rfft)
_HANN_ENV = np.hanning(WINDOW_SAMPLES).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """'bearing bpfo 3' -> 'bearing_bpfo_3' (Windows-safe folder name)."""
    return re.sub(r"\s+", "_", name.strip())


def rewrite_path(raw_path: str) -> str:
    """Map raw-dataset path -> denoised-dataset path."""
    return raw_path.replace(RAW_ROOT, DENOISED_ROOT)


def load_full_three_phases(ch1_path: str, col_idx: int) -> np.ndarray:
    """
    Load one full 15-second column across all 3 phases.
    Returns (3, 300_000) float32. Each of ch1/ch2/ch3 is read exactly once.
    """
    phases = []
    for ch in ("ch1", "ch2", "ch3"):
        path = ch1_path.replace("-ch1.csv", f"-{ch}.csv")
        df = pd.read_csv(path)
        phases.append(df.iloc[:, col_idx + 1].values.astype(np.float32))
    return np.stack(phases)


def compute_envelope_spectrum(sig_3phase: np.ndarray) -> np.ndarray:
    """
    (3, 20000) signal -> (3, 501) float32 envelope spectrum (log1p-compressed).
    NOT z-score normalized. Used as the source for both PNG rendering (raw
    log1p) and the z-score step that produces the training tensor.

    Pipeline per phase:
      detrend -> Hilbert -> |analytic| -> detrend envelope ->
      Hann window -> rfft -> magnitude -> crop 0..500 Hz -> log1p
    """
    spec_phases = []
    for ph in range(3):
        x = sig_3phase[ph].astype(np.float32)
        x = x - x.mean()
        env = np.abs(hilbert(x)).astype(np.float32)
        env = env - env.mean()
        env = env * _HANN_ENV
        spec = np.abs(np.fft.rfft(env)).astype(np.float32)
        spec = spec[:N_FREQ_BINS]
        spec_phases.append(np.log1p(spec))
    return np.stack(spec_phases).astype(np.float32)


def zscore_normalize(spec_log: np.ndarray) -> np.ndarray:
    """
    Per-image z-score across all 3 channels together. Preserves relative
    magnitudes between phases. Produces the training tensor.
    """
    mean = spec_log.mean()
    std  = spec_log.std() + 1e-8
    return (spec_log - mean) / std


def save_envelope_phase_pngs(spec_log: np.ndarray,
                             out_dir: Path,
                             col_idx: int,
                             seg_idx: int) -> None:
    """
    Save 3 PNGs per segment, one per phase channel, using the same
    colXX_segYYY_chN.png naming convention as the STFT branch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.arange(N_FREQ_BINS)
    phase_colors = ("#c0392b", "#27ae60", "#2980b9")   # ch1 / ch2 / ch3
    for ch_i in range(3):
        fig, ax = plt.subplots(figsize=(4, 1.2))
        ax.plot(freqs, spec_log[ch_i],
                linewidth=0.55, color=phase_colors[ch_i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=0, y=0.05)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        fname = f"col{col_idx:02d}_seg{seg_idx:03d}_ch{ch_i + 1}.png"
        plt.savefig(out_dir / fname, dpi=100, facecolor="white")
        plt.close(fig)


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
                  save_png_flag: bool) -> tuple:
    """Process one split; write .pt + PNGs. No augmentation."""
    print()
    print("-" * 70)
    print(f"Processing: {split.upper()}")
    print("-" * 70)

    df = splits_df[
        (splits_df["split"] == split) &
        (splits_df["full_path"].str.contains("-ch1.csv"))
    ].copy()

    # Group by (ch1_path, col_idx) so each column is loaded once
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

    spec_shape = (3, N_FREQ_BINS)
    print(f"Per-sample shape   : {spec_shape}")
    print(f"Samples to process : {n_samples:,}")
    print(f"Unique columns     : {n_columns}")

    features_buf = np.zeros((n_samples, *spec_shape), dtype=np.float32)
    labels_buf   = np.zeros((n_samples,), dtype=np.int64)
    metadata     = [None] * n_samples
    print(f"Pre-allocated      : {fmt_size(features_buf.nbytes)}")
    if save_png_flag:
        print(f"PNGs will be written under: {images_dir / split}")
        print(f"  3 phase PNGs per segment -> {n_samples * 3:,} files total")
    print()

    sample_i = 0
    t_start  = time.time()

    for col_i, ((ch1_path, col_idx), entries) in enumerate(groups.items(), 1):
        try:
            full_3phase = load_full_three_phases(ch1_path, col_idx)
        except Exception as e:
            print(f"  ERROR loading col {col_idx} of "
                  f"{os.path.basename(ch1_path)}: {e}")
            sample_i += len(entries)
            continue

        n_segs = entries[0]["n_segments"]
        stride = (300_000 - WINDOW_SAMPLES) // (n_segs - 1) if n_segs > 1 else 0

        for entry in entries:
            start = entry["seg_idx"] * stride
            sig   = full_3phase[:, start:start + WINDOW_SAMPLES]

            try:
                spec_log = compute_envelope_spectrum(sig)
            except Exception as e:
                print(f"  ERROR envelope col {col_idx} seg {entry['seg_idx']}: {e}")
                sample_i += 1
                continue

            # Training tensor uses z-scored version
            features_buf[sample_i] = zscore_normalize(spec_log)
            labels_buf[sample_i]   = entry["label_idx"]
            metadata[sample_i] = {
                "sample_idx":  sample_i,
                "ch1_path":    ch1_path,
                "col_index":   col_idx,
                "seg_idx":     entry["seg_idx"],
                "class_label": entry["class_label"],
                "speed_pct":   entry["speed_pct"],
            }

            # PNG uses raw log1p — different scale, same underlying data
            if save_png_flag:
                cls_folder   = sanitize(entry["class_label"])
                speed_folder = f"speed_{entry['speed_pct']}"
                out_dir = images_dir / split / cls_folder / speed_folder
                save_envelope_phase_pngs(spec_log, out_dir,
                                          col_idx, entry["seg_idx"])

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
    print(f"\nEnvelope-spectrum computation done in {fmt_time(elapsed)}")

    # ── Save .pt ────────────────────────────────────────────────────────────
    save_path = features_dir / f"{split}.pt"
    print(f"Saving {split}.pt ...", end=" ", flush=True)
    t0 = time.time()
    torch.save({
        "features":    torch.from_numpy(features_buf),
        "labels":      torch.from_numpy(labels_buf),
        "class_names": CLASS_NAMES,
        "metadata":    metadata,
        "envelope_params": {
            "fs":                 FS,
            "window_samples":     WINDOW_SAMPLES,
            "analytic":           "scipy.signal.hilbert",
            "fft_window":         "hann",
            "fft_length":         WINDOW_SAMPLES,
            "freq_cap_hz":        FREQ_CAP_HZ,
            "n_freq_bins":        N_FREQ_BINS,
            "freq_resolution_hz": FS / WINDOW_SAMPLES,   # 1.0
            "magnitude_form":     "log1p",
            "normalization":      "per_image_zscore",
        },
    }, save_path)
    save_elapsed = time.time() - t0
    file_size = save_path.stat().st_size
    print(f"done in {fmt_time(save_elapsed)}  ({fmt_size(file_size)})")

    counts = np.bincount(labels_buf, minlength=len(CLASS_NAMES))
    print("Per-class counts in saved file:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<22} {counts[i]:>5}")

    return n_samples, file_size, elapsed + save_elapsed


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PRECOMPUTE ENVELOPE-SPECTRUM FEATURES + PER-SEGMENT PANELS - 4 classes")
    print("=" * 70)
    print(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"splits.csv:     {SPLITS_CSV}")
    print(f"denoised root:  {DENOISED_ROOT}")
    print(f"features dir:   {FEATURES_DIR}")
    print(f"images dir:     {IMAGES_DIR}")
    print(f"save PNGs:      {SAVE_PNGS}  (3 phase PNGs per segment, no labeling)")
    print(f"torch threads:  {torch.get_num_threads()}")
    print()
    print("LOCKED ENVELOPE PARAMETERS:")
    print(f"  fs                = {FS}")
    print(f"  window_samples    = {WINDOW_SAMPLES} (1.0 s)")
    print(f"  analytic          = scipy.signal.hilbert")
    print(f"  fft_window        = hann (applied to envelope)")
    print(f"  fft_length        = {WINDOW_SAMPLES} -> 1 Hz resolution")
    print(f"  freq_cap_hz       = {FREQ_CAP_HZ}  -> {N_FREQ_BINS} bins")
    print(f"  magnitude_form    = log1p(|rfft|)")
    print(f"  normalization     = per-image z-score (across 3 phases)")
    print(f"  expected sample shape: (3, {N_FREQ_BINS}) float32")
    print()
    print("PNG CONVENTIONS:")
    print(f"  filename : colXX_segYYY_chN.png  (N=1..3, no labels)")
    print(f"  content  : one envelope spectrum line plot per phase")
    print(f"  size     : ~400 x 120 px")

    if not os.path.exists(SPLITS_CSV):
        print(f"\nERROR: splits.csv not found at {SPLITS_CSV}")
        sys.exit(1)
    if not os.path.isdir(DENOISED_ROOT):
        print(f"\nERROR: denoised root not found at {DENOISED_ROOT}")
        sys.exit(1)

    splits_df = pd.read_csv(SPLITS_CSV)
    print(f"\nLoaded splits.csv: {len(splits_df)} rows")
    print(f"Splits present : {sorted(splits_df['split'].unique())}")
    print(f"Speeds present : {sorted(splits_df['speed_pct'].unique())}")
    print(f"Classes present: {sorted(splits_df['class_label'].unique())}")

    features_dir = Path(FEATURES_DIR)
    images_dir   = Path(IMAGES_DIR)
    features_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PNGS:
        images_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()
    summary = []

    for split in ("train", "val", "test"):
        n, sz, dt = process_split(
            splits_df, split, features_dir, images_dir,
            save_png_flag=SAVE_PNGS)
        summary.append((f"{split}.pt", n, sz, dt))

    total_elapsed = time.time() - t_total_start

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(f"Total elapsed : {fmt_time(total_elapsed)}")
    print()
    print("Files written:")
    total_bytes = total_samples = 0
    for name, n, sz, dt in summary:
        print(f"  {name:<18} {n:>6} samples  {fmt_size(sz):>9}  ({fmt_time(dt)})")
        total_bytes   += sz
        total_samples += n
    print(f"  {'TOTAL':<18} {total_samples:>6} samples  {fmt_size(total_bytes):>9}")

    if SAVE_PNGS:
        total_png = sum(1 for _ in images_dir.rglob("*.png"))
        print(f"\nPNGs written  : {total_png}")
        print(f"PNG tree root : {images_dir}")

    print()
    print("Sanity check:")
    print("  import torch")
    print(f"  d = torch.load(r'{features_dir / 'train.pt'}',")
    print("                 weights_only=False, mmap=True)")
    print("  print(d['features'].shape, d['labels'].shape)")
    print("  print(d['metadata'][0])")
    print()
    print("Next: run train_envelope_cnn_v1.py")


if __name__ == "__main__":
    main()
