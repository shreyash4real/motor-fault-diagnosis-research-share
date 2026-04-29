"""
STEP 4 — PRECOMPUTE STFT FEATURES + PER-CHANNEL PNG TREE (4-class)
==================================================================
Reads splits.csv and produces two parallel artifacts:

  1. THREE .pt files for training (no augmentation):
        features/train.pt     pure train samples
        features/val.pt       pure val samples
        features/test.pt      pure test samples

  2. PER-CHANNEL PNG TREE for inspection:
        stft_images/<split>/<class>/<speed>/colXX_segYYY_chN.png
        3 PNGs per segment — one for each phase channel (ch1/ch2/ch3).
        No labels, no axes, no title. Jet colormap, raw dB values
        over the range [-50, +10] dB. Native resolution 154 × 149.

REPRESENTATIONS DIVERGE — INTENTIONALLY
---------------------------------------
  PNG     = raw log-magnitude (dB), jet colormap, range [-50, +10]
            Matches how spectrograms are conventionally visualized.
  .pt     = z-score normalized log-magnitude, float32, (3, 154, 149)
            Better for neural network training convergence.
  Both are views of the same underlying STFT — just scaled differently.

LOCKED STFT PARAMETERS (match original Phase 1 exactly)
-------------------------------------------------------
  window_samples = 20000
  nperseg        = 1024
  noverlap       = 896
  boundary       = None, padded = False
  freq_cutoff    = 3000 Hz → 154 freq bins
  per-sample shape = (3, 154, 149) float32
  normalization  = per-image z-score (across all 3 channels)

Usage
-----
    python precompute_stft.py

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

import numpy as np
import pandas as pd
import torch
from scipy.signal import stft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
RAW_ROOT       = r"C:\Project Work\Dataset\Electric\Motor-2"
DENOISED_ROOT  = r"C:\Project Work\Denoised\Motor-2"
SPLITS_CSV    = r"C:\Project Work\outputs\4class\v2_speed_strat\splits.csv"
FEATURES_DIR  = r"C:\Project Work\outputs\4class\v2_speed_strat\features"
IMAGES_DIR    = r"C:\Project Work\outputs\4class\v2_speed_strat\stft_images"
SAVE_PNGS      = True
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "healthy 1",
    "stator short 1",
    "bearing bpfo 3",
    "broken rotor bar",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Locked STFT parameters — do not edit here
FS              = 20_000
WINDOW_SAMPLES  = 20_000
NPERSEG         = 1024
NOVERLAP        = 896
FREQ_CUTOFF_HZ  = 3000

# PNG visualization — raw dB values, jet colormap, matches conventional
# MATLAB / MCSA literature rendering. Easy to eyeball against reference
# spectrograms in papers.
PNG_VMIN_DB, PNG_VMAX_DB = -50.0, 10.0
PNG_CMAP = "jet"


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """'bearing bpfo 3' → 'bearing_bpfo_3' (Windows-safe folder name)."""
    return re.sub(r"\s+", "_", name.strip())


def rewrite_path(raw_path: str) -> str:
    """Map raw-dataset path → denoised-dataset path."""
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


def compute_spectrogram_db(sig_3phase: np.ndarray) -> np.ndarray:
    """
    (3, 20000) signal → (3, 154, 149) float32 log-magnitude dB spectrogram.
    NOT normalized. Used as the source for both PNG rendering (raw dB) and
    the z-score step that produces the training tensor.
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
    """
    Per-image z-score across all 3 channels together. Preserves relative
    magnitudes between phases. Produces the training tensor.
    """
    mean = spec_db.mean()
    std  = spec_db.std() + 1e-8
    return (spec_db - mean) / std


def save_channel_pngs(spec_db: np.ndarray, out_dir: Path,
                      col_idx: int, seg_idx: int) -> None:
    """
    Save 3 PNGs — one per phase channel — from the raw dB spectrogram.
    No axes, title, or labels. Native 154 × 149 pixels, jet colormap.
    """
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
    """Process one split; write .pt + PNGs. No augmentation."""
    print()
    print("─" * 70)
    print(f"Processing: {split.upper()}")
    print("─" * 70)

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

    if spec_shape is None:
        first_key = next(iter(groups))
        full = load_full_three_phases(*first_key)
        spec_shape = compute_spectrogram_db(full[:, :WINDOW_SAMPLES]).shape
    print(f"Per-sample shape   : {spec_shape}")
    print(f"Samples to process : {n_samples:,}")
    print(f"Unique columns     : {n_columns}")

    features_buf = np.zeros((n_samples, *spec_shape), dtype=np.float32)
    labels_buf   = np.zeros((n_samples,), dtype=np.int64)
    metadata     = [None] * n_samples
    print(f"Pre-allocated      : {fmt_size(features_buf.nbytes)}")
    if save_png_flag:
        print(f"PNGs will be written under: {images_dir / split}")
        print(f"   3 PNGs per segment → {n_samples * 3:,} files total")
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
                spec_db = compute_spectrogram_db(sig)
            except Exception as e:
                print(f"  ERROR STFT col {col_idx} seg {entry['seg_idx']}: {e}")
                sample_i += 1
                continue

            # Training tensor uses z-scored version
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

            # PNG uses raw dB version — different scale, same underlying data
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
    print(f"\nSpectrogram computation done in {fmt_time(elapsed)}")

    # ── Save .pt ────────────────────────────────────────────────────────────
    save_path = features_dir / f"{split}.pt"
    print(f"Saving {split}.pt ...", end=" ", flush=True)
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
    print(f"done in {fmt_time(save_elapsed)}  ({fmt_size(file_size)})")

    counts = np.bincount(labels_buf, minlength=len(CLASS_NAMES))
    print("Per-class counts in saved file:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<22} {counts[i]:>5}")

    return n_samples, file_size, elapsed + save_elapsed, spec_shape


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PRECOMPUTE STFT FEATURES + PER-CHANNEL PNGS — 4 classes")
    print("=" * 70)
    print(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"splits.csv:     {SPLITS_CSV}")
    print(f"denoised root:  {DENOISED_ROOT}")
    print(f"features dir:   {FEATURES_DIR}")
    print(f"images dir:     {IMAGES_DIR}")
    print(f"save PNGs:      {SAVE_PNGS}  (3 per segment, no labeling)")
    print(f"torch threads:  {torch.get_num_threads()}")
    print()
    print("LOCKED STFT PARAMETERS:")
    print(f"  nperseg={NPERSEG}, noverlap={NOVERLAP} (87.5% overlap)")
    print(f"  boundary=None, padded=False")
    print(f"  freq_cutoff={FREQ_CUTOFF_HZ} Hz → 154 freq bins")
    print(f"  expected sample shape: (3, 154, 149) float32")
    print(f"  .pt normalization: per-image z-score (across 3 channels)")
    print()
    print("PNG CONVENTIONS:")
    print(f"  filename : colXX_segYYY_ch{{1,2,3}}.png  (no labels)")
    print(f"  colormap : {PNG_CMAP}")
    print(f"  range    : [{PNG_VMIN_DB}, {PNG_VMAX_DB}] dB  (raw log-magnitude)")
    print(f"  size     : native (154 × 149 pixels)")

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
    spec_shape = None

    for split in ("train", "val", "test"):
        n, sz, dt, spec_shape = process_split(
            splits_df, split, features_dir, images_dir,
            save_png_flag=SAVE_PNGS, spec_shape=spec_shape)
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
    print("Next: run train_stft_cnn.py")


if __name__ == "__main__":
    main()