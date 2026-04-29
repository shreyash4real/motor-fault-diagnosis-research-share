"""
STEP 4 — PRECOMPUTE CLARKE IMAGES + PNG TREE (4-class)
======================================================
Reads splits.csv and produces two parallel artifacts:

  1. THREE .pt files for training (no augmentation):
        clarke_features/train.pt
        clarke_features/val.pt
        clarke_features/test.pt

  2. PER-SEGMENT PNG TREE for inspection:
        clarke_images/<split>/<class>/<speed>/colXX_segYYY.png
        ONE PNG per segment (unlike STFT's three-per-segment) because
        Clarke produces a single-channel density image. No labels,
        no axes, no title. Viridis colormap over [0, 1].

REPRESENTATIONS DIVERGE — INTENTIONALLY
---------------------------------------
  PNG     = log1p of raw bin counts, per-image max-normalized to [0, 1],
            viridis colormap. Matches how Park's-vector trajectories
            are conventionally rendered in the MCSA literature.
  .pt     = the same density image after per-image z-score, float32,
            (1, 128, 128). Better for network training convergence.
  Both are views of the same 2D histogram, scaled differently.

CLARKE TRANSFORM (amplitude-invariant form)
-------------------------------------------
  i_alpha = (2/3) * (i_a - 0.5*i_b - 0.5*i_c)
  i_beta  = (2/3) * ((sqrt(3)/2) * i_b - (sqrt(3)/2) * i_c)

  For a balanced 3-phase system i_alpha is numerically equal to i_a,
  which makes debugging easier than the power-invariant form.

LOCKED PARAMETERS
-----------------
  window_samples = 20000           (1.0 s at 20 kHz, matches STFT)
  img_size       = 128             (native — NOT upsampled)
  density_xform  = log1p           (compresses bin-count dynamic range)
  extent         = per-sample symmetric  (± max(|i_alpha|, |i_beta|))
  normalization  = max-norm -> z-score (per image)
  channels       = 1
  per-sample shape = (1, 128, 128) float32

Usage
-----
    python precompute_clarke.py

Requires: pandas, numpy, torch, matplotlib
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
# ─── CONFIGURE ────────────────────────────────────────────────────────────────
RAW_ROOT       = r"C:\Project Work\Dataset\Electric\Motor-2"
DENOISED_ROOT  = r"C:\Project Work\Denoised\Motor-2"
SPLITS_CSV     = r"C:\Project Work\outputs\4class\v2_speed_strat\splits.csv"
FEATURES_DIR   = r"C:\Project Work\outputs\4class\v2_speed_strat\clarke_features"
IMAGES_DIR     = r"C:\Project Work\outputs\4class\v2_speed_strat\clarke_images"
SAVE_PNGS      = True
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "healthy 1",
    "stator short 1",
    "bearing bpfo 3",
    "broken rotor bar",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Locked Clarke parameters — do not edit here
FS              = 20_000
WINDOW_SAMPLES  = 20_000        # 1.0 s, matches STFT branch
IMG_SIZE        = 128           # native resolution, no train-time upsample
# Bin extent is per-sample symmetric: [-max(|i_alpha|, |i_beta|), +max]
# on both axes. Keeps the trajectory scale-invariant across load/speed.

# PNG visualization — log1p density shown on [0, 1] after max-norm
PNG_CMAP        = "viridis"

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


# Clarke transform coefficients (amplitude-invariant form)
_C_ALPHA = np.array([1.0, -0.5, -0.5], dtype=np.float32) * (2.0 / 3.0)
_C_BETA  = np.array([0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2],
                    dtype=np.float32) * (2.0 / 3.0)


def clarke_transform(sig_3phase: np.ndarray) -> np.ndarray:
    """
    (3, N) time-domain currents -> (2, N) (i_alpha, i_beta) trajectory.
    Amplitude-invariant form: i_alpha equals i_a for balanced 3-phase.
    """
    i_alpha = (_C_ALPHA @ sig_3phase).astype(np.float32)
    i_beta  = (_C_BETA  @ sig_3phase).astype(np.float32)
    return np.stack([i_alpha, i_beta])


def compute_clarke_density(sig_3phase: np.ndarray) -> np.ndarray:
    """
    (3, 20000) -> (1, IMG_SIZE, IMG_SIZE) float32 log-density image in
    [0, 1], per-image max-normalized. NOT yet z-scored — this is the
    PNG-ready representation and also the source for the z-score step
    that produces the training tensor. Mirrors the STFT script's
    (spec_db, zscore_normalize(spec_db)) split exactly.
    """
    ab = clarke_transform(sig_3phase)
    i_alpha, i_beta = ab[0], ab[1]

    # Square extent preserves the trajectory's aspect ratio
    extent = float(max(np.abs(i_alpha).max(), np.abs(i_beta).max()))
    if extent <= 0:
        extent = 1e-8                                  # dead-phase guard

    hist, _, _ = np.histogram2d(
        i_alpha, i_beta, bins=IMG_SIZE,
        range=[[-extent, extent], [-extent, extent]]
    )
    # histogram2d bins i_alpha along axis 0; transpose so rows=beta, cols=alpha
    img = np.log1p(hist.T.astype(np.float32))
    mx  = img.max()
    if mx > 0:
        img = img / mx

    return img[np.newaxis, :, :]                       # (1, H, W)


def zscore_normalize(img_density: np.ndarray) -> np.ndarray:
    """Per-image z-score. Gives the network zero-mean unit-variance input."""
    mean = img_density.mean()
    std  = img_density.std() + 1e-8
    return (img_density - mean) / std


def save_clarke_png(img_density: np.ndarray, out_dir: Path,
                    col_idx: int, seg_idx: int) -> None:
    """
    Save ONE PNG per segment from the log1p max-normalized density.
    No axes, title, or labels. Native IMG_SIZE x IMG_SIZE pixels,
    viridis colormap over [0, 1].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = img_density[0]
    fname = f"col{col_idx:02d}_seg{seg_idx:03d}.png"
    plt.imsave(out_dir / fname, img,
               cmap=PNG_CMAP, vmin=0.0, vmax=1.0, origin="lower")

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
        spec_shape = compute_clarke_density(full[:, :WINDOW_SAMPLES]).shape
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
                clarke_density = compute_clarke_density(sig)
            except Exception as e:
                print(f"  ERROR Clarke col {col_idx} seg {entry['seg_idx']}: {e}")
                sample_i += 1
                continue

            # Training tensor uses z-scored version
            features_buf[sample_i] = zscore_normalize(clarke_density)

            # Training tensor uses z-scored version
            features_buf[sample_i] = zscore_normalize(clarke_density)
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
                save_clarke_png(clarke_density, out_dir, col_idx, entry["seg_idx"])

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
        "clarke_params": {
            "fs":              FS,
            "window_samples":  WINDOW_SAMPLES,
            "img_size":        IMG_SIZE,
            "transform":       "amplitude_invariant_clarke",
            "density_xform":   "log1p",
            "extent":          "per_sample_symmetric_max",
            "normalization":   "maxnorm_then_zscore",
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
    print("LOCKED CLARKE PARAMETERS:")
    print(f"  window_samples = {WINDOW_SAMPLES}  (1.0 s at {FS} Hz)")
    print(f"  img_size       = {IMG_SIZE}  (native, no train-time upsample)")
    print(f"  transform      = amplitude-invariant Clarke")
    print(f"  density_xform  = log1p")
    print(f"  extent         = per-sample symmetric max")
    print(f"  normalization  = max-norm then per-image z-score")
    print(f"  expected sample shape: (1, {IMG_SIZE}, {IMG_SIZE}) float32")
    print()
    print("PNG CONVENTIONS:")
    print(f"  filename : colXX_segYYY.png  (ONE per segment, no labels)")
    print(f"  colormap : {PNG_CMAP}")
    print(f"  range    : [0, 1]  (log1p density after max-norm)")
    print(f"  size     : native ({IMG_SIZE} x {IMG_SIZE} pixels)")

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
        n, sz, dt, spec_shape = process_split(splits_df, split, features_dir, images_dir,
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