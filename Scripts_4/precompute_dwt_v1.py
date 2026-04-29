"""
STEP 4 — PRECOMPUTE DWT COEFFICIENTS + PNG TREE (4-class)
=========================================================
Sibling of precompute_stft_v2.py and precompute_clarke_v2.py.
Reads the locked v2_speed_strat splits and produces two artefacts:

  1. THREE .pt files for training (no augmentation):
        dwt_features/train.pt
        dwt_features/val.pt
        dwt_features/test.pt

  2. PER-SEGMENT PNG TREE for inspection:
        dwt_images/<split>/<class>/<speed>/colXX_segYYY.png
        One diagnostic panel per segment showing |cD_j| envelopes for
        each kept level, stacked vertically. No labels, no axes.

REPRESENTATION
--------------
  db8 DWT to decomposition level J = 10 (max for 20,000 samples).
  cD1 (5 - 10 kHz) dropped because it lies above the 5 kHz denoise
  cut-off and carries only filter leakage.

  KEPT sub-bands (10 of them):
      cD2   2500 - 5000   Hz   5011 coefs
      cD3   1250 - 2500   Hz   2513 coefs
      cD4    625 - 1250   Hz   1264 coefs
      cD5    312 -  625   Hz    639 coefs
      cD6    156 -  312   Hz    327 coefs
      cD7     78 -  156   Hz    171 coefs   upper BPFO sideband @100%
      cD8     39 -   78   Hz     93 coefs   line fundamental + BPFO
      cD9     20 -   39   Hz     54 coefs   BPFO at 50%, lower sidebands
      cD10    10 -   20   Hz     34 coefs   shaft rotation
      cA10     0 -   10   Hz     34 coefs   slip-modulated content

NORMALIZATION
-------------
  Per-level z-score, computed per-segment across the 3 current channels.
  Done per-level (not per-segment global) because cD8 energy dwarfs cD2
  energy by orders of magnitude — a single global z-score would let the
  line fundamental band wash out every other band.

.PT FILE STRUCTURE
------------------
  {
      "features_per_level": [T2, T3, ..., T10, TA10]        # 10 tensors
                              each shaped (N_samples, 3, L_j)
      "level_names":        ["cD2", ..., "cD10", "cA10"],
      "level_bands_hz":     [(2500,5000), ..., (0, 9.77)],
      "level_lengths":      [5011, 2513, ..., 34, 34],
      "labels":             (N,) int64,
      "class_names":        [...],
      "metadata":           [per-sample dicts],
      "dwt_params":         {...},
  }

Usage
-----
    python precompute_dwt_v1.py

Requires: pandas, numpy, torch, matplotlib, pywavelets (pywt)
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
import pywt
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
RAW_ROOT       = r"C:\Project Work\Dataset\Electric\Motor-2"
DENOISED_ROOT  = r"C:\Project Work\Denoised\Motor-2"
SPLITS_CSV     = r"C:\Project Work\outputs\4class\v2_speed_strat\splits.csv"
FEATURES_DIR   = r"C:\Project Work\outputs\4class\v2_speed_strat\dwt\features"
IMAGES_DIR     = r"C:\Project Work\outputs\4class\v2_speed_strat\dwt\images"
SAVE_PNGS      = True
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "healthy 1",
    "stator short 1",
    "bearing bpfo 3",
    "broken rotor bar",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ─── LOCKED DWT PARAMETERS ────────────────────────────────────────────────────
FS              = 20_000
WINDOW_SAMPLES  = 20_000        # 1.0 s, matches STFT and Clarke branches
WAVELET         = "db8"
DECOMP_LEVEL    = 10            # max for 20,000 samples with db8
DROP_DETAIL_LEVELS = {1}        # cD1 = 5-10 kHz, above denoise cut-off
DWT_MODE        = "symmetric"   # boundary handling
# ──────────────────────────────────────────────────────────────────────────────


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


def get_level_info() -> list[dict]:
    """
    Return a list describing each KEPT DWT sub-band.
    Each entry: {'name', 'j', 'band_hz', 'length', 'is_approx'}
    Order matches pywt.wavedec output after we reverse and filter.

    Output order (what we store):
        [cD2, cD3, cD4, cD5, cD6, cD7, cD8, cD9, cD10, cA10]
    i.e. detail bands from high-f to low-f, then approximation.
    """
    # Lengths derived analytically for db8 (filter len 16) on 20,000
    # samples with mode='symmetric': L_{j+1} = floor((L_j + 15)/2),
    # starting L_1 = floor((20000+15)/2) = 10007.
    lengths = [10007, 5011, 2513, 1264, 639, 327, 171, 93, 54, 34]

    level_info = []
    # Detail bands cD1..cD10
    for j in range(1, DECOMP_LEVEL + 1):
        if j in DROP_DETAIL_LEVELS:
            continue
        lo = FS / 2 ** (j + 1)
        hi = FS / 2 ** j
        level_info.append({
            "name":      f"cD{j}",
            "j":         j,
            "band_hz":   (round(lo, 2), round(hi, 2)),
            "length":    lengths[j - 1],
            "is_approx": False,
        })
    # Approximation cA_J
    lo_a = 0.0
    hi_a = FS / 2 ** (DECOMP_LEVEL + 1)
    level_info.append({
        "name":      f"cA{DECOMP_LEVEL}",
        "j":         DECOMP_LEVEL,
        "band_hz":   (round(lo_a, 2), round(hi_a, 2)),
        "length":    lengths[DECOMP_LEVEL - 1],  # cA length matches cD at deepest
        "is_approx": True,
    })
    return level_info


def compute_dwt_per_channel(sig_3phase: np.ndarray,
                             level_info: list[dict]) -> list[np.ndarray]:
    """
    Compute db8 DWT to level J for each of the 3 phase currents and
    return the KEPT sub-bands, z-scored per-level across the 3 channels.

    Input  : (3, WINDOW_SAMPLES) float32
    Output : list of 10 arrays, one per kept sub-band,
             each shaped (3, L_j) float32, z-scored per sub-band
             (mean/std over all 3*L_j values in that sub-band).
    """
    per_phase_coeffs = []
    for phase_sig in sig_3phase:
        # pywt.wavedec returns [cAJ, cDJ, cDJ-1, ..., cD1]
        coeffs = pywt.wavedec(phase_sig, WAVELET,
                              level=DECOMP_LEVEL, mode=DWT_MODE)
        # reorder to [cD1, cD2, ..., cDJ, cAJ] for easier indexing
        cD_reversed = list(coeffs[1:])[::-1]   # [cD1, cD2, ..., cDJ]
        cA_J        = coeffs[0]
        per_phase_coeffs.append(cD_reversed + [cA_J])

    # per_phase_coeffs: list of 3 lists, each with DECOMP_LEVEL+1 arrays
    # Stack per level across the 3 phases, then apply per-level z-score.
    out_per_level: list[np.ndarray] = []

    for info in level_info:
        if info["is_approx"]:
            src_idx = DECOMP_LEVEL        # last slot holds cA_J
        else:
            src_idx = info["j"] - 1       # cD_j sits at index j-1
        stacked = np.stack([
            per_phase_coeffs[0][src_idx],
            per_phase_coeffs[1][src_idx],
            per_phase_coeffs[2][src_idx],
        ]).astype(np.float32)            # (3, L_j)

        mean = float(stacked.mean())
        std  = float(stacked.std()) + 1e-8
        stacked = (stacked - mean) / std
        out_per_level.append(stacked)

    return out_per_level


def save_dwt_panel_png(levels_3ch: list[np.ndarray],
                        level_info: list[dict],
                        out_dir: Path,
                        col_idx: int, seg_idx: int) -> None:
    """
    One diagnostic panel per segment. For each kept sub-band, plot the
    magnitude envelope |cD_j| of phase 1 on its own row, labelled by
    band. No title, axis ticks, or legends — matches the minimalist
    convention of the STFT / Clarke PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(levels_3ch)
    # Figure height: ~0.4 inch per row; width fits the longest array
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 0.45 * n_rows),
                              sharex=False)
    if n_rows == 1:
        axes = [axes]
    for ax, band_tensor, info in zip(axes, levels_3ch, level_info):
        env = np.abs(band_tensor[0])                 # phase-1 envelope
        ax.plot(env, linewidth=0.4, color="#222")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=0, y=0.1)
        # Small corner tag so the panel is still decodable by eye
        ax.text(0.01, 0.8, info["name"],
                transform=ax.transAxes, fontsize=6, color="#555")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.99,
                        bottom=0.01, hspace=0.15)
    fname = f"col{col_idx:02d}_seg{seg_idx:03d}.png"
    plt.savefig(out_dir / fname, dpi=110, facecolor="white")
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
                  save_png_flag: bool,
                  level_info: list[dict]) -> tuple:
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

    print(f"Samples to process : {n_samples:,}")
    print(f"Unique columns     : {n_columns}")
    print(f"Sub-bands stored   : {len(level_info)}")
    print(f"Per-sample shape   : list of "
          f"{len(level_info)} tensors, lengths "
          f"{[info['length'] for info in level_info]}")

    # Pre-allocate one buffer per sub-band
    feature_bufs = [
        np.zeros((n_samples, 3, info["length"]), dtype=np.float32)
        for info in level_info
    ]
    labels_buf = np.zeros((n_samples,), dtype=np.int64)
    metadata   = [None] * n_samples
    total_bytes_preallocated = sum(b.nbytes for b in feature_bufs)
    print(f"Pre-allocated      : {fmt_size(total_bytes_preallocated)}")
    if save_png_flag:
        print(f"PNGs will be written under: {images_dir / split}")
        print(f"  1 panel PNG per segment -> {n_samples:,} files total")
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
                bands = compute_dwt_per_channel(sig, level_info)
            except Exception as e:
                print(f"  ERROR DWT col {col_idx} seg {entry['seg_idx']}: {e}")
                sample_i += 1
                continue

            for lvl_i, band in enumerate(bands):
                feature_bufs[lvl_i][sample_i] = band

            labels_buf[sample_i] = entry["label_idx"]
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
                save_dwt_panel_png(bands, level_info, out_dir,
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
    print(f"\nDWT computation done in {fmt_time(elapsed)}")

    # ── Save .pt ────────────────────────────────────────────────────────────
    save_path = features_dir / f"{split}.pt"
    print(f"Saving {split}.pt ...", end=" ", flush=True)
    t0 = time.time()
    torch.save({
        "features_per_level": [torch.from_numpy(b) for b in feature_bufs],
        "level_names":        [info["name"]     for info in level_info],
        "level_bands_hz":     [info["band_hz"]  for info in level_info],
        "level_lengths":      [info["length"]   for info in level_info],
        "labels":             torch.from_numpy(labels_buf),
        "class_names":        CLASS_NAMES,
        "metadata":           metadata,
        "dwt_params": {
            "fs":             FS,
            "window_samples": WINDOW_SAMPLES,
            "wavelet":        WAVELET,
            "decomp_level":   DECOMP_LEVEL,
            "drop_detail":    sorted(DROP_DETAIL_LEVELS),
            "mode":           DWT_MODE,
            "normalization":  "per_level_zscore_over_3_phases",
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
    print("PRECOMPUTE DWT FEATURES + PER-SEGMENT PANELS - 4 classes")
    print("=" * 70)
    print(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"splits.csv:     {SPLITS_CSV}")
    print(f"denoised root:  {DENOISED_ROOT}")
    print(f"features dir:   {FEATURES_DIR}")
    print(f"images dir:     {IMAGES_DIR}")
    print(f"save PNGs:      {SAVE_PNGS}")
    print(f"torch threads:  {torch.get_num_threads()}")
    print()
    print("LOCKED DWT PARAMETERS:")
    print(f"  wavelet         = {WAVELET}")
    print(f"  decomp level    = {DECOMP_LEVEL}")
    print(f"  boundary mode   = {DWT_MODE}")
    print(f"  dropped details = {sorted(DROP_DETAIL_LEVELS)} "
          f"(above 5 kHz denoise cut-off)")

    level_info = get_level_info()
    print()
    print("KEPT SUB-BANDS:")
    print(f"  {'name':<6} {'length':<8} {'band (Hz)':<20}")
    for info in level_info:
        lo, hi = info["band_hz"]
        print(f"  {info['name']:<6} {info['length']:<8} {lo:>8.2f} - {hi:<10.2f}")
    total_coefs = sum(info["length"] for info in level_info)
    print(f"  TOTAL coefs/channel/segment: {total_coefs}")
    print(f"  per segment (3 channels): "
          f"{fmt_size(total_coefs * 3 * 4)}")

    if not os.path.exists(SPLITS_CSV):
        print(f"\nERROR: splits.csv not found at {SPLITS_CSV}")
        sys.exit(1)
    if not os.path.isdir(DENOISED_ROOT):
        print(f"\nERROR: denoised root not found at {DENOISED_ROOT}")
        sys.exit(1)

    splits_df = pd.read_csv(SPLITS_CSV)
    print(f"\nLoaded splits.csv: {len(splits_df)} rows")
    print(f"Splits present: {sorted(splits_df['split'].unique())}")
    print(f"Speeds present: {sorted(splits_df['speed_pct'].unique())}")
    print(f"Classes present: {len(splits_df['class_label'].unique())}")

    features_dir = Path(FEATURES_DIR)
    images_dir   = Path(IMAGES_DIR)
    features_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PNGS:
        images_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()
    summary = []

    for split in ["train", "val", "test"]:
        n, sz, dt = process_split(
            splits_df, split, features_dir, images_dir,
            save_png_flag=SAVE_PNGS, level_info=level_info)
        summary.append((f"{split}.pt", n, sz, dt))

    total_elapsed = time.time() - t_total_start

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(f"Total elapsed: {fmt_time(total_elapsed)}")
    print()
    print("Files written:")
    total_bytes = total_samples = 0
    for name, n, sz, dt in summary:
        print(f"  {name:<18} {n:>6} samples  {fmt_size(sz):>9}  ({fmt_time(dt)})")
        total_bytes  += sz
        total_samples += n
    print(f"  {'TOTAL':<18} {total_samples:>6} samples  {fmt_size(total_bytes):>9}")
    print()
    print("Sanity check:")
    print("  import torch")
    print("  d = torch.load(r'" + str(Path(FEATURES_DIR) / 'train.pt')
          + "', weights_only=False, mmap=True)")
    print("  print(len(d['features_per_level']), 'sub-bands')")
    print("  for name, t in zip(d['level_names'], d['features_per_level']):")
    print("      print(f'  {name:<5} {tuple(t.shape)}')")
    print("  print(d['labels'].shape)")
    print("  print(d['metadata'][0])")
    print()
    print("Next: run train_dwt_cnn.py to train the multi-branch 1D CNN.")


if __name__ == "__main__":
    main()
