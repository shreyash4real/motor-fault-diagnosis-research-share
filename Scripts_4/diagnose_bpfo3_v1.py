"""
STEP 6 — BPFO-3 MISCLASSIFICATION DIAGNOSTIC (4-class)
======================================================
Follow-up to the STFTFreqNet v1/v2 redesigns, both of which failed to
beat the E4 Modified AlexNet baseline. A cross-reference of misclassified
samples across five trained models (STFT AlexNet, STFTFreqNet v1/v2,
DWT multibranch, Envelope) revealed that Motor-2 `bearing bpfo 3` at
100% speed, `col_index=5`, is misclassified identically as `healthy 1`
on every test segment by every model — across STFT, DWT, AND envelope
representations.

This script characterises that column at the raw-signal level and
compares it to the other six bpfo-3@100% columns (all train/val) to
decide whether:

  (A) col5@100% genuinely lacks the BPFO modulation present in the
      other bpfo-3@100% columns  — data-level ceiling, paper framing.
  (B) col5@100% has BPFO modulation but our features don't capture it
      — feature-engineering opportunity (MCSA-specific precompute).
  (C) col5@100% is fine and some other systemic issue is at play.

Three parts, all reading existing artefacts — no training, no splits
change, no `.pt` regeneration:

  Part 1  Aggregate misclassified samples across all solo runs.
          Quantify how much of each model's bpfo-3 error concentrates
          on col5@100%. Find the always-wrong intersection.

  Part 2  Full-length (15-second) raw-signal characterisation for
          every bpfo-3@100% column + healthy@100% references. Compute
          line-fundamental amplitude, envelope spectrum at BPFO
          harmonics, MCSA sideband energy, basic stats. Plot.

  Part 3  Cross-representation cosine similarity of col5@100% segments
          against healthy@100% and against other bpfo-3@100% (train)
          in each precomputed feature space (STFT, DWT, Clarke, env).

Outputs to Outputs/4class/diagnostics/.

Usage
-----
    python diagnose_bpfo3_v1.py
"""

from __future__ import annotations

import sys
import re
import warnings
from collections import defaultdict
from pathlib import Path
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT  = Path(r"C:\Project Work")
TRAINING_ROOT = PROJECT_ROOT / "Outputs" / "4class" / "training"
SPLITS_CSV    = PROJECT_ROOT / "Outputs" / "4class" / "v2_speed_strat" / "splits.csv"
FEATURES_V2   = PROJECT_ROOT / "Outputs" / "4class" / "v2_speed_strat"
DENOISED_ROOT = PROJECT_ROOT / "Denoised" / "Motor-2"

DIAG_DIR = PROJECT_ROOT / "Outputs" / "4class" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants / project invariants
FS                  = 20_000                 # Hz
LINE_FUND_HZ        = 50.0                   # VFD-sourced 3-phase
BPFO_HZ_BY_SPEED    = {50: 44.0, 75: 66.0, 100: 87.0}
BPFO_SEARCH_HALF_HZ = 2.0                    # look in [f-2, f+2] for peak

# Runs to include in aggregate + intersection. Excludes ensembles and the
# healthy-subsampled runs (known to hurt; not comparable to full-data runs).
EXCLUDED_NAMES = {
    "v1",
    "B_v2_healthy_sub",
    "C_v1_healthy_sub",
}
EXCLUDED_PREFIXES = ("ENSEMBLE_",)

# Target-class focus
TARGET_CLASS = "bearing bpfo 3"
TARGET_SPEED = 100
TARGET_COL   = 5


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 — misclassification aggregate
# ═══════════════════════════════════════════════════════════════════════════

def list_solo_runs() -> list[Path]:
    """Return subdirectories under TRAINING_ROOT that look like solo runs."""
    runs = []
    for d in sorted(TRAINING_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if d.name in EXCLUDED_NAMES:
            continue
        if any(d.name.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        runs.append(d)
    return runs


def find_misclass_file(run_dir: Path) -> Path | None:
    """Some runs used suffixed filenames (e.g. misclassified_samples_dwt.csv,
    misclassified_samples_stft.csv, misclassified_samples_clarke.csv). Accept
    any `misclassified_samples*.csv` in the run folder.
    """
    candidates = sorted(run_dir.glob("misclassified_samples*.csv"))
    # Prefer the unsuffixed name if present (some runs write both).
    for c in candidates:
        if c.name == "misclassified_samples.csv":
            return c
    return candidates[0] if candidates else None


def load_misclass_long(runs: list[Path]) -> pd.DataFrame:
    """
    Concatenate all runs' misclassified CSVs into one long-format DataFrame.
    Expected columns: run_name, true_class, pred_class, ch1_path, col_index,
    seg_idx, speed_pct.
    """
    rows = []
    for run in runs:
        mc_path = find_misclass_file(run)
        if mc_path is None:
            print(f"  [skip] no misclassified file in {run.name}")
            continue
        try:
            df = pd.read_csv(mc_path)
        except Exception as e:
            print(f"  [error] failed to read {mc_path}: {e}")
            continue
        df = df.copy()
        df["run_name"] = run.name
        required = {"true_class", "pred_class", "ch1_path", "col_index",
                    "seg_idx", "speed_pct"}
        missing = required - set(df.columns)
        if missing:
            print(f"  [warn] {run.name}: missing columns {missing} — skipping")
            continue
        rows.append(df[list(required) + ["run_name"]])
        print(f"  [ok]  {run.name}: {len(df)} errors")
    if not rows:
        raise RuntimeError("No misclassified CSVs found.")
    return pd.concat(rows, ignore_index=True)


def part1_aggregate(runs: list[Path]) -> pd.DataFrame:
    print("=" * 70)
    print("PART 1 — aggregating misclassifications across runs")
    print("=" * 70)
    mc = load_misclass_long(runs)
    print(f"\nTotal rows: {len(mc)}")

    # Focus on bpfo-3 true-class errors
    bp = mc[mc["true_class"] == TARGET_CLASS].copy()
    print(f"bpfo-3 true-class errors: {len(bp)} across {bp['run_name'].nunique()} runs")

    # Table: rows = (speed_pct, col_index), cols = run_name, values = error count
    pivot = (bp.groupby(["speed_pct", "col_index", "run_name"])
               .size()
               .unstack("run_name", fill_value=0)
               .sort_index())
    pivot["total_across_runs"] = pivot.sum(axis=1)
    pivot_path = DIAG_DIR / "bpfo3_error_counts_by_column.csv"
    pivot.to_csv(pivot_path)
    print(f"  → {pivot_path.name}")

    # Error share: col5@100% as fraction of each run's bpfo-3 errors
    share_lines = ["col5@100% share of bearing-bpfo-3 errors per run",
                    "(true_class=bearing bpfo 3; counted as segment errors)",
                    "=" * 70]
    totals = bp.groupby("run_name").size()
    col5_err = (bp[(bp["col_index"] == TARGET_COL) &
                   (bp["speed_pct"] == TARGET_SPEED)]
                  .groupby("run_name").size())
    for run_name in sorted(totals.index):
        t = int(totals.get(run_name, 0))
        c = int(col5_err.get(run_name, 0))
        pct = 100 * c / t if t else 0.0
        share_lines.append(f"  {run_name:<36} col5@100% = {c:>3}/{t:<3}  ({pct:5.1f}%)")
    share_lines.append("")
    share_lines.append(f"Test column composition for bpfo-3@100%: only col_index={TARGET_COL} "
                       f"(see splits.csv; v2 speed-stratified split at seed=42).")
    share_path = DIAG_DIR / "bpfo3_error_share.txt"
    share_path.write_text("\n".join(share_lines), encoding="utf-8")
    print(f"  → {share_path.name}")

    # Always-wrong intersection (bpfo-3 segments missed by every solo run)
    run_names = bp["run_name"].unique()
    bp["key"] = list(zip(bp["ch1_path"], bp["col_index"].astype(int),
                          bp["seg_idx"].astype(int)))
    key_runs = bp.groupby("key")["run_name"].nunique()
    always_keys = key_runs[key_runs == len(run_names)].index
    always = (bp[bp["key"].isin(always_keys)]
                .drop_duplicates(subset=["key"])
                [["ch1_path", "col_index", "seg_idx", "speed_pct", "true_class"]]
                .sort_values(["speed_pct", "col_index", "seg_idx"]))
    always_path = DIAG_DIR / "bpfo3_always_wrong_samples.csv"
    always.to_csv(always_path, index=False)
    print(f"  → {always_path.name}: "
          f"{len(always)} bpfo-3 segments missed by ALL {len(run_names)} solo runs")
    return bp


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — raw-signal characterisation at 100% speed
# ═══════════════════════════════════════════════════════════════════════════

from functools import lru_cache

@lru_cache(maxsize=32)
def _read_denoised_file(cls: str, speed: int, ch: int) -> pd.DataFrame:
    """Cache the full denoised CSV (one per class/speed/channel)."""
    filename = f"Electric_Motor-2_{speed}_time-{cls}-ch{ch}.csv"
    path = DENOISED_ROOT / str(speed) / cls / filename
    return pd.read_csv(path)


def read_denoised_col(cls: str, speed: int, col_index: int, ch: int) -> np.ndarray:
    """Load one phase of one column from the denoised CSVs. Returns (N,) float32."""
    df = _read_denoised_file(cls, speed, ch)
    return df.iloc[:, col_index + 1].values.astype(np.float32)


def envelope_spectrum(sig: np.ndarray, fs: int = FS
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Full-length Hilbert envelope spectrum. Returns (freqs_hz, mag).
      - detrend input
      - |hilbert(x)| = envelope
      - detrend envelope (remove DC)
      - Hann window
      - rfft → magnitude
    """
    x = sig - sig.mean()
    env = np.abs(hilbert(x)).astype(np.float64)
    env = env - env.mean()
    env = env * np.hanning(len(env))
    mag = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), d=1 / fs)
    return freqs.astype(np.float32), mag.astype(np.float32)


def peak_in_window(freqs: np.ndarray, mag: np.ndarray,
                    center_hz: float, half_hz: float) -> float:
    lo, hi = center_hz - half_hz, center_hz + half_hz
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0
    return float(mag[m].max())


def energy_in_band(freqs: np.ndarray, mag: np.ndarray,
                    lo_hz: float, hi_hz: float) -> float:
    m = (freqs >= lo_hz) & (freqs <= hi_hz)
    return float(np.sum(mag[m] ** 2))


def linear_spectrum(sig: np.ndarray, fs: int = FS
                     ) -> tuple[np.ndarray, np.ndarray]:
    x = sig - sig.mean()
    x = x * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    return freqs.astype(np.float32), mag.astype(np.float32)


def spectral_centroid(sig: np.ndarray, max_hz: float = 3000.0) -> float:
    freqs, mag = linear_spectrum(sig)
    m = freqs <= max_hz
    f = freqs[m]
    s = mag[m]
    if s.sum() == 0:
        return 0.0
    return float((f * s).sum() / s.sum())


def get_column_indices_per_split(class_name: str, speed: int
                                  ) -> dict[str, list[int]]:
    """
    Parse splits.csv and return {split: [col_index, ...]} for the given
    (class, speed). Deduplicates across the 3 per-channel rows.

    splits.csv schema: full_path, class_label, fault_label, speed_pct,
    col_index, split, n_segments.
    """
    df = pd.read_csv(SPLITS_CSV)
    sub = df[(df["class_label"] == class_name) & (df["speed_pct"] == speed)]
    out: dict[str, list[int]] = defaultdict(list)
    for _, row in sub.iterrows():
        out[row["split"]].append(int(row["col_index"]))
    return {k: sorted(set(v)) for k, v in out.items()}


def part2_column_signatures() -> pd.DataFrame:
    print()
    print("=" * 70)
    print("PART 2 — raw-signal characterisation (bpfo-3 @ 100% + healthy @ 100%)")
    print("=" * 70)

    splits_bpfo = get_column_indices_per_split(TARGET_CLASS, TARGET_SPEED)
    splits_hlth = get_column_indices_per_split("healthy 1", TARGET_SPEED)
    print(f"bpfo-3@100% cols by split: {dict(splits_bpfo)}")
    print(f"healthy@100% cols by split: {dict(splits_hlth)}")

    # Flatten with (class, col_idx, split) records, process bpfo-3 all columns
    # + a healthy reference (first val col) + healthy cols involved in
    # bidirectional confusion (col16 at 100% per the Explore report).
    records = []
    for split, cols in splits_bpfo.items():
        for c in cols:
            records.append(("bearing bpfo 3", c, split))

    # Reference set of healthy cols: pick the first from each split so we
    # get train/val/test diversity without exploding the stat table.
    for split, cols in splits_hlth.items():
        if cols:
            records.append(("healthy 1", cols[0], split))

    # Targeted healthy references called out in Part 1: cols 14 and 16 at 100%
    # were predicted as bpfo-3 by several models (reverse confusion). Include
    # them if present in any split so they land in bpfo3_column_signatures.csv.
    all_healthy_cols = sum(splits_hlth.values(), [])
    existing_healthy_cols = [c for (cls_, c, _) in records if cls_ == "healthy 1"]
    for extra in (14, 16):
        if extra in all_healthy_cols and extra not in existing_healthy_cols:
            for split, cols in splits_hlth.items():
                if extra in cols:
                    records.append(("healthy 1", extra, split))
                    existing_healthy_cols.append(extra)
                    break

    rows = []
    bpfo_freq = BPFO_HZ_BY_SPEED[TARGET_SPEED]
    for (cls, col_idx, split) in records:
        for ch in (1, 2, 3):
            sig = read_denoised_col(cls, TARGET_SPEED, col_idx, ch)
            freqs_l, mag_l = linear_spectrum(sig)
            freqs_e, mag_e = envelope_spectrum(sig)

            rms = float(np.sqrt(np.mean(sig ** 2)))
            peak = float(np.max(np.abs(sig)))
            crest = peak / (rms + 1e-12)
            line_amp = peak_in_window(freqs_l, mag_l, LINE_FUND_HZ, 1.0)
            sb_low = peak_in_window(freqs_l, mag_l, LINE_FUND_HZ - bpfo_freq, 1.0)
            sb_high = peak_in_window(freqs_l, mag_l, LINE_FUND_HZ + bpfo_freq, 1.0)
            sb_energy = energy_in_band(freqs_l, mag_l,
                                        LINE_FUND_HZ - bpfo_freq * 1.2,
                                        LINE_FUND_HZ + bpfo_freq * 1.2) - \
                        energy_in_band(freqs_l, mag_l,
                                        LINE_FUND_HZ - 0.5, LINE_FUND_HZ + 0.5)
            env_bpfo1 = peak_in_window(freqs_e, mag_e, bpfo_freq, BPFO_SEARCH_HALF_HZ)
            env_bpfo2 = peak_in_window(freqs_e, mag_e, bpfo_freq * 2, BPFO_SEARCH_HALF_HZ)
            env_bpfo3 = peak_in_window(freqs_e, mag_e, bpfo_freq * 3, BPFO_SEARCH_HALF_HZ)
            centroid = spectral_centroid(sig)

            rows.append({
                "class":          cls,
                "speed_pct":      TARGET_SPEED,
                "col_index":      col_idx,
                "channel":        ch,
                "split":          split,
                "is_test_col5":   (cls == TARGET_CLASS and col_idx == TARGET_COL),
                "rms":            rms,
                "peak":           peak,
                "crest_factor":   crest,
                "line_fund_50Hz": line_amp,
                "sideband_low":   sb_low,
                "sideband_high":  sb_high,
                "sideband_energy": sb_energy,
                "env_bpfo_h1":    env_bpfo1,
                "env_bpfo_h2":    env_bpfo2,
                "env_bpfo_h3":    env_bpfo3,
                "env_bpfo_sum":   env_bpfo1 + env_bpfo2 + env_bpfo3,
                "centroid_0-3k":  centroid,
            })

    sig_df = pd.DataFrame(rows)
    sig_path = DIAG_DIR / "bpfo3_column_signatures.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"  → {sig_path.name}: {len(sig_df)} (col, channel) rows")

    # Quick textual report: compare col5 vs other bpfo-3@100% on env BPFO h1+h2+h3
    bpfo_df = sig_df[sig_df["class"] == TARGET_CLASS]
    col5_stats = bpfo_df[bpfo_df["col_index"] == TARGET_COL]
    others_stats = bpfo_df[bpfo_df["col_index"] != TARGET_COL]
    print()
    print("envelope BPFO h1+h2+h3 (sum of first 3 harmonic peaks):")
    print(f"  col5@100%               mean = {col5_stats['env_bpfo_sum'].mean():.3f}   "
          f"(channels: {col5_stats['env_bpfo_sum'].values.round(3).tolist()})")
    print(f"  other bpfo-3@100%       mean = {others_stats['env_bpfo_sum'].mean():.3f}   "
          f"(n={len(others_stats)}, "
          f"min={others_stats['env_bpfo_sum'].min():.3f}, "
          f"max={others_stats['env_bpfo_sum'].max():.3f})")
    print("MCSA sideband-energy (f_line ± BPFO, ex. line bin):")
    print(f"  col5@100%               mean = {col5_stats['sideband_energy'].mean():.3e}")
    print(f"  other bpfo-3@100%       mean = {others_stats['sideband_energy'].mean():.3e}")

    plot_col5_vs_others(bpfo_df, sig_df)
    return sig_df


def plot_col5_vs_others(bpfo_df: pd.DataFrame, full_sig_df: pd.DataFrame):
    """4-panel figure comparing col5@100% to other bpfo-3@100% columns."""
    print()
    print("rendering bpfo3_col5_vs_others.png ...")

    bpfo_freq = BPFO_HZ_BY_SPEED[TARGET_SPEED]

    # Load envelope + linear spectra once per (col, ch=1) for plotting
    env_curves = {}  # col_index -> (freqs, mag)
    lin_curves = {}
    for col_idx in sorted(bpfo_df["col_index"].unique()):
        sig = read_denoised_col(TARGET_CLASS, TARGET_SPEED, col_idx, 1)
        env_curves[col_idx] = envelope_spectrum(sig)
        lin_curves[col_idx] = linear_spectrum(sig)

    # Healthy reference (first healthy col at 100%, any split)
    h_cols = full_sig_df[full_sig_df["class"] == "healthy 1"]["col_index"].unique()
    healthy_ref_col = int(h_cols[0]) if len(h_cols) else None
    if healthy_ref_col is not None:
        h_sig = read_denoised_col("healthy 1", TARGET_SPEED, healthy_ref_col, 1)
        env_healthy = envelope_spectrum(h_sig)
        lin_healthy = linear_spectrum(h_sig)
    else:
        env_healthy = lin_healthy = None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ── Panel 1: envelope spectrum 0-500 Hz (all bpfo-3 cols + healthy)
    ax = axes[0, 0]
    for col_idx, (f, m) in env_curves.items():
        m_plot = m / (m.max() + 1e-12)
        style = dict(color="red", lw=1.6, zorder=10, label=f"col{col_idx} (TEST)") \
            if col_idx == TARGET_COL else \
            dict(color="grey", lw=0.9, alpha=0.7,
                  label=f"col{col_idx}" if col_idx == min([c for c in env_curves if c != TARGET_COL]) else None)
        mask = f <= 500
        ax.plot(f[mask], m_plot[mask], **style)
    if env_healthy is not None:
        f, m = env_healthy
        mask = f <= 500
        ax.plot(f[mask], (m / (m.max() + 1e-12))[mask],
                 color="blue", lw=1.2, ls="--", label=f"healthy col{healthy_ref_col}")
    for k in range(1, 6):
        ax.axvline(bpfo_freq * k, color="black", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlim(0, 500)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Envelope spectrum (peak-normalized)")
    ax.set_title(f"Envelope spectrum — bpfo-3@100% all cols (BPFO ≈ {bpfo_freq:.0f} Hz)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 2: linear FFT around line fundamental + BPFO sidebands
    ax = axes[0, 1]
    lo, hi = 20, 180
    for col_idx, (f, m) in lin_curves.items():
        m_plot = m / (m.max() + 1e-12)
        mask = (f >= lo) & (f <= hi)
        style = dict(color="red", lw=1.6, zorder=10) \
            if col_idx == TARGET_COL else dict(color="grey", lw=0.9, alpha=0.7)
        ax.plot(f[mask], m_plot[mask], **style)
    if lin_healthy is not None:
        f, m = lin_healthy
        mask = (f >= lo) & (f <= hi)
        ax.plot(f[mask], (m / (m.max() + 1e-12))[mask],
                 color="blue", lw=1.2, ls="--")
    ax.axvline(LINE_FUND_HZ,                 color="black", ls=":", lw=0.6)
    ax.axvline(LINE_FUND_HZ - bpfo_freq,     color="darkgreen", ls=":", lw=0.6)
    ax.axvline(LINE_FUND_HZ + bpfo_freq,     color="darkgreen", ls=":", lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Linear FFT (peak-normalized)")
    ax.set_title(f"Current-side FFT around line fund (50 Hz ± BPFO sidebands)")
    ax.grid(alpha=0.3)

    # ── Panel 3: boxplot of BPFO harmonic peaks across columns (bpfo-3 only)
    ax = axes[1, 0]
    h1 = bpfo_df.groupby("col_index")["env_bpfo_h1"].mean()
    h2 = bpfo_df.groupby("col_index")["env_bpfo_h2"].mean()
    h3 = bpfo_df.groupby("col_index")["env_bpfo_h3"].mean()
    x = np.arange(3)
    width = 0.11
    for i, col_idx in enumerate(sorted(bpfo_df["col_index"].unique())):
        vals = [h1[col_idx], h2[col_idx], h3[col_idx]]
        color = "red" if col_idx == TARGET_COL else f"C{i}"
        lw = 1.4 if col_idx == TARGET_COL else 0.8
        ax.bar(x + i * width, vals, width=width, color=color,
                edgecolor="black", lw=lw, label=f"col{col_idx}" +
                (" (test)" if col_idx == TARGET_COL else ""))
    ax.set_xticks(x + width * (len(bpfo_df["col_index"].unique()) - 1) / 2)
    ax.set_xticklabels([f"h1 ({bpfo_freq:.0f} Hz)",
                         f"h2 ({bpfo_freq*2:.0f} Hz)",
                         f"h3 ({bpfo_freq*3:.0f} Hz)"])
    ax.set_ylabel("Envelope-spectrum peak (a.u.)")
    ax.set_title("BPFO harmonic amplitudes (mean across 3 phases)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 4: zoomed envelope near BPFO harmonics
    ax = axes[1, 1]
    lo, hi = bpfo_freq - 15, bpfo_freq * 3 + 15
    for col_idx, (f, m) in env_curves.items():
        m_plot = m / (m.max() + 1e-12)
        mask = (f >= lo) & (f <= hi)
        style = dict(color="red", lw=1.6, zorder=10) \
            if col_idx == TARGET_COL else dict(color="grey", lw=0.9, alpha=0.7)
        ax.plot(f[mask], m_plot[mask], **style)
    if env_healthy is not None:
        f, m = env_healthy
        mask = (f >= lo) & (f <= hi)
        ax.plot(f[mask], (m / (m.max() + 1e-12))[mask],
                 color="blue", lw=1.2, ls="--")
    for k in range(1, 4):
        ax.axvline(bpfo_freq * k, color="black", ls=":", lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Envelope spectrum (peak-normalized)")
    ax.set_title("Zoom: BPFO h1–h3 region")
    ax.grid(alpha=0.3)

    fig.suptitle(f"BPFO-3 col{TARGET_COL} @ {TARGET_SPEED}% vs other bpfo-3@{TARGET_SPEED}% "
                  f"columns (ch1 for plotting)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = DIAG_DIR / "bpfo3_col5_vs_others.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — cross-representation similarity
# ═══════════════════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1); b = b.reshape(-1)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _mean_feature(features_tensor: torch.Tensor,
                   indices: np.ndarray) -> np.ndarray:
    """Return the mean feature vector across the given sample indices."""
    if len(indices) == 0:
        return None
    sub = features_tensor[indices].float().cpu().numpy()
    return sub.mean(axis=0)


def part3_cross_representation() -> None:
    print()
    print("=" * 70)
    print("PART 3 — cross-representation similarity for col5@100%")
    print("=" * 70)

    # Representations with (N, ...) `features` tensor in their test.pt file:
    reps = {
        "stft":   FEATURES_V2 / "features"         / "test.pt",
        "mel":    FEATURES_V2 / "mel" / "features" / "test.pt",
        "clarke": FEATURES_V2 / "clarke_features"  / "test.pt",
        "env":    FEATURES_V2 / "envelope" / "features" / "test.pt",
    }
    # DWT stores features as a list of per-level tensors under a different
    # key. Handle separately.
    dwt_test = FEATURES_V2 / "dwt" / "features" / "test.pt"

    out_rows = []

    for name, path in reps.items():
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        print(f"  loading {name}: {path}")
        data = torch.load(path, weights_only=False, mmap=True)
        features = data["features"]
        labels = data["labels"].numpy()
        md = data["metadata"]
        class_names = data["class_names"]
        bpfo_idx = class_names.index(TARGET_CLASS)
        hlth_idx = class_names.index("healthy 1")

        col5_mask = np.array([
            (int(m.get("col_index", -1)) == TARGET_COL and
             int(m.get("speed_pct", -1)) == TARGET_SPEED and
             labels[i] == bpfo_idx)
            for i, m in enumerate(md)
        ])
        h100_mask = np.array([
            (int(m.get("speed_pct", -1)) == TARGET_SPEED and
             labels[i] == hlth_idx)
            for i, m in enumerate(md)
        ])
        # Other bpfo-3@100% in the test file — if none (expected, only col5 is test),
        # fall back to bpfo-3@75 and @50 test samples.
        other_bpfo_mask = np.array([
            (int(m.get("col_index", -1)) != TARGET_COL and
             int(m.get("speed_pct", -1)) == TARGET_SPEED and
             labels[i] == bpfo_idx)
            for i, m in enumerate(md)
        ])
        bpfo_any_mask = np.array([
            labels[i] == bpfo_idx and int(m.get("col_index", -1)) != TARGET_COL
            for i, m in enumerate(md)
        ])

        col5_idx = np.where(col5_mask)[0]
        h100_idx = np.where(h100_mask)[0]
        other_idx = np.where(other_bpfo_mask)[0]
        any_idx = np.where(bpfo_any_mask)[0]

        print(f"    col5@100% segments in test.pt: {len(col5_idx)}")
        print(f"    healthy@100% segments:         {len(h100_idx)}")
        print(f"    other bpfo-3@100% segments:    {len(other_idx)}  "
              f"(any speed non-col5: {len(any_idx)})")

        if len(col5_idx) == 0 or len(h100_idx) == 0:
            print(f"    [skip] insufficient coverage for {name}")
            continue

        mean_col5 = _mean_feature(features, col5_idx)
        mean_h100 = _mean_feature(features, h100_idx)
        mean_other = _mean_feature(features,
                                     other_idx if len(other_idx) else any_idx)

        sim_to_healthy = cosine_sim(mean_col5, mean_h100) if mean_h100 is not None else np.nan
        sim_to_bpfo    = cosine_sim(mean_col5, mean_other) if mean_other is not None else np.nan

        out_rows.append({
            "representation":                   name,
            "cos(col5@100, healthy@100)":       sim_to_healthy,
            "cos(col5@100, bpfo-3 non-col5)":   sim_to_bpfo,
            "closer_to":                        ("healthy" if sim_to_healthy > sim_to_bpfo
                                                  else "bpfo" if sim_to_bpfo > sim_to_healthy
                                                  else "tie"),
            "n_col5":                           int(len(col5_idx)),
            "n_healthy":                        int(len(h100_idx)),
            "n_other_bpfo":                     int(max(len(other_idx), len(any_idx))),
            "features_path":                    str(path),
        })

    # DWT (multi-level list)
    if dwt_test.exists():
        print(f"  loading dwt: {dwt_test}")
        data = torch.load(dwt_test, weights_only=False, mmap=True)
        feats_per_level = data["features_per_level"]
        labels = data["labels"].numpy()
        md = data["metadata"]
        class_names = data["class_names"]
        bpfo_idx = class_names.index(TARGET_CLASS)
        hlth_idx = class_names.index("healthy 1")

        col5_idx = np.where(np.array([
            int(m.get("col_index", -1)) == TARGET_COL and
            int(m.get("speed_pct", -1)) == TARGET_SPEED and
            labels[i] == bpfo_idx
            for i, m in enumerate(md)
        ]))[0]
        h100_idx = np.where(np.array([
            int(m.get("speed_pct", -1)) == TARGET_SPEED and labels[i] == hlth_idx
            for i, m in enumerate(md)
        ]))[0]
        any_idx = np.where(np.array([
            labels[i] == bpfo_idx and int(m.get("col_index", -1)) != TARGET_COL
            for i, m in enumerate(md)
        ]))[0]

        if len(col5_idx) and len(h100_idx):
            # Concat per-level mean vectors into one flat vector per group
            def flat(idx):
                return np.concatenate([
                    feats_per_level[j][idx].float().cpu().numpy()
                        .mean(axis=0).reshape(-1)
                    for j in range(len(feats_per_level))
                ])
            mean_col5 = flat(col5_idx)
            mean_h100 = flat(h100_idx)
            mean_any  = flat(any_idx) if len(any_idx) else None
            sim_to_healthy = cosine_sim(mean_col5, mean_h100)
            sim_to_bpfo    = cosine_sim(mean_col5, mean_any) if mean_any is not None else np.nan
            out_rows.append({
                "representation":                   "dwt",
                "cos(col5@100, healthy@100)":       sim_to_healthy,
                "cos(col5@100, bpfo-3 non-col5)":   sim_to_bpfo,
                "closer_to":                        ("healthy" if sim_to_healthy > sim_to_bpfo
                                                      else "bpfo" if sim_to_bpfo > sim_to_healthy
                                                      else "tie"),
                "n_col5":                           int(len(col5_idx)),
                "n_healthy":                        int(len(h100_idx)),
                "n_other_bpfo":                     int(len(any_idx)),
                "features_path":                    str(dwt_test),
            })
        else:
            print("    [skip] dwt: insufficient coverage")

    if not out_rows:
        print("  no representations produced similarity rows.")
        return

    df = pd.DataFrame(out_rows)
    out = DIAG_DIR / "bpfo3_cross_representation_similarity.csv"
    df.to_csv(out, index=False)
    print(f"  → {out.name}")
    print()
    for row in out_rows:
        print(f"  {row['representation']:<8} "
              f"cos(col5, healthy@100)={row['cos(col5@100, healthy@100)']:.3f}  "
              f"cos(col5, bpfo-non-col5)={row['cos(col5@100, bpfo-3 non-col5)']:.3f}  "
              f"→ closer to {row['closer_to']}")


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("BPFO-3 MISCLASSIFICATION DIAGNOSTIC")
    print(f"started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"diagnostics dir: {DIAG_DIR}")
    print("=" * 70)

    runs = list_solo_runs()
    print(f"\nDiscovered {len(runs)} solo runs:")
    for r in runs:
        print(f"  {r.name}")

    part1_aggregate(runs)
    part2_column_signatures()
    part3_cross_representation()

    print()
    print("=" * 70)
    print("DONE")
    print(f"All artefacts under {DIAG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
