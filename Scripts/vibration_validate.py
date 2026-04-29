"""
STEP 2 — VIBRATION DATASET VALIDATION
=======================================
Reads vibration_manifest.csv produced by vibration_scan.py.
Runs per-file quality checks on every vibration file and produces:

  - vibration_validated_manifest.csv  : manifest + validation columns
  - vibration_validation_report.txt   : human-readable summary

Per-file checks (adapted from validate_dataset.py for g-units and 12s windows):

  1. File is readable
  2. Row count is exactly 240,000 per column (12s x 20kHz)
  3. No all-NaN columns
  4. No flat-near-zero columns (motor not running)
  5. No clipped columns (values at or near ±50 g sustained)
  6. RMS is within a plausible acceleration band
  7. All 5 channels present for each (fault, speed)

Verdict per file:
  PASS  — all checks clean
  WARN  — one or more soft flags (e.g., slightly low RMS in one col, or
          signal_columns not a multiple of 5). Still usable.
  FAIL  — unreadable, wrong row count, NaN column, or all columns flat zero.
          Skipped by later pipeline stages.

Usage:
    python vibration_validate.py

Set MANIFEST_PATH and OUTPUT_DIR before running.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
MANIFEST_PATH = r"C:\Project Work\outputs\vibration\vibration_manifest.csv"
OUTPUT_DIR    = r"C:\Project Work\outputs\vibration"
# ──────────────────────────────────────────────────────────────────────────────

# ─── PHYSICS THRESHOLDS ───────────────────────────────────────────────────────
EXPECTED_ROWS   = 240_000     # 12s x 20kHz — from dataset README
CLIP_LEVEL_G    = 49.0        # Dataset paper: max allowable ±50 g from 24-bit WAV.
                              # Flag columns where |x| >= this level for >1% of samples.
CLIP_FRAC       = 0.01        # 1% of samples at/near clip level = flag
ZERO_RMS_THRESH = 0.001       # RMS < 1 mg => sensor effectively silent
LOW_RMS_WARN    = 0.005       # RMS < 5 mg => suspiciously quiet, warn but don't fail
HIGH_RMS_WARN   = 10.0        # RMS > 10 g sustained => implausible for bearing acceleration
# ──────────────────────────────────────────────────────────────────────────────

EXPECTED_CHANNELS = {"ch1", "ch2", "ch3", "ch4", "ch5"}


def validate_file(path: str) -> dict:
    """Load a vibration CSV and run all quality checks. Returns a dict of results."""
    result = {
        "readable":      False,
        "row_count_ok":  False,
        "has_nan":       False,
        "zero_cols":     0,     # columns with RMS < ZERO_RMS_THRESH
        "low_rms_cols":  0,     # columns with ZERO_RMS_THRESH <= RMS < LOW_RMS_WARN
        "high_rms_cols": 0,     # columns with RMS > HIGH_RMS_WARN
        "clipped_cols":  0,     # columns with clipping fraction > CLIP_FRAC
        "rms_min":       None,
        "rms_max":       None,
        "n_signal_cols": 0,
        "verdict":       "FAIL",
        "notes":         "",
    }

    # ── 1. Readability ──────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path)
    except Exception as e:
        result["notes"] = f"Cannot read: {e}"
        return result
    result["readable"] = True

    signal_df = df.iloc[:, 1:]                 # drop time column
    n_cols = len(signal_df.columns)
    result["n_signal_cols"] = n_cols

    if n_cols == 0:
        result["notes"] = "No signal columns"
        return result

    # ── 2. Row count ────────────────────────────────────────────────────────
    row_ok = (len(df) == EXPECTED_ROWS)
    result["row_count_ok"] = row_ok
    if not row_ok:
        result["notes"] += f"Rows={len(df)} (expected {EXPECTED_ROWS}); "

    # ── 3. NaN check ────────────────────────────────────────────────────────
    nan_mask = signal_df.isna().all(axis=0)
    if nan_mask.any():
        result["has_nan"] = True
        result["notes"] += f"{int(nan_mask.sum())} all-NaN cols; "

    # ── 4-6. Per-column RMS, flat, clip checks ──────────────────────────────
    arr = signal_df.to_numpy(dtype=np.float64)
    # Replace NaN with 0 for RMS calc purposes; NaN already flagged above.
    arr = np.nan_to_num(arr, nan=0.0)

    col_rms = np.sqrt(np.mean(arr ** 2, axis=0))
    result["rms_min"] = float(np.min(col_rms))
    result["rms_max"] = float(np.max(col_rms))

    result["zero_cols"]     = int(np.sum(col_rms < ZERO_RMS_THRESH))
    result["low_rms_cols"]  = int(np.sum((col_rms >= ZERO_RMS_THRESH) &
                                          (col_rms < LOW_RMS_WARN)))
    result["high_rms_cols"] = int(np.sum(col_rms > HIGH_RMS_WARN))

    clip_frac_per_col = np.mean(np.abs(arr) >= CLIP_LEVEL_G, axis=0)
    result["clipped_cols"] = int(np.sum(clip_frac_per_col > CLIP_FRAC))

    # ── 7. Verdict ──────────────────────────────────────────────────────────
    hard_fail = (
        not result["readable"]
        or not result["row_count_ok"]
        or result["has_nan"]
        or result["zero_cols"] == n_cols          # EVERY col flat = motor off
    )
    soft_warn = (
        result["zero_cols"] > 0
        or result["low_rms_cols"] > 0
        or result["high_rms_cols"] > 0
        or result["clipped_cols"] > 0
        or (n_cols % 5 != 0)                      # partial 60s recording
    )

    if hard_fail:
        result["verdict"] = "FAIL"
    elif soft_warn:
        result["verdict"] = "WARN"
    else:
        result["verdict"] = "PASS"

    if result["zero_cols"]:
        result["notes"] += f"{result['zero_cols']} flat cols; "
    if result["low_rms_cols"]:
        result["notes"] += f"{result['low_rms_cols']} low-RMS cols; "
    if result["high_rms_cols"]:
        result["notes"] += f"{result['high_rms_cols']} high-RMS cols; "
    if result["clipped_cols"]:
        result["notes"] += f"{result['clipped_cols']} clipped cols; "
    if n_cols % 5 != 0:
        result["notes"] += f"n_cols={n_cols} not divisible by 5 (partial); "

    return result


def main():
    print("=" * 70)
    print("VIBRATION DATASET VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    manifest_path = Path(MANIFEST_PATH)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(manifest)} files\n")

    print(f"Thresholds:")
    print(f"  Expected rows per col: {EXPECTED_ROWS:,} (12s x 20 kHz)")
    print(f"  Clip level           : |x| >= {CLIP_LEVEL_G} g for > {CLIP_FRAC*100:.0f}% of samples")
    print(f"  Flat (zero) RMS      : < {ZERO_RMS_THRESH} g")
    print(f"  Low RMS warn         : < {LOW_RMS_WARN} g")
    print(f"  High RMS warn        : > {HIGH_RMS_WARN} g")
    print()

    # ── Run per-file validation ────────────────────────────────────────────
    results = []
    t0 = datetime.now()
    for i, row in manifest.iterrows():
        if i % 25 == 0:
            print(f"  [{i+1:>3}/{len(manifest)}] {row['fault_label']} "
                  f"({row['speed_pct']}%) {row['channel']} ...")
        r = validate_file(row["full_path"])
        r["full_path"] = row["full_path"]
        results.append(r)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nValidated {len(results)} files in {elapsed:.0f}s.\n")

    # ── Merge into validated manifest ──────────────────────────────────────
    results_df = pd.DataFrame(results)
    merged = manifest.merge(results_df, on="full_path", how="left")

    validated_path = out_dir / "vibration_validated_manifest.csv"
    merged.to_csv(validated_path, index=False)

    # ── Cross-channel completeness re-check ────────────────────────────────
    # A (fault, speed) group is usable only if all 5 channels PASS or WARN.
    usable_mask = merged["verdict"].isin(["PASS", "WARN"])
    usable = merged[usable_mask]
    pivot = (usable.groupby(["fault_label", "speed_pct", "channel"])
                   .size().unstack(fill_value=0))
    for ch in sorted(EXPECTED_CHANNELS):
        if ch not in pivot.columns:
            pivot[ch] = 0
    pivot = pivot[sorted(EXPECTED_CHANNELS)]
    pivot["complete"] = (pivot.sum(axis=1) == len(EXPECTED_CHANNELS)).map(
        {True: "YES", False: "NO "}
    )
    incomplete_groups = pivot[pivot["complete"] == "NO "]

    # ── Print summary ───────────────────────────────────────────────────────
    print("=" * 70)
    print("VERDICT BREAKDOWN")
    print("=" * 70)
    vc = merged["verdict"].value_counts()
    for v in ["PASS", "WARN", "FAIL"]:
        n = int(vc.get(v, 0))
        print(f"  {v:<6} {n:>4} files")
    print()

    if (merged["verdict"] == "FAIL").any():
        print("FAIL DETAILS:")
        fails = merged[merged["verdict"] == "FAIL"]
        print(fails[["fault_label", "speed_pct", "channel", "notes"]].to_string(index=False))
        print()

    if (merged["verdict"] == "WARN").any():
        print(f"WARN DETAILS (first 20 of {int(vc.get('WARN', 0))}):")
        warns = merged[merged["verdict"] == "WARN"].head(20)
        print(warns[["fault_label", "speed_pct", "channel",
                     "zero_cols", "low_rms_cols", "high_rms_cols",
                     "clipped_cols", "notes"]].to_string(index=False))
        print()

    print("=" * 70)
    print("POST-VALIDATION 5-CHANNEL COMPLETENESS (PASS + WARN only)")
    print("=" * 70)
    if len(incomplete_groups) == 0:
        print("All (fault, speed) groups retain all 5 channels. Good.")
    else:
        print("The following (fault, speed) groups lost a channel to FAIL:")
        print(incomplete_groups.to_string())
    print()

    # ── 4-class RMS summary ────────────────────────────────────────────────
    target_classes = [
        "healthy 1", "stator short 1", "bearing bpfo 3", "broken rotor bar",
    ]
    print("=" * 70)
    print("4-CLASS RMS SUMMARY (median rms_max across files, per channel)")
    print("=" * 70)
    for cls in target_classes:
        sub = merged[(merged["fault_label"] == cls) &
                     (merged["verdict"].isin(["PASS", "WARN"]))]
        if len(sub) == 0:
            print(f"  {cls:<22}  NO USABLE FILES")
            continue
        by_ch = sub.groupby("channel")["rms_max"].median()
        ch_summary = "  ".join(
            f"{ch}={by_ch.get(ch, float('nan')):.3f}g"
            for ch in sorted(EXPECTED_CHANNELS)
        )
        print(f"  {cls:<22}  {ch_summary}")
    print()

    # ── Report file ─────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append(f"Vibration Validation Report - "
                        f"{datetime.now().isoformat(timespec='seconds')}")
    report_lines.append("=" * 70)
    report_lines.append(f"Total files          : {len(merged)}")
    for v in ["PASS", "WARN", "FAIL"]:
        report_lines.append(f"  {v:<6} {int(vc.get(v, 0)):>4}")
    report_lines.append("")
    report_lines.append("Thresholds:")
    report_lines.append(f"  EXPECTED_ROWS   = {EXPECTED_ROWS}")
    report_lines.append(f"  CLIP_LEVEL_G    = {CLIP_LEVEL_G}  (frac > {CLIP_FRAC})")
    report_lines.append(f"  ZERO_RMS_THRESH = {ZERO_RMS_THRESH}")
    report_lines.append(f"  LOW_RMS_WARN    = {LOW_RMS_WARN}")
    report_lines.append(f"  HIGH_RMS_WARN   = {HIGH_RMS_WARN}")
    report_lines.append("")
    report_lines.append("Verdict breakdown by class:")
    class_verdict = (merged.groupby(["fault_label", "verdict"])
                           .size().unstack(fill_value=0))
    for col in ["PASS", "WARN", "FAIL"]:
        if col not in class_verdict.columns:
            class_verdict[col] = 0
    class_verdict = class_verdict[["PASS", "WARN", "FAIL"]]
    report_lines.append(class_verdict.to_string())
    report_lines.append("")
    report_lines.append("5-channel completeness after validation:")
    report_lines.append(pivot.to_string())

    report_path = out_dir / "vibration_validation_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)
    print(f"  {validated_path}")
    print(f"  {report_path}")
    print()
    print("Next step: review the report, then proceed to vibration_denoise_all.py.")


if __name__ == "__main__":
    main()
