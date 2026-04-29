"""
STEP 1 — DATASET VALIDATION
===========================
Reads manifest.csv produced by setup_pc.py, runs per-file quality checks,
and writes a validated_manifest.csv with verdict columns attached. Also
writes a human-readable validation_report.txt.

Checks per file
---------------
  1. Readable             — pandas can open the CSV
  2. Row count            — exactly 300,000 (15 s × 20 kHz)
  3. No all-NaN columns
  4. No flat-zero columns (motor not running)
  5. No clipped columns   (>98% of samples at max abs value → sensor saturation)
  6. RMS within plausible range (>0.001 A, <50 A)
  7. Implicit: reports per-file column count

Verdicts
--------
  PASS  — all checks passed
  WARN  — some columns flat/clipped/off-range but file usable
  FAIL  — unreadable, wrong row count, or all-column failure

Why re-validate here
--------------------
setup_pc.py carries verdicts over from an old validated_manifest.csv if
one is present. On first-time setup it wasn't, so this script regenerates
them from scratch. File bytes are identical to the old machine so verdicts
will match what the old machine produced — we're just recomputing because
the data moved.

Usage
-----
    python validate_dataset.py

Configure the two paths below and run.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
MANIFEST_PATH = r"C:\Project Work\outputs\manifest.csv"
OUTPUT_DIR    = r"C:\Project Work\outputs"
# ──────────────────────────────────────────────────────────────────────────────

EXPECTED_ROWS    = 300_000      # 15 s × 20 kHz
CLIP_FRACTION    = 0.98         # clip-flag: >98% of samples at max abs value
ZERO_RMS_THRESH  = 0.001        # A — below this is "motor not running"
MAX_RMS_THRESH   = 50.0         # A — above this is likely sensor/scaling error


def validate_file(path: str) -> dict:
    """Run all checks on a single file; return per-file stats + verdict."""
    result = {
        "readable":      False,
        "row_count_ok":  False,
        "has_nan":       False,
        "zero_cols":     0,
        "clipped_cols":  0,
        "rms_min":       np.nan,
        "rms_max":       np.nan,
        "rms_flag":      False,
        "verdict":       "FAIL",
        "notes":         "",
    }

    # 1. Readability
    try:
        df = pd.read_csv(path)
    except Exception as e:
        result["notes"] = f"Cannot read: {e}"
        return result
    result["readable"] = True

    # Time column is first; signals are the rest
    signal_df = df.iloc[:, 1:]
    n_cols = signal_df.shape[1]

    if n_cols == 0:
        result["notes"] = "No signal columns"
        return result

    # 2. Row count
    row_ok = (len(df) == EXPECTED_ROWS)
    result["row_count_ok"] = row_ok
    if not row_ok:
        result["notes"] += f"rows={len(df)} (expected {EXPECTED_ROWS}); "

    # 3–6. Per-column checks
    arr = signal_df.to_numpy(dtype=float, na_value=np.nan)

    # NaN check (any column entirely NaN)
    nan_per_col = np.isnan(arr).all(axis=0)
    if nan_per_col.any():
        result["has_nan"] = True
        result["notes"] += f"{int(nan_per_col.sum())} all-NaN col(s); "

    # Replace NaN with 0 for stats so we don't blow up
    arr_clean = np.nan_to_num(arr, nan=0.0)

    # RMS per column
    rms = np.sqrt(np.mean(arr_clean ** 2, axis=0))
    if rms.size > 0:
        result["rms_min"] = float(np.nanmin(rms))
        result["rms_max"] = float(np.nanmax(rms))

    # Flat-zero (motor not running on that column)
    zero_mask = rms < ZERO_RMS_THRESH
    result["zero_cols"] = int(zero_mask.sum())
    if result["zero_cols"]:
        result["notes"] += f"{result['zero_cols']} flat-zero col(s); "

    # Clipped (most samples at abs-max)
    col_abs_max = np.max(np.abs(arr_clean), axis=0)
    # Avoid div-by-zero for flat columns
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_at_max = np.where(
            col_abs_max > 0,
            np.mean(np.abs(arr_clean) >= 0.999 * col_abs_max[np.newaxis, :], axis=0),
            0.0,
        )
    clip_mask = frac_at_max > CLIP_FRACTION
    result["clipped_cols"] = int(clip_mask.sum())
    if result["clipped_cols"]:
        result["notes"] += f"{result['clipped_cols']} clipped col(s); "

    # RMS range sanity
    rms_out_of_range = ((rms < ZERO_RMS_THRESH) | (rms > MAX_RMS_THRESH)).any()
    result["rms_flag"] = bool(rms_out_of_range)
    if rms_out_of_range:
        result["notes"] += f"RMS out-of-range [{result['rms_min']:.4f}, {result['rms_max']:.2f}]; "

    # Verdict
    all_cols_bad = (result["zero_cols"] + result["clipped_cols"]) >= n_cols
    if not row_ok or all_cols_bad or result["has_nan"] and nan_per_col.all():
        result["verdict"] = "FAIL"
    elif result["zero_cols"] or result["clipped_cols"] or result["rms_flag"] or result["has_nan"]:
        result["verdict"] = "WARN"
    else:
        result["verdict"] = "PASS"

    return result


def main():
    print("=" * 70)
    print("DATASET VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    manifest_path = Path(MANIFEST_PATH)
    out_dir       = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        print("Run setup_pc.py first.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    n = len(manifest)
    print(f"\nValidating {n} files...")
    print(f"Expected rows per file : {EXPECTED_ROWS:,}")
    print(f"Clip threshold         : >{int(CLIP_FRACTION * 100)}% at abs-max")
    print(f"RMS acceptable range   : [{ZERO_RMS_THRESH}, {MAX_RMS_THRESH}] A\n")

    # Drop any pre-existing verdict columns so we can re-create cleanly
    verdict_cols = ["verdict", "readable", "row_count_ok", "has_nan",
                    "zero_cols", "clipped_cols", "rms_min", "rms_max",
                    "rms_flag", "notes"]
    manifest = manifest.drop(columns=[c for c in verdict_cols if c in manifest.columns])

    # Run validation
    results = []
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        res = validate_file(row.full_path)
        results.append(res)
        if i % 20 == 0 or i == n:
            print(f"  [{i:>3}/{n}] {Path(row.full_path).name:<60} → {res['verdict']}")

    results_df = pd.DataFrame(results)
    validated  = pd.concat([manifest.reset_index(drop=True), results_df], axis=1)

    # Save
    out_csv = out_dir / "validated_manifest.csv"
    validated.to_csv(out_csv, index=False)

    # Summary report
    report = []
    report.append(f"Validation Report — {datetime.now().isoformat(timespec='seconds')}")
    report.append("=" * 60)
    report.append(f"Files validated : {n}")
    report.append("")
    report.append("Verdict breakdown:")
    report.append(validated["verdict"].value_counts().to_string())
    report.append("")
    report.append("Per-class verdict counts:")
    by_class = validated.groupby(["fault_label", "verdict"]).size().unstack(fill_value=0)
    report.append(by_class.to_string())
    report.append("")
    report.append("Per-speed verdict counts:")
    by_speed = validated.groupby(["speed_pct", "verdict"]).size().unstack(fill_value=0)
    report.append(by_speed.to_string())

    fails = validated[validated["verdict"] == "FAIL"]
    if not fails.empty:
        report.append("")
        report.append(f"FAIL files ({len(fails)}):")
        for _, r in fails.iterrows():
            report.append(f"  {r['fault_label']} / {r['speed_pct']}% / {r['channel']} :: {r['notes']}")

    warns = validated[validated["verdict"] == "WARN"]
    if not warns.empty:
        report.append("")
        report.append(f"WARN files ({len(warns)}):")
        for _, r in warns.iterrows():
            report.append(f"  {r['fault_label']} / {r['speed_pct']}% / {r['channel']} :: {r['notes']}")

    report_text = "\n".join(report)
    (out_dir / "validation_report.txt").write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(validated["verdict"].value_counts().to_string())
    print(f"\n✓ {out_csv}")
    print(f"✓ {out_dir / 'validation_report.txt'}")
    print("\nUpload validation_report.txt back to the chat.")


if __name__ == "__main__":
    main()
