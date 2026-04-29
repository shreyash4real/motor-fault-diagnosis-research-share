"""
STEP 2 — DENOISING (manifest-driven)
====================================
Reads validated_manifest.csv and processes every PASS/WARN file through
a 4th-order Butterworth bandpass (5 Hz – 5000 Hz, zero-phase filtfilt).
FAIL files are skipped automatically.

Preserves full multi-column CSV structure, mirrors input folder tree in
the output, logs column counts per file for verification.

Wavelet denoising is intentionally excluded — universal threshold
attenuates real MCSA fault sidebands.

Usage
-----
    python denoise_all.py

Requires: pandas, numpy, scipy
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
VALIDATED_MANIFEST = r"C:\Project Work\outputs\validated_manifest.csv"
INPUT_ROOT         = r"C:\Project Work\Dataset\Electric\Motor-2"
OUTPUT_ROOT        = r"C:\Project Work\Denoised\Motor-2"
DENOISE_LOG        = r"C:\Project Work\outputs\denoise_log.csv"
# ──────────────────────────────────────────────────────────────────────────────

FS          = 20_000           # Hz
LOWCUT_HZ   = 5.0              # removes DC drift, sub-Hz VFD artefacts
HIGHCUT_HZ  = 5000.0           # removes VFD PWM switching noise (2-16 kHz)
FILT_ORDER  = 4                # filtfilt doubles effective order to 8, zero-phase


def build_filter(fs: int, lowcut: float, highcut: float, order: int):
    """Design the Butterworth bandpass once; reuse across all files."""
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band")


def denoise_signal(raw: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Bandpass a single column. Skip flat/NaN columns (preserve unchanged).
    """
    if np.all(np.isnan(raw)) or np.nanmax(np.abs(raw)) < 1e-9:
        return raw  # column is flat or all-NaN; pass through unchanged
    # filtfilt doesn't tolerate NaNs; zero-fill first
    raw_clean = np.nan_to_num(raw, nan=0.0)
    return filtfilt(b, a, raw_clean).astype(np.float32)


def mirror_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    """Map an input file path to the equivalent path under output_root."""
    rel = input_path.relative_to(input_root)
    return output_root / rel


def process_file(input_path: Path, output_path: Path,
                 b: np.ndarray, a: np.ndarray) -> dict:
    log = {
        "input_path":  str(input_path),
        "output_path": str(output_path),
        "input_cols":  None,
        "output_cols": None,
        "status":      "OK",
        "note":        "",
    }

    try:
        df = pd.read_csv(input_path)
        time_col  = df.iloc[:, 0]
        signal_df = df.iloc[:, 1:]
        n_signal  = signal_df.shape[1]
        log["input_cols"] = n_signal

        if n_signal == 0:
            log["status"] = "SKIP"
            log["note"]   = "No signal columns"
            return log

        # Apply filter to every signal column
        denoised = {}
        for col in signal_df.columns:
            raw = signal_df[col].to_numpy(dtype=float)
            denoised[col] = denoise_signal(raw, b, a)

        out_df = pd.DataFrame(denoised)
        out_df.insert(0, df.columns[0], time_col.values)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)

        log["output_cols"] = n_signal
        if log["input_cols"] != log["output_cols"]:
            log["status"] = "MISMATCH"
            log["note"]  += "Column count mismatch"

    except Exception as e:
        log["status"] = "ERROR"
        log["note"]   = str(e)

    return log


def main():
    print("=" * 70)
    print("DENOISING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    val_path   = Path(VALIDATED_MANIFEST)
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)

    if not val_path.exists():
        print(f"ERROR: validated_manifest.csv not found at {val_path}")
        print("Run validate_dataset.py first.")
        sys.exit(1)

    manifest = pd.read_csv(val_path)
    print(f"\nFilter: Butterworth order {FILT_ORDER}, {LOWCUT_HZ}-{HIGHCUT_HZ} Hz, "
          f"zero-phase (filtfilt)")
    print(f"Input  root : {input_root}")
    print(f"Output root : {output_root}")

    # Only process files that passed validation (PASS or WARN, not FAIL)
    to_process = manifest[manifest["verdict"].isin(["PASS", "WARN"])].copy()
    skipped    = manifest[manifest["verdict"] == "FAIL"]

    print(f"\nFiles in manifest       : {len(manifest)}")
    print(f"To process (PASS + WARN): {len(to_process)}")
    print(f"Skipped (FAIL)          : {len(skipped)}")
    print()

    b, a = build_filter(FS, LOWCUT_HZ, HIGHCUT_HZ, FILT_ORDER)

    output_root.mkdir(parents=True, exist_ok=True)

    logs = []
    t_start = datetime.now()
    n = len(to_process)
    for i, row in enumerate(to_process.itertuples(index=False), start=1):
        in_path  = Path(row.full_path)
        out_path = mirror_path(in_path, input_root, output_root)
        log = process_file(in_path, out_path, b, a)
        logs.append(log)

        # Progress line
        elapsed = (datetime.now() - t_start).total_seconds()
        rate = i / elapsed if elapsed > 0 else 0
        eta  = (n - i) / rate if rate > 0 else 0
        print(f"  [{i:>3}/{n}] {log['status']:<9} "
              f"{in_path.name:<55} "
              f"cols={log['input_cols']}  "
              f"eta={eta:5.0f}s")

    # Save log
    log_df = pd.DataFrame(logs)
    Path(DENOISE_LOG).parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(DENOISE_LOG, index=False)

    elapsed_total = (datetime.now() - t_start).total_seconds()

    print()
    print("=" * 70)
    print("DENOISING COMPLETE")
    print("=" * 70)
    print(f"Elapsed : {elapsed_total/60:.1f} min")
    print()
    print("Status breakdown:")
    print(log_df["status"].value_counts().to_string())
    print()

    errs = log_df[log_df["status"] == "ERROR"]
    if not errs.empty:
        print(f"ERROR files ({len(errs)}):")
        for _, r in errs.iterrows():
            print(f"  {Path(r['input_path']).name} :: {r['note']}")

    mismatches = log_df[log_df["status"] == "MISMATCH"]
    if not mismatches.empty:
        print(f"\nMISMATCH files ({len(mismatches)}):")
        for _, r in mismatches.iterrows():
            print(f"  {Path(r['input_path']).name} :: "
                  f"in={r['input_cols']} out={r['output_cols']}")

    # Sanity: column-count totals
    total_in  = int(log_df["input_cols"].fillna(0).sum())
    total_out = int(log_df["output_cols"].fillna(0).sum())
    print(f"\nColumn totals: input={total_in}  output={total_out}  "
          f"{'✓ match' if total_in == total_out else '✗ MISMATCH'}")

    print(f"\n✓ Denoise log: {DENOISE_LOG}")
    print(f"✓ Output tree: {output_root}")
    print("\nUpload denoise_log.csv back to the chat.")


if __name__ == "__main__":
    main()
