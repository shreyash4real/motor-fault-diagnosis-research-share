"""
STEP 3 — VIBRATION DENOISING (manifest-driven)
================================================
Reads vibration_validated_manifest.csv produced by vibration_validate.py.
Processes PASS and WARN files only — skips FAILs automatically.

Applies a 4th-order Butterworth zero-phase BANDPASS filter, 6 Hz – 9500 Hz.
Uses second-order-section (SOS) representation for numerical stability at
the 0.95-Nyquist highcut — see note in bandpass_filter().

Wavelet denoising intentionally excluded — same reasoning as the current
branch (universal Donoho-Johnstone threshold is wrong for VFD noise).

  - Processes ALL signal columns, preserves full multi-column CSV structure
  - Mirrors input folder structure in output
  - Logs column-count verification per file

Usage:
    python vibration_denoise_all.py

Set VALIDATED_MANIFEST, INPUT_ROOT, OUTPUT_ROOT before running.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
VALIDATED_MANIFEST = r"C:\Project Work\Outputs\vibration\vibration_validated_manifest.csv"
INPUT_ROOT         = r"C:\Project Work\Dataset\Vibration\Motor-2"
OUTPUT_ROOT        = r"C:\Project Work\Denoised\Vibration\Motor-2"
DENOISE_LOG        = r"C:\Project Work\Outputs\vibration\vibration_denoise_log.csv"
# ──────────────────────────────────────────────────────────────────────────────

# ─── FILTER PARAMETERS ────────────────────────────────────────────────────────
FS      = 20_000
LOWCUT  = 6.0       # removes DC drift, sub-Hz thermal / table motion
HIGHCUT = 9500.0    # keep everything the hardware bandpass already admitted
ORDER   = 4         # 4th-order; filtfilt doubles effective order (zero-phase)
# ──────────────────────────────────────────────────────────────────────────────

EXPECTED_ROWS = 240_000   # 12s x 20 kHz — sanity check per column


def bandpass_filter(signal: np.ndarray,
                    lowcut: float = LOWCUT,
                    highcut: float = HIGHCUT,
                    fs: int = FS,
                    order: int = ORDER) -> np.ndarray:
    """
    4th-order Butterworth zero-phase bandpass filter, SOS form.

    lowcut  =    6 Hz  removes DC drift and sub-Hz thermal drift.
    highcut = 9500 Hz  keeps content the hardware bandpass already admitted
                       (hardware cutoff 10 kHz). Preserves bearing-fault
                       envelope structure up near Nyquist.
    order   =    4     filtfilt doubles effective order to 8 (zero-phase).

    Uses second-order-section representation (output='sos' + sosfiltfilt)
    because the highcut sits at 0.95 Nyquist — high-order BA coefficients
    can accumulate roundoff there. SOS splits the filter into 2nd-order
    blocks which are numerically well-conditioned at any passband.
    """
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq],
                 btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def denoise_file(input_path: str, output_path: str) -> dict:
    """
    Read one vibration CSV, bandpass-filter every signal column, save to mirror path.
    Returns a status dict for the log.
    """
    status = {
        "input_path":    input_path,
        "output_path":   output_path,
        "n_rows":        None,
        "n_signal_cols": None,
        "row_count_ok":  False,
        "success":       False,
        "notes":         "",
    }
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        status["notes"] = f"Cannot read: {e}"
        return status

    status["n_rows"] = len(df)
    status["row_count_ok"] = (len(df) == EXPECTED_ROWS)

    # Preserve time column as-is, filter every signal column
    time_col   = df.iloc[:, 0].to_numpy()
    signal_arr = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    n_cols = signal_arr.shape[1]
    status["n_signal_cols"] = n_cols

    if n_cols == 0:
        status["notes"] = "No signal columns"
        return status

    # Apply bandpass per column (columns are independent recording sessions)
    filtered = np.empty_like(signal_arr)
    for c in range(n_cols):
        filtered[:, c] = bandpass_filter(signal_arr[:, c])

    # Reassemble with the original headers and time column
    out_df = pd.DataFrame(
        np.column_stack([time_col.reshape(-1, 1), filtered]),
        columns=df.columns,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    status["success"] = True
    if not status["row_count_ok"]:
        status["notes"] += f"Rows={len(df)} (expected {EXPECTED_ROWS}); "
    return status


def mirror_path(input_path: str) -> str:
    """Map input_path under INPUT_ROOT to the same relative path under OUTPUT_ROOT."""
    in_root  = Path(INPUT_ROOT).resolve()
    out_root = Path(OUTPUT_ROOT).resolve()
    in_p     = Path(input_path).resolve()
    try:
        rel = in_p.relative_to(in_root)
    except ValueError:
        # Fall back: keep the filename only under a flat subdir
        rel = Path(in_p.name)
    return str(out_root / rel)


def main():
    print("=" * 70)
    print("VIBRATION DENOISING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Manifest:    {VALIDATED_MANIFEST}")
    print(f"Input root:  {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Filter:      Butterworth order {ORDER} bandpass {LOWCUT} Hz - {HIGHCUT} Hz (SOS, zero-phase)")
    print(f"fs:          {FS} Hz  (Nyquist {FS/2} Hz)")
    print()

    manifest_path = Path(VALIDATED_MANIFEST)
    if not manifest_path.exists():
        print(f"ERROR: validated manifest not found at {manifest_path}")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)

    # Keep PASS + WARN, drop FAIL
    eligible = manifest[manifest["verdict"].isin(["PASS", "WARN"])].copy()
    n_skipped = len(manifest) - len(eligible)
    print(f"Manifest total  : {len(manifest)}")
    print(f"PASS + WARN     : {len(eligible)}")
    print(f"FAIL (skipped)  : {n_skipped}")
    print()

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    log_rows = []
    t_start = datetime.now()
    for i, row in enumerate(eligible.itertuples(index=False), start=1):
        in_path  = row.full_path
        out_path = mirror_path(in_path)

        if i % 20 == 0 or i == 1:
            elapsed = (datetime.now() - t_start).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(eligible) - i) / rate if rate > 0 else 0
            print(f"  [{i:>3}/{len(eligible)}] {row.fault_label} "
                  f"({row.speed_pct}%) {row.channel}  "
                  f"rate={rate:.2f}/s  ETA={eta:.0f}s")

        status = denoise_file(in_path, out_path)
        status.update({
            "fault_label": row.fault_label,
            "speed_pct":   row.speed_pct,
            "channel":     row.channel,
            "verdict_in":  row.verdict,
        })
        log_rows.append(status)

    total_elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\nDenoised {len(eligible)} files in {total_elapsed:.0f}s "
          f"({len(eligible)/total_elapsed:.2f}/s average).\n")

    # ── Log + summary ──────────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_path = Path(DENOISE_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_ok   = int(log_df["success"].sum())
    n_fail = len(log_df) - n_ok
    print(f"  Successful      : {n_ok}")
    print(f"  Failed          : {n_fail}")
    if n_fail > 0:
        print("\n  Failed files:")
        fails = log_df[~log_df["success"]]
        for _, r in fails.iterrows():
            print(f"    {r['fault_label']} ({r['speed_pct']}%) {r['channel']}: {r['notes']}")

    # Row-count sanity
    bad_rows = log_df[log_df["success"] & ~log_df["row_count_ok"]]
    if len(bad_rows) > 0:
        print(f"\n  NOTE: {len(bad_rows)} files have row count != {EXPECTED_ROWS}. "
              "They were still denoised, just flagging for awareness.")

    # Total cols processed
    tot_cols = int(log_df.loc[log_df["success"], "n_signal_cols"].sum())
    print(f"\n  Total signal columns denoised: {tot_cols:,}")
    print()

    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)
    print(f"  Denoised tree: {OUTPUT_ROOT}")
    print(f"  Log:           {log_path}")
    print()
    print("Next step: vibration splits generation + STFT precompute.")


if __name__ == "__main__":
    main()
