"""
STEP 1 — VIBRATION DATASET SCANNER
===================================
Walks the Vibration\Motor-2 dataset folder, identifies every accelerometer
channel file (ch1..ch5), and outputs:
  1. A printed tree of the folder structure
  2. vibration_manifest.csv : one row per file with full path, fault label,
     speed, channel, sensor location, and column count
  3. vibration_tree.txt     : the full printed tree saved for reference

DIFFERENCES vs. scan_dataset.py (current branch)
------------------------------------------------
  - Scans Vibration\Motor-2 instead of Electric\Motor-2
  - Targets 5 channels (ch1..ch5), not 3 — all accelerometers, no voltage
  - Records sensor_location per channel for downstream traceability
  - Does NOT filter by class — emits all rows so the user can see the full
    structure before filtering happens at the splits stage

NO VALIDATION IS DONE HERE — just a structural walk and column count per file.
Content quality checks (row count, flat cols, clipping, RMS) come in step 2.

Usage:
    python vibration_scan.py

Set DATASET_ROOT and OUTPUT_DIR below.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
DATASET_ROOT = r"C:\Project Work\Dataset\Vibration\Motor-2"
OUTPUT_DIR   = r"C:\Project Work\outputs\vibration"
# ──────────────────────────────────────────────────────────────────────────────

# Five accelerometer channels — all target, none skipped
TARGET_CHANNELS = {"ch1", "ch2", "ch3", "ch4", "ch5"}

# Physical sensor locations (from dataset README)
SENSOR_LOCATION = {
    "ch1": "Motor NDE horizontal",
    "ch2": "Motor DE vertical",
    "ch3": "Motor DE axial",
    "ch4": "Pump DE horizontal",
    "ch5": "Pump NDE vertical",
}


def get_channel_name(filename: str) -> str | None:
    """Extract channel id from filename, e.g. 'Vibration_Motor-2_50_time-healthy 1-ch1.csv' -> 'ch1'"""
    stem = Path(filename).stem.lower()
    for ch in TARGET_CHANNELS:
        if stem.endswith(ch):
            return ch
    return None


def count_signal_columns(filepath: str):
    """Return number of signal columns (total columns minus timestamp column)."""
    try:
        df = pd.read_csv(filepath, nrows=2)
        return max(0, len(df.columns) - 1)
    except Exception as e:
        return f"ERROR: {e}"


def scan(root: str):
    root_path = Path(root)
    if not root_path.exists():
        print(f"\nERROR: dataset root not found at {root}")
        print("Check DATASET_ROOT at top of script.")
        sys.exit(1)

    records = []
    tree_lines = []
    tree_lines.append(f"ROOT: {root_path}")
    tree_lines.append("")

    for speed_dir in sorted(root_path.iterdir()):
        if not speed_dir.is_dir():
            continue
        speed = speed_dir.name  # "50", "75", "100"
        tree_lines.append(f"├── Speed: {speed}%")

        for fault_dir in sorted(speed_dir.iterdir()):
            if not fault_dir.is_dir():
                continue
            fault = fault_dir.name
            tree_lines.append(f"│   ├── Fault: {fault}")

            csv_files = sorted(
                [f for f in fault_dir.iterdir() if f.suffix.lower() == ".csv"]
            )
            vib_files   = [f for f in csv_files if get_channel_name(f.name) is not None]
            other_files = [f for f in csv_files if get_channel_name(f.name) is None]

            # Sort vib files by channel number so ch1..ch5 print in order
            vib_files.sort(key=lambda f: get_channel_name(f.name))

            for f in vib_files:
                ch = get_channel_name(f.name)
                n_cols = count_signal_columns(str(f))
                loc = SENSOR_LOCATION[ch]
                tree_lines.append(
                    f"│   │   ├── [VIB] {f.name}  |  cols={n_cols}  |  {loc}"
                )
                records.append({
                    "full_path":       str(f),
                    "motor":           "Motor-2",
                    "speed_pct":       speed,
                    "fault_label":     fault,
                    "channel":         ch,
                    "sensor_location": loc,
                    "filename":        f.name,
                    "signal_columns":  n_cols,
                    "usable":          (
                        "CHECK" if isinstance(n_cols, str)
                        else ("YES" if n_cols > 0 else "EMPTY")
                    ),
                })

            for f in other_files:
                tree_lines.append(f"│   │   └── [UNKNOWN] {f.name}")

        tree_lines.append("│")

    return records, tree_lines


def summarize(df: pd.DataFrame) -> list[str]:
    """Return human-readable summary lines."""
    lines = []
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total files found          : {len(df)}")
    lines.append(f"Unique fault labels        : {df['fault_label'].nunique()}")
    lines.append(f"Unique speeds              : {sorted(df['speed_pct'].unique())}")
    lines.append("")

    # Channel completeness check — every (fault, speed) should have all 5 channels
    lines.append("Channel completeness per (fault, speed):")
    pivot = (df.groupby(["fault_label", "speed_pct", "channel"])
               .size().unstack(fill_value=0))
    # Ensure all expected channel columns exist
    for ch in sorted(TARGET_CHANNELS):
        if ch not in pivot.columns:
            pivot[ch] = 0
    pivot = pivot[sorted(TARGET_CHANNELS)]
    pivot["complete"] = (pivot.sum(axis=1) == len(TARGET_CHANNELS)).map(
        {True: "YES", False: "NO "}
    )
    lines.append(pivot.to_string())
    lines.append("")

    # Files per fault label
    lines.append("Files per fault label:")
    fp = (df.groupby("fault_label")["signal_columns"]
            .agg(["count", "sum"])
            .rename(columns={"count": "n_files", "sum": "total_cols"}))
    lines.append(fp.to_string())
    lines.append("")

    # Files per speed
    lines.append("Files per speed:")
    lines.append(df["speed_pct"].value_counts().sort_index().to_string())
    lines.append("")

    # Column-count distribution
    # Vibration recordings are 12 s each; each 60 s measurement yields 5 columns.
    # So signal_columns should normally be a multiple of 5.
    lines.append("Signal-columns-per-file distribution:")
    numeric_cols = pd.to_numeric(df["signal_columns"], errors="coerce").dropna().astype(int)
    lines.append(numeric_cols.value_counts().sort_index().to_string())
    off_by_5 = numeric_cols[numeric_cols % 5 != 0]
    if len(off_by_5) > 0:
        lines.append("")
        lines.append(f"NOTE: {len(off_by_5)} files have signal_columns NOT divisible by 5 "
                     f"(vibration chunks of 12 s suggest multiples of 5 per 60-s recording). "
                     f"This may indicate partial recordings — fine, just a flag.")
    lines.append("")

    # Anomalies
    empty = df[df["signal_columns"] == 0]
    error = df[df["usable"] == "CHECK"]
    if len(empty) > 0:
        lines.append(f"EMPTY FILES ({len(empty)}):")
        lines.append(empty[["fault_label", "speed_pct", "filename"]].to_string(index=False))
        lines.append("")
    if len(error) > 0:
        lines.append(f"UNREADABLE FILES ({len(error)}):")
        lines.append(error[["fault_label", "speed_pct", "filename"]].to_string(index=False))
        lines.append("")
    if len(empty) == 0 and len(error) == 0:
        lines.append("No empty or unreadable files detected.")
        lines.append("")

    # 4-class preview — how many files per target class?
    target_classes = [
        "healthy 1",
        "stator short 1",
        "bearing bpfo 3",
        "broken rotor bar",
    ]
    lines.append("4-class preview (target classes for Run A):")
    for cls in target_classes:
        sub = df[df["fault_label"] == cls]
        if len(sub) == 0:
            lines.append(f"  {cls:<22}  NOT FOUND in this scan")
            continue
        ch1_rows = sub[sub["channel"] == "ch1"]
        cols_per_speed = ch1_rows.set_index("speed_pct")["signal_columns"].to_dict()
        lines.append(f"  {cls:<22}  ch1 cols per speed: {cols_per_speed}")
    lines.append("")

    return lines


def main():
    print("=" * 70)
    print("VIBRATION DATASET SCAN")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Scanning: {DATASET_ROOT}\n")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, tree = scan(DATASET_ROOT)
    df = pd.DataFrame(records)

    # Print + save tree
    for line in tree:
        print(line)

    tree_path = out_dir / "vibration_tree.txt"
    tree_path.write_text("\n".join(tree), encoding="utf-8")

    # Print + save summary
    print()
    summary = summarize(df)
    for line in summary:
        print(line)

    # Save manifest
    manifest_path = out_dir / "vibration_manifest.csv"
    df.to_csv(manifest_path, index=False)

    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)
    print(f"  {manifest_path}")
    print(f"  {tree_path}")
    print()
    print("Next step: run vibration_validate.py against the manifest.")


if __name__ == "__main__":
    main()
