"""
STEP 0 — PC SETUP (one-time per machine)
=========================================
Regenerates manifest.csv and validated_manifest.csv for the new PC root.

What it does
------------
1. Auto-finds the Motor-2 dataset folder under DATA_ROOT (so you don't
   have to know the exact sub-path).
2. Walks the tree and produces a fresh manifest.csv with Windows paths
   that point to THIS machine.
3. If an OLD validated_manifest.csv is found (drop it in C:\\Project Work),
   verdict columns are carried over by matching on
   (speed_pct, fault_label, channel, filename). File contents haven't
   changed across machines, so old verdicts remain valid — no
   re-validation needed.
4. Writes folder_tree.txt and setup_summary.txt for review.

Assumptions
-----------
- Dataset file contents are identical to the previous machine
  (only paths differ).
- You are running from a clean Python env. Only pandas is required.

Usage
-----
    python setup_pc.py

No command-line args — configure the three paths below and run.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
DATA_ROOT              = r"C:\Project Work"
OUTPUT_DIR             = r"C:\Project Work\outputs"
OLD_VALIDATED_MANIFEST = r"C:\Project Work\validated_manifest.csv"  # optional
# ──────────────────────────────────────────────────────────────────────────────

TARGET_CHANNELS = {"ch1", "ch2", "ch3"}  # current channels only

# Columns carried over from the old validated_manifest (added by validate_dataset.py)
VERDICT_COLS = [
    "verdict", "readable", "row_count_ok", "has_nan",
    "zero_cols", "clipped_cols", "rms_min", "rms_max", "rms_flag", "notes",
]


def find_motor2_root(root: Path) -> Path | None:
    """Walk the tree under `root` and return the first folder named Motor-2."""
    if not root.exists():
        return None
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "motor-2":
            return p
    return None


def get_channel_name(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    for ch in TARGET_CHANNELS:
        if stem.endswith(ch):
            return ch
    return None


def count_signal_columns(filepath: Path) -> int | str:
    """Header-only read; return (cols - 1) to exclude the time column."""
    try:
        df = pd.read_csv(filepath, nrows=1)
        return max(0, len(df.columns) - 1)
    except Exception as e:
        return f"ERROR: {e}"


def scan(motor2_root: Path):
    records = []
    tree_lines = [f"ROOT: {motor2_root}"]

    for speed_dir in sorted(motor2_root.iterdir()):
        if not speed_dir.is_dir():
            continue
        speed = speed_dir.name  # "100" / "75" / "50"
        tree_lines.append(f"  ├── Speed: {speed}%")

        for fault_dir in sorted(speed_dir.iterdir()):
            if not fault_dir.is_dir():
                continue
            fault = fault_dir.name
            tree_lines.append(f"  │   ├── Fault: {fault}")

            csv_files = sorted(f for f in fault_dir.iterdir() if f.suffix.lower() == ".csv")

            for f in csv_files:
                ch = get_channel_name(f.name)
                if ch is None:
                    tree_lines.append(f"  │   │   └── [SKIP]    {f.name}")
                    continue

                n_cols = count_signal_columns(f)
                tree_lines.append(
                    f"  │   │   ├── [CURRENT] {f.name}  |  signal_cols={n_cols}"
                )
                records.append({
                    "full_path":      str(f),
                    "motor":          "Motor-2",
                    "speed_pct":      speed,
                    "fault_label":    fault,
                    "channel":        ch,
                    "filename":       f.name,
                    "signal_columns": n_cols,
                    "usable": (
                        "CHECK" if isinstance(n_cols, str)
                        else ("YES" if n_cols > 0 else "EMPTY")
                    ),
                })
        tree_lines.append("  │")

    return records, tree_lines


def carry_over_verdicts(new_manifest: pd.DataFrame,
                        old_validated_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Join new manifest with old validated manifest on (speed_pct, fault_label,
    channel, filename) and bring across all verdict columns. Match key is
    content-addressable — does not depend on full_path.
    """
    stats = {"old_rows": 0, "new_rows": len(new_manifest), "matched": 0, "unmatched": 0}

    if not old_validated_path.exists():
        for c in VERDICT_COLS:
            new_manifest[c] = pd.NA
        return new_manifest, stats

    old = pd.read_csv(old_validated_path, dtype={"speed_pct": str})
    stats["old_rows"] = len(old)

    # Normalize join keys to strings
    new_manifest["speed_pct"] = new_manifest["speed_pct"].astype(str)
    old["speed_pct"] = old["speed_pct"].astype(str)

    keys = ["speed_pct", "fault_label", "channel", "filename"]
    cols_to_bring = [c for c in VERDICT_COLS if c in old.columns]

    merged = new_manifest.merge(
        old[keys + cols_to_bring], on=keys, how="left", validate="one_to_one"
    )

    stats["matched"]   = merged["verdict"].notna().sum() if "verdict" in merged.columns else 0
    stats["unmatched"] = len(merged) - stats["matched"]
    return merged, stats


def main():
    print("=" * 70)
    print("PC SETUP")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    data_root = Path(DATA_ROOT)
    out_dir   = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find Motor-2
    print(f"\n[1/4] Searching for Motor-2 folder under: {data_root}")
    motor2 = find_motor2_root(data_root)
    if motor2 is None:
        print(f"  ✗ No folder named 'Motor-2' found under {data_root}.")
        print(f"    Set DATA_ROOT to the parent of the Motor-2 folder and re-run.")
        sys.exit(1)
    print(f"  ✓ Found: {motor2}")

    # 2. Scan
    print(f"\n[2/4] Scanning tree...")
    records, tree_lines = scan(motor2)
    manifest = pd.DataFrame(records)
    print(f"  ✓ Found {len(manifest)} current-channel files across "
          f"{manifest['fault_label'].nunique()} fault labels "
          f"and {manifest['speed_pct'].nunique()} speeds.")

    # 3. Carry over verdicts
    print(f"\n[3/4] Carrying over validation verdicts...")
    old_path = Path(OLD_VALIDATED_MANIFEST)
    validated, stats = carry_over_verdicts(manifest, old_path)
    if stats["old_rows"] == 0:
        print(f"  ! No old validated_manifest.csv at {old_path}")
        print(f"    validated_manifest.csv will be written with empty verdict columns.")
    else:
        print(f"  ✓ Old manifest rows : {stats['old_rows']}")
        print(f"    Matched           : {stats['matched']}")
        print(f"    Unmatched (new)   : {stats['unmatched']}")

    # 4. Write outputs
    print(f"\n[4/4] Writing outputs to {out_dir} ...")
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    validated.to_csv(out_dir / "validated_manifest.csv", index=False)
    (out_dir / "folder_tree.txt").write_text("\n".join(tree_lines), encoding="utf-8")

    summary = []
    summary.append(f"PC Setup Summary — {datetime.now().isoformat(timespec='seconds')}")
    summary.append("=" * 60)
    summary.append(f"Data root           : {data_root}")
    summary.append(f"Motor-2 folder      : {motor2}")
    summary.append(f"Output dir          : {out_dir}")
    summary.append("")
    summary.append(f"Files scanned       : {len(manifest)}")
    summary.append(f"Unique fault labels : {manifest['fault_label'].nunique()}")
    summary.append(f"Unique speeds       : {sorted(manifest['speed_pct'].unique())}")
    summary.append(f"Total signal cols   : "
                   f"{manifest['signal_columns'].apply(lambda x: x if isinstance(x, int) else 0).sum()}")
    summary.append("")
    summary.append("Files per fault label:")
    summary.append(manifest.groupby('fault_label')['channel'].count().to_string())
    summary.append("")
    summary.append("Files per speed:")
    summary.append(manifest['speed_pct'].value_counts().to_string())
    summary.append("")
    if stats["old_rows"] > 0:
        summary.append(f"Verdict carry-over: {stats['matched']}/{stats['new_rows']} matched")
        if "verdict" in validated.columns and validated["verdict"].notna().any():
            summary.append("Verdict breakdown:")
            summary.append(validated["verdict"].value_counts(dropna=False).to_string())
    else:
        summary.append("Verdict carry-over: SKIPPED (old validated_manifest.csv not found)")

    (out_dir / "setup_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("  ✓ manifest.csv")
    print("  ✓ validated_manifest.csv")
    print("  ✓ folder_tree.txt")
    print("  ✓ setup_summary.txt")
    print("\nDone. Upload setup_summary.txt + folder_tree.txt back to the chat.")


if __name__ == "__main__":
    main()
