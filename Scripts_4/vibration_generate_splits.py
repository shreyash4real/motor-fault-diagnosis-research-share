"""
STEP 4 — VIBRATION SPEED-STRATIFIED COLUMN SPLITS (4-class)
============================================================
Direct mirror of generate_splits_v2.py for the vibration branch.

Same design:
  - Column-level splits (no segment leakage across splits)
  - Stratified WITHIN each (class, speed) group so every class gets
    honest train/val/test coverage at every speed
  - 70 / 15 / 15 ratios, with a floor of >=1 col per split when possible
  - Seed 42 for reproducibility

What changes vs the current-branch version:
  - Manifest path: vibration_validated_manifest.csv
  - Each column emits FIVE rows (ch1..ch5), not three (ch1..ch3)
  - N_SEGMENTS = 45 (12-s recording) instead of 57 (15-s recording)
  - Cross-channel n_cols equality check extended across all 5 channels
  - Per-class RNG seeded deterministically via sha256 of the class name
    (the current branch used hash(), which is non-deterministic across
    Python processes without PYTHONHASHSEED set)

OUTPUT
------
  C:\\Project Work\\outputs\\vibration\\v2_speed_strat\\splits.csv
  C:\\Project Work\\outputs\\vibration\\v2_speed_strat\\split_report.txt

Usage
-----
    python vibration_generate_splits.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
VALIDATED_MANIFEST = r"C:\Project Work\outputs\vibration\vibration_validated_manifest.csv"
OUTPUT_DIR         = r"C:\Project Work\outputs\vibration\v2_speed_strat"

CLASS_NAMES = [
    "healthy 1",
    "stator short 1",
    "bearing bpfo 3",
    "broken rotor bar",
]

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC inferred as 1 - TRAIN_FRAC - VAL_FRAC

SEED = 42

CHANNELS = ("ch1", "ch2", "ch3", "ch4", "ch5")
# ──────────────────────────────────────────────────────────────────────────────

# ─── RECORDING GEOMETRY (vibration) ───────────────────────────────────────────
FS                    = 20_000
RECORDING_SECONDS     = 12
SIGNAL_LENGTH_SAMPLES = FS * RECORDING_SECONDS     # 240_000
WINDOW_SAMPLES        = 20_000                     # 1 s
OVERLAP_FRAC          = 0.75
STRIDE_SAMPLES        = int(WINDOW_SAMPLES * (1 - OVERLAP_FRAC))  # 5000
N_SEGMENTS            = (SIGNAL_LENGTH_SAMPLES - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1  # 45
# ──────────────────────────────────────────────────────────────────────────────


def allocate_splits_for_group(n: int) -> tuple[int, int, int]:
    """
    Given n columns at one (class, speed), return (n_train, n_val, n_test).
    Ensures val and test each get at least 1 column whenever n >= 3.
    Same rule as generate_splits_v2.py.
    """
    if n == 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    if n == 3:
        return 1, 1, 1
    n_val  = max(1, round(n * VAL_FRAC))
    n_test = max(1, round(n * (1.0 - TRAIN_FRAC - VAL_FRAC)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_test  = n - n_train - n_val
    return n_train, n_val, n_test


def class_seed(cls: str) -> int:
    """
    Deterministic per-class offset, so each class splits independently
    of its position in CLASS_NAMES. sha256-based instead of hash() so
    runs are reproducible without PYTHONHASHSEED set.
    """
    h = hashlib.sha256(cls.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10_000


def build_class_splits(class_manifest: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
    """For one class, split EACH (speed) group independently."""
    class_name = class_manifest.iloc[0]["fault_label"]
    rows = []

    # Check n_cols agrees across all 5 channels within each (speed)
    for speed, grp in class_manifest.groupby("speed_pct"):
        cols_per_ch = grp.set_index("channel")["signal_columns"].to_dict()
        counts = [cols_per_ch.get(ch) for ch in CHANNELS]
        if len(set(counts)) != 1 or counts[0] is None:
            raise ValueError(
                f"Column count mismatch for {class_name} at {speed}%: "
                f"{cols_per_ch}"
            )

    for speed in sorted(class_manifest["speed_pct"].unique()):
        ch1_row = class_manifest[
            (class_manifest["speed_pct"] == speed)
            & (class_manifest["channel"] == "ch1")
        ].iloc[0]
        n_cols_here = int(ch1_row["signal_columns"])

        n_tr, n_va, n_te = allocate_splits_for_group(n_cols_here)
        labels = (["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)
        assert len(labels) == n_cols_here

        perm = rng.permutation(n_cols_here)
        split_for_col = np.array(labels)[perm]

        for ci in range(n_cols_here):
            for ch in CHANNELS:
                match = class_manifest[
                    (class_manifest["speed_pct"] == speed)
                    & (class_manifest["channel"] == ch)
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"Expected 1 {ch} file for {class_name} at {speed}%, "
                        f"found {len(match)}."
                    )
                rec = match.iloc[0]
                rows.append({
                    "full_path":       rec["full_path"],
                    "class_label":     class_name,
                    "fault_label":     class_name,
                    "speed_pct":       speed,
                    "channel":         ch,
                    "sensor_location": rec.get("sensor_location", ""),
                    "col_index":       ci,
                    "split":           split_for_col[ci],
                    "n_segments":      N_SEGMENTS,
                })

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("GENERATE SPLITS (vibration) — speed-stratified, 4-class, 5-channel")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    val_path = Path(VALIDATED_MANIFEST)
    out_dir  = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        print(f"ERROR: validated manifest not found at {val_path}")
        sys.exit(1)

    manifest = pd.read_csv(val_path)
    manifest = manifest[manifest["verdict"].isin(["PASS", "WARN"])]
    manifest = manifest[manifest["fault_label"].isin(CLASS_NAMES)].copy()

    print(f"\nFiltered manifest: {len(manifest)} files across "
          f"{manifest['fault_label'].nunique()} classes")
    per_class = manifest.groupby("fault_label").size()
    print("Files per class (should be 5 channels * 3 speeds = 15 per class):")
    print(per_class.to_string())
    for c in CLASS_NAMES:
        if c not in per_class.index:
            print(f"\nERROR: class '{c}' not found in manifest.")
            sys.exit(1)

    print()
    print(f"Split ratios    : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / "
          f"{1 - TRAIN_FRAC - VAL_FRAC:.0%}")
    print(f"Stratify by     : (class x speed)")
    print(f"Seed            : {SEED}")
    print(f"Segments/column : {N_SEGMENTS}   ({RECORDING_SECONDS}s recording, "
          f"1s window, 75% overlap)")
    print()

    all_rows = []
    for cls in CLASS_NAMES:
        class_manifest = manifest[manifest["fault_label"] == cls]
        rng = np.random.default_rng(SEED + class_seed(cls))
        class_splits = build_class_splits(class_manifest, rng)
        all_rows.append(class_splits)

        # Count unique columns (not rows — rows are 5x columns due to channels)
        ch1_rows = class_splits[class_splits["channel"] == "ch1"]
        n_train = (ch1_rows["split"] == "train").sum()
        n_val   = (ch1_rows["split"] == "val").sum()
        n_test  = (ch1_rows["split"] == "test").sum()
        print(f"  {cls:<20}  cols: train={n_train:>3}  val={n_val:>3}  "
              f"test={n_test:>3}   "
              f"segs: train={n_train*N_SEGMENTS:>5}  "
              f"val={n_val*N_SEGMENTS:>5}  test={n_test*N_SEGMENTS:>5}")

    splits_df = pd.concat(all_rows, ignore_index=True)
    splits_path = out_dir / "splits.csv"
    splits_df.to_csv(splits_path, index=False)
    print(f"\nWrote {splits_path}  ({len(splits_df)} rows = 5 channels x "
          f"{len(splits_df) // 5} columns)")

    # ── Report ────────────────────────────────────────────────────────────
    ch1_only = splits_df[splits_df["channel"] == "ch1"]
    report = []
    report.append(f"Vibration split report (4-class, 5-channel, speed-stratified) - "
                  f"{datetime.now().isoformat(timespec='seconds')}")
    report.append("=" * 70)
    report.append(f"Classes         : {CLASS_NAMES}")
    report.append(f"Channels        : {list(CHANNELS)}")
    report.append(f"Split fractions : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / "
                  f"{1 - TRAIN_FRAC - VAL_FRAC:.0%}")
    report.append(f"Stratify by     : (class x speed)")
    report.append(f"Seed            : {SEED}   (per-class offset via sha256)")
    report.append(f"Segments/column : {N_SEGMENTS}")
    report.append("")

    cls_split = (ch1_only
                 .groupby(["class_label", "split"])
                 .size()
                 .unstack(fill_value=0))
    for col in ("train", "val", "test"):
        if col not in cls_split.columns:
            cls_split[col] = 0
    cls_split = cls_split[["train", "val", "test"]]
    cls_split["total"] = cls_split.sum(axis=1)
    report.append("COLUMNS per class x split:")
    report.append(cls_split.to_string())
    report.append("")

    seg_split = cls_split[["train", "val", "test"]] * N_SEGMENTS
    seg_split["total"] = seg_split.sum(axis=1)
    report.append(f"SEGMENTS per class x split (x{N_SEGMENTS} segs/col):")
    report.append(seg_split.to_string())
    report.append("")

    speed_x = (ch1_only
               .groupby(["class_label", "split", "speed_pct"])
               .size()
               .unstack(["split", "speed_pct"], fill_value=0))
    report.append("COLUMNS per class x split x speed:")
    report.append(speed_x.to_string())
    report.append("")

    # Sanity: total samples per channel should match
    ch_counts = (splits_df.groupby(["channel", "split"]).size()
                         .unstack(fill_value=0))
    report.append("ROWS per channel x split (sanity — should be equal across channels):")
    report.append(ch_counts.to_string())
    report.append("")

    report_path = out_dir / "split_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path}")

    # Print the per-class summary table to stdout too
    print()
    print("=" * 70)
    print("CLASS x SPLIT (columns):")
    print("=" * 70)
    print(cls_split.to_string())
    print()
    print(f"Total train segments: {int(seg_split.loc[:, 'train'].sum()):,}")
    print(f"Total val   segments: {int(seg_split.loc[:, 'val'].sum()):,}")
    print(f"Total test  segments: {int(seg_split.loc[:, 'test'].sum()):,}")
    print()
    print("Next step: run vibration_precompute_stft.py against this splits.csv.")


if __name__ == "__main__":
    main()
