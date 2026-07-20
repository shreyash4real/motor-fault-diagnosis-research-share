"""
STEP 3 (v2) — SPEED-STRATIFIED COLUMN SPLITS (4-class)
=======================================================
Replaces the random-per-class split with a speed-stratified one.

WHY THIS VERSION EXISTS
-----------------------
v1 (random within class, pool speeds) left the bearing bpfo 3 class with:
    val  = 3 cols, all at 50% speed
    test = 3 cols, all at 75/100% speed (no 50%)
That split worked OK for other classes by luck, but for bpfo 3 it meant
the model validated on 50%-speed features and got tested on 75/100%-
speed features — so val F1 looked great but test F1 cratered to 0.564.

v2 stratifies WITHIN each (class, speed) pair. Every class now gets
train/val/test coverage at every speed. Overall totals shift slightly
(because small-n groups round differently) but every split has honest
cross-speed coverage.

OUTPUT
------
  C:\\Project Work\\outputs\\4class\\v2_speed_strat\\splits.csv
  C:\\Project Work\\outputs\\4class\\v2_speed_strat\\split_report.txt

Usage
-----
    python generate_splits.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pipeline_paths import legacy_path

import numpy as np
import pandas as pd

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
VALIDATED_MANIFEST = legacy_path(r"C:\Project Work\outputs\main\validated_manifest.csv")
OUTPUT_DIR         = legacy_path(r"C:\Project Work\outputs\4class\v2_speed_strat")

CLASS_NAMES = [
    "healthy 1",
    "stator short 1",
    "bearing bpfo 3",
    "broken rotor bar",
]

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

SIGNAL_LENGTH_SAMPLES = 300_000
WINDOW_SAMPLES        = 20_000
OVERLAP_FRAC          = 0.75
STRIDE_SAMPLES        = int(WINDOW_SAMPLES * (1 - OVERLAP_FRAC))
N_SEGMENTS            = (SIGNAL_LENGTH_SAMPLES - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1  # 57


def allocate_splits_for_group(n: int) -> tuple[int, int, int]:
    """
    Given n columns at one (class, speed), return (n_train, n_val, n_test).
    Ensures val and test each get at least 1 column whenever n >= 3.
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


def build_class_splits(class_manifest: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
    """For one class, split EACH (speed) group independently."""
    for speed, grp in class_manifest.groupby("speed_pct"):
        cols_per_ch = grp.set_index("channel")["signal_columns"].to_dict()
        if not (cols_per_ch.get("ch1") == cols_per_ch.get("ch2")
                == cols_per_ch.get("ch3")):
            raise ValueError(
                f"Column count mismatch for "
                f"{class_manifest.iloc[0]['fault_label']} at {speed}%: "
                f"{cols_per_ch}")

    class_name = class_manifest.iloc[0]["fault_label"]
    rows = []

    for speed in sorted(class_manifest["speed_pct"].unique()):
        ch1_row = class_manifest[
            (class_manifest["speed_pct"] == speed) &
            (class_manifest["channel"] == "ch1")
        ].iloc[0]
        n_cols_here = int(ch1_row["signal_columns"])

        n_tr, n_va, n_te = allocate_splits_for_group(n_cols_here)
        labels = (["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)
        assert len(labels) == n_cols_here

        perm = rng.permutation(n_cols_here)
        split_for_col = np.array(labels)[perm]

        for ci in range(n_cols_here):
            for ch in ("ch1", "ch2", "ch3"):
                match = class_manifest[
                    (class_manifest["speed_pct"] == speed) &
                    (class_manifest["channel"] == ch)
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"Expected 1 {ch} file for {class_name} at {speed}%, "
                        f"found {len(match)}.")
                rows.append({
                    "full_path":   match.iloc[0]["full_path"],
                    "class_label": class_name,
                    "fault_label": class_name,
                    "speed_pct":   speed,
                    "col_index":   ci,
                    "split":       split_for_col[ci],
                    "n_segments":  N_SEGMENTS,
                })

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("GENERATE SPLITS v2 — speed-stratified (4-class)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    val_path = Path(VALIDATED_MANIFEST)
    out_dir  = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        print(f"ERROR: {val_path} not found.")
        sys.exit(1)

    manifest = pd.read_csv(val_path)
    manifest = manifest[manifest["verdict"].isin(["PASS", "WARN"])]
    manifest = manifest[manifest["fault_label"].isin(CLASS_NAMES)].copy()

    print(f"\nFiltered manifest: {len(manifest)} files across "
          f"{manifest['fault_label'].nunique()} classes")
    per_class = manifest.groupby("fault_label").size()
    print("Files per class:")
    print(per_class.to_string())
    for c in CLASS_NAMES:
        if c not in per_class.index:
            print(f"ERROR: class '{c}' not found.")
            sys.exit(1)

    print(f"\nSplit ratios    : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / "
          f"{1 - TRAIN_FRAC - VAL_FRAC:.0%}")
    print(f"Stratify by     : (class x speed)")
    print(f"Seed            : {SEED}")
    print(f"Segments/column : {N_SEGMENTS}")
    print()

    all_rows = []
    for cls in CLASS_NAMES:
        class_manifest = manifest[manifest["fault_label"] == cls]
        class_rng = np.random.default_rng(SEED + hash(cls) % 10_000)
        class_splits = build_class_splits(class_manifest, class_rng)
        all_rows.append(class_splits)

        ch1_rows = class_splits[class_splits["full_path"].str.contains("-ch1.csv")]
        n_train = (ch1_rows["split"] == "train").sum()
        n_val   = (ch1_rows["split"] == "val").sum()
        n_test  = (ch1_rows["split"] == "test").sum()
        print(f"  {cls:<20}  cols: train={n_train:>2}  val={n_val:>2}  "
              f"test={n_test:>2}   "
              f"segs: train={n_train*N_SEGMENTS:>4}  val={n_val*N_SEGMENTS:>4}  "
              f"test={n_test*N_SEGMENTS:>4}")

    splits_df = pd.concat(all_rows, ignore_index=True)
    splits_path = out_dir / "splits.csv"
    splits_df.to_csv(splits_path, index=False)
    print(f"\nWrote {splits_path}  ({len(splits_df)} rows)")

    ch1_only = splits_df[splits_df["full_path"].str.contains("-ch1.csv")]
    report = []
    report.append(f"Split Report v2 (speed-stratified) - "
                  f"{datetime.now().isoformat(timespec='seconds')}")
    report.append("=" * 60)
    report.append(f"Classes         : {CLASS_NAMES}")
    report.append(f"Split fractions : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / "
                  f"{1 - TRAIN_FRAC - VAL_FRAC:.0%}")
    report.append(f"Stratify by     : (class x speed)")
    report.append(f"Seed            : {SEED}")
    report.append(f"Segments/column : {N_SEGMENTS}")
    report.append("")

    cls_split = (ch1_only
                 .groupby(["class_label", "split"])
                 .size()
                 .unstack(fill_value=0))
    if "train" in cls_split.columns:
        cls_split = cls_split[["train", "val", "test"]]
    cls_split["total"] = cls_split.sum(axis=1)
    report.append("COLUMNS per class x split:")
    report.append(cls_split.to_string())
    report.append("")

    seg_split = cls_split * N_SEGMENTS
    seg_split["total"] = seg_split[["train", "val", "test"]].sum(axis=1)
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

    (out_dir / "split_report.txt").write_text("\n".join(report),
                                               encoding="utf-8")
    print(f"Wrote {out_dir / 'split_report.txt'}")


if __name__ == "__main__":
    main()
