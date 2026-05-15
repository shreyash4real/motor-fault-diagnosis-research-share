"""
STEP 3 (v2) — SPEED-STRATIFIED COLUMN SPLITS (4-class) [NO-BPFO-100 VERSION]
====================================================================
Reads the canonical splits.csv and reassigns every `bearing bpfo 3` column
at 100% speed currently marked `test` to `train`. The result: the test set
has no bpfo-3 @ 100% data at all, so the residual col5@100% issue cannot
contaminate the test metrics.

OUTPUT
------
  C:\\Project Work\\Outputs_nobpfo100\\splits.csv
  C:\\Project Work\\Outputs_nobpfo100\\split_report.txt

Usage
-----
    python generate_splits_nobpfo100.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
CANONICAL_SPLITS = r"C:\Project Work\Outputs\4class\v2_speed_strat\splits.csv"
OUT_DIR          = r"C:\Project Work\Outputs_nobpfo100"
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"GENERATE SPLITS — nobpfo100 (exclude bpfo-3 @ 100% from test)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    val_path = Path(CANONICAL_SPLITS)
    out_dir  = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        print(f"ERROR: {val_path} not found.")
        sys.exit(1)

    splits_df = pd.read_csv(val_path)

    # Move every bpfo-3 @ 100% test row to train. Train/val/test split is
    # otherwise unchanged from canonical v2.
    target_mask = (
        (splits_df["class_label"] == "bearing bpfo 3")
        & (splits_df["speed_pct"] == 100)
        & (splits_df["split"] == "test")
    )
    n_moved = int(target_mask.sum())
    moved_cols = sorted(splits_df.loc[target_mask, "col_index"].unique().tolist())
    splits_df.loc[target_mask, "split"] = "train"
    print(f"\nMoved {n_moved} rows (cols {moved_cols}) of bpfo-3 @ 100% from test → train")

    # Sanity: no bpfo-3 @ 100% should remain in test.
    remaining = splits_df[
        (splits_df["class_label"] == "bearing bpfo 3")
        & (splits_df["speed_pct"] == 100)
        & (splits_df["split"] == "test")
    ]
    assert len(remaining) == 0, f"FAILED: {len(remaining)} bpfo-3@100% rows still in test"

    splits_path = out_dir / "splits.csv"
    splits_df.to_csv(splits_path, index=False)
    print(f"\nWrote {splits_path}  ({len(splits_df)} rows)")

    ch1_only = splits_df[splits_df["full_path"].str.contains("-ch1.csv")]
    report = []
    report.append(f"Split Report — nobpfo100 - {datetime.now().isoformat(timespec='seconds')}")
    report.append("=" * 60)
    report.append(f"Moved {n_moved} bpfo-3 @ 100% test rows (cols {moved_cols}) to train.")
    report.append("")

    cls_split = ch1_only.groupby(["class_label", "split"]).size().unstack(fill_value=0)
    if "train" in cls_split.columns:
        cls_split = cls_split[[c for c in ["train", "val", "test"] if c in cls_split.columns]]
    cls_split["total"] = cls_split.sum(axis=1)
    report.append("COLUMNS per class x split:")
    report.append(cls_split.to_string())
    report.append("")

    spd_split = ch1_only.groupby(["class_label", "split", "speed_pct"]).size().unstack(fill_value=0)
    report.append("COLUMNS per class x split x speed:")
    report.append(spd_split.to_string())
    report.append("")

    (out_dir / "split_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out_dir / 'split_report.txt'}")

if __name__ == "__main__":
    main()
