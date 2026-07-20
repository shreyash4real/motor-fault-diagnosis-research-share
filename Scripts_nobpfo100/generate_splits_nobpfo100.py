"""Create an auditable source-level split for the motor-fault pipeline.

By default this copies and validates the supplied canonical split unchanged.
The historical ``nobpfo100`` exclusion is retained as an explicit scoped
evaluation because the available current measurement could not reliably
separate one BPFO-3-at-100% source column from the healthy operating regime.
It is valid only when reported as that bounded operating envelope, never as a
claim of full condition coverage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from validate_experiment import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-splits", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy-exclude-bpfo100",
        action="store_true",
        help="Create the historical scoped evaluation with BPFO-3 at 100%% absent from test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.canonical_splits.is_file():
        raise SystemExit(f"Canonical splits not found: {args.canonical_splits}")

    splits = pd.read_csv(args.canonical_splits)
    # Establish that the source file is a valid, fully covered evaluation split
    # before copying it or intentionally producing a marked legacy ablation.
    validate(splits, allow_missing_test_strata=False)

    scope = "canonical"
    moved_columns: list[int] = []
    if args.legacy_exclude_bpfo100:
        scope = "legacy_nobpfo100_ablation"
        target = (
            (splits["class_label"] == "bearing bpfo 3")
            & (splits["speed_pct"] == 100)
            & (splits["split"] == "test")
        )
        moved_columns = sorted(splits.loc[target, "col_index"].unique().tolist())
        splits.loc[target, "split"] = "train"
        validate(splits, allow_missing_test_strata=True)
    else:
        validate(splits, allow_missing_test_strata=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = args.out_dir / "splits.csv"
    splits.to_csv(splits_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "canonical_splits": str(args.canonical_splits.resolve()),
        "output_splits": str(splits_path.resolve()),
        "legacy_exclude_bpfo100": args.legacy_exclude_bpfo100,
        "moved_bpfo100_test_column_indices": moved_columns,
        "metric_scope": (
            "bounded_operating_envelope" if args.legacy_exclude_bpfo100 else "full_split_coverage"
        ),
        "warning": (
            "This split has no BPFO-3 at 100% speed in test because the available current measurement "
            "could not reliably distinguish that operating condition. Report metrics only within this declared scope."
            if args.legacy_exclude_bpfo100
            else "Canonical split copied unchanged and validated for source-level isolation and class/speed test coverage."
        ),
    }
    (args.out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    ch1 = splits[splits["full_path"].astype(str).str.contains(r"-ch1\.csv$", case=False, regex=True)]
    report = [
        f"Split report — {scope}",
        f"Created: {manifest['created_at']}",
        f"Metric scope: {manifest['metric_scope']}",
        manifest["warning"],
        "",
        "Source columns per class / split:",
        ch1.groupby(["class_label", "split"]).size().unstack(fill_value=0).to_string(),
        "",
        "Source columns per class / speed / split:",
        ch1.groupby(["class_label", "speed_pct", "split"]).size().unstack(fill_value=0).to_string(),
    ]
    (args.out_dir / "split_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote {splits_path}")
    print(f"Wrote {args.out_dir / 'split_manifest.json'}")
    print(f"Scope: {scope}")
    if moved_columns:
        print(f"Legacy-only moved BPFO-3 @100% test columns: {moved_columns}")


if __name__ == "__main__":
    main()
