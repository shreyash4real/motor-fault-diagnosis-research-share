"""Validate source-level splits before feature generation or model training.

The project segments each 15-second recording into overlapping one-second
windows. Therefore, split integrity must be checked at the source-column level,
not at the segment level. This command also rejects a missing class/speed test
stratum by default. A deliberately bounded operating envelope is supported as
an explicit legacy/scoped mode, but it must be declared with the metric.

Usage
-----
    python validate_experiment.py --splits path/to/splits.csv
    python validate_experiment.py --splits path/to/splits.csv \
        --allow-missing-test-strata  # legacy ablation only; never headline it
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "full_path", "class_label", "speed_pct", "col_index", "split", "n_segments"
}
EXPECTED_SPLITS = {"train", "val", "test"}


def source_column_key(row: pd.Series) -> tuple[str, int]:
    """Return one stable identity for all three phase rows of a source column."""
    path = str(row["full_path"]).replace("\\", "/").lower()
    path = re.sub(r"-ch[123]\.csv$", "-ch1.csv", path)
    return path, int(row["col_index"])


def validate(splits: pd.DataFrame, allow_missing_test_strata: bool) -> list[str]:
    missing = REQUIRED_COLUMNS.difference(splits.columns)
    if missing:
        raise ValueError(f"splits.csv is missing required columns: {sorted(missing)}")

    found_splits = set(splits["split"].dropna().unique())
    if found_splits != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected exactly {sorted(EXPECTED_SPLITS)} splits, found {sorted(found_splits)}"
        )

    frame = splits.copy()
    frame["_source_column"] = frame.apply(source_column_key, axis=1)
    split_counts = frame.groupby("_source_column")["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        examples = ", ".join(map(str, leaked.index[:5]))
        raise ValueError(
            "Source-column leakage: a recording appears in multiple splits. "
            f"Examples: {examples}"
        )

    ch1 = frame[frame["full_path"].astype(str).str.contains(r"-ch1\.csv$", case=False, regex=True)]
    if ch1.empty:
        raise ValueError("No ch1 rows found; cannot audit source columns.")

    expected_strata = ch1.groupby(["class_label", "speed_pct"]).size().index
    observed_test = ch1[ch1["split"] == "test"].groupby(["class_label", "speed_pct"]).size()
    missing_test = [str(stratum) for stratum in expected_strata if stratum not in observed_test.index]
    warnings: list[str] = []
    if missing_test:
        message = "Missing test coverage for class/speed strata: " + ", ".join(missing_test)
        if allow_missing_test_strata:
            warnings.append("DECLARED BOUNDED SCOPE — " + message)
        else:
            raise ValueError(message)

    inconsistent_segments = ch1.groupby("_source_column")["n_segments"].nunique()
    if (inconsistent_segments > 1).any():
        raise ValueError("A source column has inconsistent n_segments values.")
    if (ch1["n_segments"] <= 0).any():
        raise ValueError("n_segments must be positive for every source column.")

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True, help="Path to canonical splits.csv")
    parser.add_argument(
        "--allow-missing-test-strata",
        action="store_true",
        help="Allow a deliberately scoped evaluation with missing class/speed test coverage.",
    )
    args = parser.parse_args()

    if not args.splits.is_file():
        raise SystemExit(f"splits.csv not found: {args.splits}")

    splits = pd.read_csv(args.splits)
    warnings = validate(splits, args.allow_missing_test_strata)

    ch1 = splits[splits["full_path"].astype(str).str.contains(r"-ch1\.csv$", case=False, regex=True)]
    by_split = ch1.groupby("split").size().to_dict()
    print(f"Split validation passed: {args.splits}")
    print(f"Source columns (ch1): {len(ch1)} | train={by_split.get('train', 0)} "
          f"val={by_split.get('val', 0)} test={by_split.get('test', 0)}")
    for warning in warnings:
        print(warning)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, pd.errors.ParserError) as error:
        print(f"Split validation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
