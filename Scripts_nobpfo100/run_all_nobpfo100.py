"""Run the reproducible motor-current pipeline with explicit data locations.

The filename is retained for backwards compatibility. A normal invocation
copies the supplied canonical split unchanged and validates every class/speed
stratum in test. The historical ``--legacy-exclude-bpfo100`` mode is available
only for a clearly declared, bounded operating envelope.

Example
-------
python run_all_nobpfo100.py \
  --canonical-splits /data/splits.csv \
  --raw-root /data/raw/Motor-2 \
  --denoised-root /data/denoised/Motor-2 \
  --out-dir /data/motor-hakeem-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-splits", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--denoised-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--save-pngs", action="store_true")
    parser.add_argument(
        "--legacy-exclude-bpfo100",
        action="store_true",
        help="Run the historical ablation with no BPFO-3 @100%% test coverage.",
    )
    return parser.parse_args()


def run_step(label: str, command: list[str], env: dict[str, str], log_file) -> None:
    banner = f"\n{'=' * 72}\n>>> {label}\n{'=' * 72}\n"
    print(banner, end="", flush=True)
    log_file.write(banner)
    log_file.flush()
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    log_file.write(result.stdout)
    log_file.flush()
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def main() -> None:
    args = parse_args()
    for name, path in {
        "canonical splits": args.canonical_splits,
        "raw root": args.raw_root,
        "denoised root": args.denoised_root,
    }.items():
        if not path.exists():
            raise SystemExit(f"{name} not found: {path}")

    out_dir = args.out_dir.resolve()
    training_dir = out_dir / "training"
    env = os.environ.copy()
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "MFDS_RAW_ROOT": str(args.raw_root.resolve()),
        "MFDS_DENOISED_ROOT": str(args.denoised_root.resolve()),
        "MFDS_OUTPUT_DIR": str(out_dir),
        "MFDS_TRAINING_DIR": str(training_dir),
        "MFDS_SPLITS_CSV": str(out_dir / "splits.csv"),
        "MFDS_STFT_FEATURES_DIR": str(out_dir / "features" / "stft"),
        "MFDS_DWT_FEATURES_DIR": str(out_dir / "features" / "dwt"),
        "MFDS_ENVELOPE_FEATURES_DIR": str(out_dir / "features" / "envelope"),
        "MFDS_SAVE_PNGS": "1" if args.save_pngs else "0",
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    generator = [
        sys.executable, str(SCRIPTS_DIR / "generate_splits_nobpfo100.py"),
        "--canonical-splits", str(args.canonical_splits.resolve()),
        "--out-dir", str(out_dir),
    ]
    validator = [
        sys.executable, str(SCRIPTS_DIR / "validate_experiment.py"),
        "--splits", str(out_dir / "splits.csv"),
    ]
    if args.legacy_exclude_bpfo100:
        generator.append("--legacy-exclude-bpfo100")
        validator.append("--allow-missing-test-strata")

    steps = [
        ("Generate split", generator),
        ("Validate split", validator),
        ("Precompute STFT features", [sys.executable, str(SCRIPTS_DIR / "precompute_stft_nobpfo100.py")]),
        ("Precompute DWT features", [sys.executable, str(SCRIPTS_DIR / "precompute_dwt_nobpfo100.py")]),
        ("Precompute envelope features", [sys.executable, str(SCRIPTS_DIR / "precompute_envelope_nobpfo100.py")]),
        ("Train STFT classifier", [sys.executable, str(SCRIPTS_DIR / "train_stft_cnn_nobpfo100.py")]),
        ("Train DWT classifier", [sys.executable, str(SCRIPTS_DIR / "train_dwt_cnn_nobpfo100.py")]),
        ("Train regularized envelope classifier", [sys.executable, str(SCRIPTS_DIR / "train_envelope_cnn_reg_nobpfo100.py")]),
        ("Train dilated envelope classifier", [sys.executable, str(SCRIPTS_DIR / "train_envelope_cnn_dilated_nobpfo100.py")]),
        ("Compute validation and test logits", [sys.executable, str(SCRIPTS_DIR / "compute_logits_nobpfo100.py")]),
        ("Evaluate temperature-calibrated ensembles", [sys.executable, str(SCRIPTS_DIR / "ensemble_evaluate_temperature_nobpfo100.py")]),
        ("Evaluate validation-F1-weighted ensembles", [sys.executable, str(SCRIPTS_DIR / "ensemble_evaluate_nobpfo100.py")]),
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Motor Hakeem pipeline started: {datetime.now(timezone.utc).isoformat()}\n")
        log_file.write(f"Scope: {'legacy_nobpfo100_ablation' if args.legacy_exclude_bpfo100 else 'canonical'}\n")
        for label, command in steps:
            run_step(label, command, env, log_file)

    print(f"Pipeline complete. Artefacts: {out_dir}")


if __name__ == "__main__":
    main()
