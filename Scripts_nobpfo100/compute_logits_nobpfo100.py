"""
COMPUTE LOGITS (one-time helper for temperature calibration)
============================================================
For each member in MODEL_REGISTRY (defined in ensemble_evaluate_v1.py),
load best_model.pt and run inference on the val + test splits, saving
the *pre-softmax logits* (and labels and metadata signatures) per split.

Outputs per member's run folder:
  val_logits.npy        shape (N_val,  4) float32  — raw model outputs
  test_logits.npy       shape (N_test, 4) float32
  val_labels.npy        shape (N_val,)    int64
  test_labels.npy       shape (N_test,)   int64
  val_meta_sigs.csv     per-sample (ch1_path, col_index, seg_idx)
  test_meta_sigs.csv    same, for test split

Idempotent: members whose logits files already exist are skipped unless --force.

Why this script exists
----------------------
Temperature calibration fits a scalar T per member by minimizing
val cross-entropy on the *logits* (not the softmax). It then applies
T to *test* logits before re-softmaxing for ensembling. We can't
recover logits from saved softmax (especially under AMP), so we re-run
inference here and persist the logits directly.

Usage
-----
    python compute_logits.py
    python compute_logits.py --force
    python compute_logits.py --only stft dwt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPTS_4 = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_4))

from ensemble_evaluate_v1_nobpfo100 import (
    MODEL_REGISTRY,
    OUTPUT_BASE,
    USE_AMP,
    BATCH,
    build_model,
    load_split_data,
)


@torch.no_grad()
def infer_logits(kind: str, model, data: dict, device: torch.device,
                 use_amp: bool) -> np.ndarray:
    """Mirror of infer_softmax but returns raw (N, C) float32 logits."""
    model.eval().to(device)

    labels = data["labels"]
    n = int(labels.shape[0])
    all_logits = []

    if kind == "dwt":
        feats_per_level = data["features_per_level"]
        for i in range(0, n, BATCH):
            bands = tuple(
                feats_per_level[j][i:i + BATCH].to(device, non_blocking=True)
                for j in range(len(feats_per_level))
            )
            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(bands)
            else:
                logits = model(bands)
            all_logits.append(logits.float().cpu().numpy())
    else:
        feats = data["features"]
        for i in range(0, n, BATCH):
            x = feats[i:i + BATCH].to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(x)
            else:
                logits = model(x)
            all_logits.append(logits.float().cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def write_meta_sigs(metadata: list, out_path: Path) -> None:
    rows = [
        {"ch1_path":  m["ch1_path"],
         "col_index": int(m["col_index"]),
         "seg_idx":   int(m["seg_idx"])}
        for m in metadata
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def process_split(key: str, kind: str, features_dir: Path, split: str,
                  model, device, use_amp_eff: bool, run_dir: Path,
                  force: bool) -> None:
    out_logits = run_dir / f"{split}_logits.npy"
    out_labels = run_dir / f"{split}_labels.npy"
    out_sigs   = run_dir / f"{split}_meta_sigs.csv"

    if (not force and out_logits.exists() and out_labels.exists()
            and out_sigs.exists()):
        print(f"   [{split}] SKIP   (files exist)")
        return

    t0 = time.time()
    data = load_split_data(kind, features_dir, split=split)
    logits = infer_logits(kind, model, data, device, use_amp_eff)
    labels = data["labels"].numpy()

    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_logits, logits.astype(np.float32))
    np.save(out_labels, labels.astype(np.int64))
    write_meta_sigs(data["metadata"], out_sigs)

    preds = logits.argmax(1)
    acc = float((preds == labels).mean())
    elapsed = time.time() - t0
    print(f"   [{split}] wrote {logits.shape} logits   acc={acc:.4f}   "
          f"({elapsed:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-compute even if files exist")
    ap.add_argument("--only", nargs="*",
                    help="restrict to these member keys (e.g. --only stft dwt)")
    args, _ = ap.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp_effective = USE_AMP and device.type == "cuda"
    print(f"Device: {device}   AMP: {use_amp_effective}")
    print()

    keys = list(MODEL_REGISTRY.keys())
    if args.only:
        keys = [k for k in keys if k in args.only]
        if not keys:
            print(f"No members matched --only {args.only}")
            return

    out_base = Path(OUTPUT_BASE)

    for key in keys:
        cfg = MODEL_REGISTRY[key]
        run_dir = out_base / cfg["run_name"]
        print(f"-- [{key}] run={cfg['run_name']} kind={cfg['kind']}")
        print(f"   features : {cfg['features_dir']}")
        print(f"   ckpt     : {cfg['checkpoint']}")

        # We'll build the model once if either split needs work.
        need_val  = args.force or not (
            (run_dir / "val_logits.npy").exists()
            and (run_dir / "val_labels.npy").exists()
            and (run_dir / "val_meta_sigs.csv").exists())
        need_test = args.force or not (
            (run_dir / "test_logits.npy").exists()
            and (run_dir / "test_labels.npy").exists()
            and (run_dir / "test_meta_sigs.csv").exists())

        if not (need_val or need_test):
            print("   SKIP all splits (use --force to overwrite)\n")
            continue

        # Build model from the test features dict (works for val too — same dim)
        # Use whichever split we need first to drive the build call.
        bootstrap_split = "val" if need_val else "test"
        boot_data = load_split_data(cfg["kind"], Path(cfg["features_dir"]),
                                    split=bootstrap_split)
        ckpt = torch.load(cfg["checkpoint"], map_location=device,
                          weights_only=False)
        model = build_model(cfg["kind"], ckpt, boot_data)
        del boot_data

        if need_val:
            process_split(key, cfg["kind"], Path(cfg["features_dir"]), "val",
                          model, device, use_amp_effective, run_dir,
                          args.force)
        if need_test:
            process_split(key, cfg["kind"], Path(cfg["features_dir"]), "test",
                          model, device, use_amp_effective, run_dir,
                          args.force)

        print()
        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("DONE")


if __name__ == "__main__":
    main()
