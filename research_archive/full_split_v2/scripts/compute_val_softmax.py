"""
COMPUTE VAL SOFTMAX (one-time helper)
======================================
For each member in MODEL_REGISTRY (defined in ensemble_evaluate_v1.py), load
its best_model.pt and val.pt features, run inference, and save:
  <run_folder>/val_softmax.npy    shape (N_val, 4) float32
  <run_folder>/val_labels.npy     shape (N_val,)   int64
  <run_folder>/val_meta_sigs.csv  per-sample (ch1_path, col_index, seg_idx)

Idempotent: members whose three files already exist are skipped unless --force.
The ensemble script reads these files via load_val_outputs().

Usage:
    python compute_val_softmax.py
    python compute_val_softmax.py --force
    python compute_val_softmax.py --only stft dwt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pipeline_paths import legacy_path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

SCRIPTS_4 = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_4))

from ensemble_evaluate_v1 import (
    MODEL_REGISTRY,
    OUTPUT_BASE,
    USE_AMP,
    build_model,
    infer_softmax,
    load_split_data,
    _sig,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-compute even if val_softmax.npy exists")
    ap.add_argument("--only", nargs="*",
                    help="restrict to these member keys (e.g. --only stft dwt)")
    args = ap.parse_args()

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
        out_softmax = run_dir / "val_softmax.npy"
        out_labels  = run_dir / "val_labels.npy"
        out_sigs    = run_dir / "val_meta_sigs.csv"

        if (not args.force and out_softmax.exists()
                and out_labels.exists() and out_sigs.exists()):
            print(f"-- [{key}] SKIP   (files exist; use --force to overwrite)")
            continue

        print(f"-- [{key}] run={cfg['run_name']} kind={cfg['kind']}")
        print(f"   features : {cfg['features_dir']}")
        print(f"   ckpt     : {cfg['checkpoint']}")

        t0 = time.time()
        val_data = load_split_data(cfg["kind"], Path(cfg["features_dir"]),
                                   split="val")
        ckpt = torch.load(cfg["checkpoint"], map_location=device,
                          weights_only=False)
        model = build_model(cfg["kind"], ckpt, val_data)

        probs = infer_softmax(cfg["kind"], model, val_data, device,
                              use_amp_effective)
        labels = val_data["labels"].numpy()

        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_softmax, probs.astype(np.float32))
        np.save(out_labels,  labels.astype(np.int64))

        sig_rows = [
            {"ch1_path":  m["ch1_path"],
             "col_index": int(m["col_index"]),
             "seg_idx":   int(m["seg_idx"])}
            for m in val_data["metadata"]
        ]
        pd.DataFrame(sig_rows).to_csv(out_sigs, index=False)

        elapsed = time.time() - t0
        preds = probs.argmax(1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        ckpt_val_f1 = ckpt.get("val_f1", "n/a")
        print(f"   wrote {probs.shape} softmax + {labels.shape} labels   "
              f"({elapsed:.1f}s)")
        print(f"   val acc={acc:.4f}  macro_f1={macro_f1:.4f}  "
              f"(ckpt val_f1={ckpt_val_f1})")
        print()

        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("DONE")


if __name__ == "__main__":
    main()
