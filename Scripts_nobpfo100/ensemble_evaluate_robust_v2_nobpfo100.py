"""
ENSEMBLE EVALUATION — ROBUST STFT + DWT + ENVELOPE v2 (nobpfo100)
=================================================================
Evaluates a new ensemble combination for the nobpfo100 perturbation experiment,
using the un-regularized Envelope v2 model instead of v3, based on our 
ablation findings that v2 was more resilient to the distribution shift.

MEMBERS:
  1. STFT:     STFT_robust_resnet (New robust 2D ResNet on 1000Hz cropped STFT)
  2. DWT:      DWT_multibranch_plain_v1 (1D native multibranch)
  3. ENVELOPE: ENVELOPE_resnet_v2 (1D ResNet v2 without heavy regularization)

STRATEGY:
  Temperature-calibrated equal-weight soft voting. 
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Import helpers from v1
SCRIPTS_4 = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_4))
from ensemble_evaluate_v1_nobpfo100 import write_outputs, CLASS_NAMES

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_BASE = Path(r"C:\Project Work\Outputs_nobpfo100\training")
ENSEMBLE_NAME = "ENSEMBLE_robust_stft_dwt_envelope_v2_temperature"

MEMBERS = {
    "stft_robust": "STFT_robust_resnet",
    "dwt":         "DWT_multibranch_plain_v1",
    "envelope_v2": "ENVELOPE_resnet_v2"
}

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_split_logits(run_name: str, split: str) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    run_dir = OUTPUT_BASE / run_name
    lg_path = run_dir / f"{split}_logits.npy"
    lb_path = run_dir / f"{split}_labels.npy"
    sg_path = run_dir / f"{split}_meta_sigs.csv"
    
    if not (lg_path.exists() and lb_path.exists() and sg_path.exists()):
        # Try to fallback to computing logits if missing, but normally we expect them
        raise FileNotFoundError(f"Missing {split} outputs in {run_dir}. Need to run compute_logits or ensure training script exports them.")
        
    logits = np.load(lg_path)
    labels = np.load(lb_path)
    sigs_df = pd.read_csv(sg_path)
    sigs = [(str(r["ch1_path"]), int(r["col_index"]), int(r["seg_idx"])) for _, r in sigs_df.iterrows()]
    
    return logits, labels, sigs

def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray, device: torch.device) -> float:
    """Fit scalar T by minimizing Cross Entropy on validation logits using LBFGS."""
    logits_t = torch.from_numpy(val_logits.astype(np.float32)).to(device)
    labels_t = torch.from_numpy(val_labels.astype(np.int64)).to(device)
    
    log_T = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=50, line_search_fn="strong_wolfe")
    
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_t / torch.exp(log_T), labels_t)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    return float(torch.exp(log_T).item())

def temperature_softmax(logits: np.ndarray, T: float) -> np.ndarray:
    z = torch.from_numpy(logits.astype(np.float32)) / T
    return F.softmax(z, dim=1).numpy().astype(np.float64)

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ROBUST ENSEMBLE EVALUATION w/ v2 (nobpfo100)")
    print("=" * 70)
    
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if we need to compute logits for ENVELOPE_resnet_v2 because its training script didn't export them natively
    env_v2_run_dir = OUTPUT_BASE / "ENVELOPE_resnet_v2"
    if not (env_v2_run_dir / "val_logits.npy").exists():
        print("Envelope v2 logits missing. The training script we used didn't natively export them.")
        print("We need to run compute_logits_nobpfo100.py first. Please hold.")
        sys.exit(1)

    per_member = {}
    print("Loading artifacts and evaluating solo performance...")
    for key, run_name in MEMBERS.items():
        v_lg, v_lb, v_sg = load_split_logits(run_name, "val")
        t_lg, t_lb, t_sg = load_split_logits(run_name, "test")
        
        v_acc = accuracy_score(v_lb, v_lg.argmax(1))
        v_f1  = f1_score(v_lb, v_lg.argmax(1), average="macro", zero_division=0)
        t_acc = accuracy_score(t_lb, t_lg.argmax(1))
        t_f1  = f1_score(t_lb, t_lg.argmax(1), average="macro", zero_division=0)
        
        print(f"  [{key}] {run_name}")
        print(f"    Val:  Acc={v_acc:.4f}  F1={v_f1:.4f}")
        print(f"    Test: Acc={t_acc:.4f}  F1={t_f1:.4f}")
        
        per_member[key] = {
            "val_logits": v_lg, "val_labels": v_lb, "val_sigs": v_sg,
            "test_logits": t_lg, "test_labels": t_lb, "test_sigs": t_sg
        }
    print()
    
    # ── Alignment Check ──
    ref_key = list(MEMBERS.keys())[0]
    ref_t_sg, ref_t_lb = per_member[ref_key]["test_sigs"], per_member[ref_key]["test_labels"]
    ref_v_sg, ref_v_lb = per_member[ref_key]["val_sigs"], per_member[ref_key]["val_labels"]
    for key in list(MEMBERS.keys())[1:]:
        assert per_member[key]["test_sigs"] == ref_t_sg, f"Test sig mismatch: {key}"
        assert np.array_equal(per_member[key]["test_labels"], ref_t_lb), f"Test label mismatch: {key}"
    print(f"Data Alignment OK (Val={len(ref_v_lb)}, Test={len(ref_t_lb)})\n")
    
    # ── Temperature Calibration ──
    temperatures = {}
    cal_test_probs = {}
    print("Calibrating Temperature via LBFGS on Val NLL...")
    for key in MEMBERS:
        m = per_member[key]
        T = fit_temperature(m["val_logits"], m["val_labels"], device)
        temperatures[key] = T
        cal_test_probs[key] = temperature_softmax(m["test_logits"], T)
        print(f"  {key:<15} T = {T:.4f}")
    print()
    
    # ── Ensemble Voting ──
    out_dir = OUTPUT_BASE / ENSEMBLE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    
    probs_stack = np.stack([cal_test_probs[k] for k in MEMBERS.keys()], axis=0)
    mean_probs = probs_stack.mean(axis=0)
    
    for key in MEMBERS.keys():
        np.save(out_dir / f"softmax_calibrated_{key}.npy", cal_test_probs[key])
    np.save(out_dir / "labels.npy", ref_t_lb)
    pd.DataFrame({"member": list(MEMBERS.keys()), "T": [temperatures[k] for k in MEMBERS.keys()]}).to_csv(out_dir / "temperatures.csv", index=False)
    
    strat_label = "soft voting (equal-weight mean of temperature-calibrated softmaxes)"
    ref_md = [{"ch1_path": s[0], "col_index": s[1], "seg_idx": s[2]} for s in ref_t_sg]
    
    result = write_outputs(
        run_name=ENSEMBLE_NAME, members=list(MEMBERS.keys()),
        mean_probs=mean_probs, labels=ref_t_lb, metadata=ref_md,
        class_names=CLASS_NAMES, out_dir=out_dir, strategy_label=strat_label
    )
    
    print("=" * 70)
    print(f"ENSEMBLE RESULTS: {ENSEMBLE_NAME}")
    print("=" * 70)
    print(f"  Test Accuracy: {result['acc']:.4f}")
    print(f"  Test Macro-F1: {result['macro_f1']:.4f}")
    print(f"  Misclassified: {result['n_wrong']} / {len(ref_t_lb)}")
    for c, v in result['per_class_f1'].items():
        print(f"    {c:<20} F1 = {v:.4f}")
    print(f"\nOutputs saved to: {out_dir}")

if __name__ == "__main__":
    main()