"""
ENSEMBLE EVALUATION v2 — TEMPERATURE-CALIBRATED EQUAL-WEIGHT SOFT VOTING
========================================================================
Pure ablation of *post-hoc temperature calibration* applied to each
ensemble member, then equal-weight soft voting. No per-class F1 weighting
in this script — that's a separate ablation (v3).

Pipeline
--------
1. For each member required by ENSEMBLES, load val_logits.npy + val_labels.npy
   (written by compute_logits.py).
2. Fit one scalar T_i per member by minimizing
       L(T) = CrossEntropy(val_logits / T, val_labels)
   with LBFGS. T preserves argmax (solo accuracy unchanged) but rescales
   confidence: T>1 softens, T<1 sharpens.
3. Apply T_i to each member's *test* logits, softmax → calibrated test
   probabilities.
4. For each ensemble combination, equal-weight average the calibrated
   probabilities and compute test metrics. Write artefacts to
   ENSEMBLE_<members>_temperature.

Why this script reuses v1
-------------------------
MODEL_REGISTRY, ENSEMBLES, build_model, load_split_data, write_outputs are
imported from ensemble_evaluate_v1. Only the calibration + voting logic is
new here.

Usage
-----
    # 1) Generate the logits files first (idempotent):
    python compute_logits.py
    # 2) Run this script:
    python ensemble_evaluate_v2_temperature.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
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
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)

SCRIPTS_4 = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_4))

from ensemble_evaluate_v1_nobpfo100 import (
    MODEL_REGISTRY,
    ENSEMBLES,
    OUTPUT_BASE,
    CLASS_NAMES,
    write_outputs,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Loaders (val + test logits/labels/sigs from each member's run folder)
# ──────────────────────────────────────────────────────────────────────────────

def load_split_logits(member_key: str, base_dir: Path, split: str
                      ) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    cfg = MODEL_REGISTRY[member_key]
    run_dir = base_dir / cfg["run_name"]
    lg_path = run_dir / f"{split}_logits.npy"
    lb_path = run_dir / f"{split}_labels.npy"
    sg_path = run_dir / f"{split}_meta_sigs.csv"
    if not (lg_path.exists() and lb_path.exists() and sg_path.exists()):
        raise FileNotFoundError(
            f"Missing {split} logits for member '{member_key}' under {run_dir}.\n"
            f"Run: python compute_logits.py")
    logits = np.load(lg_path)
    labels = np.load(lb_path)
    sigs_df = pd.read_csv(sg_path)
    sigs = [(str(r["ch1_path"]), int(r["col_index"]), int(r["seg_idx"]))
            for _, r in sigs_df.iterrows()]
    return logits, labels, sigs


# ──────────────────────────────────────────────────────────────────────────────
#  Temperature calibration: fit one scalar T per member
# ──────────────────────────────────────────────────────────────────────────────

def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray,
                    device: torch.device,
                    max_iter: int = 50, lr: float = 0.01) -> float:
    """
    Fit a single scalar T > 0 by minimizing
        CE(val_logits / T, val_labels)
    with LBFGS. Return the optimized T as a python float.

    We optimize log_T (so T = exp(log_T) stays positive without constraints).
    """
    logits_t = torch.from_numpy(val_logits.astype(np.float32)).to(device)
    labels_t = torch.from_numpy(val_labels.astype(np.int64)).to(device)

    log_T = torch.zeros(1, device=device, requires_grad=True)   # T = 1.0 init
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_final = float(torch.exp(log_T).item())
    return T_final


def temperature_softmax(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply scalar T then softmax. Returns (N, C) float64 for stable averaging."""
    z = torch.from_numpy(logits.astype(np.float32)) / T
    return F.softmax(z, dim=1).numpy().astype(np.float64)


def evaluate_solo(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    preds = probs.argmax(1)
    return accuracy_score(labels, preds), f1_score(
        labels, preds, average="macro", zero_division=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ENSEMBLE EVALUATION v2 — TEMPERATURE-CALIBRATED EQUAL-WEIGHT VOTING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print()

    base_dir = Path(OUTPUT_BASE)
    needed_keys = sorted({m for _, members in ENSEMBLES for m in members})
    print(f"Members required : {needed_keys}\n")

    # ── Load val + test logits/labels per member ───────────────────────────
    per_member = {}
    for key in needed_keys:
        v_logits, v_labels, v_sigs = load_split_logits(key, base_dir, "val")
        t_logits, t_labels, t_sigs = load_split_logits(key, base_dir, "test")

        # Quick sanity: pre-calibration solo metrics
        v_acc, v_f1 = evaluate_solo(
            F.softmax(torch.from_numpy(v_logits), dim=1).numpy(), v_labels)
        t_acc, t_f1 = evaluate_solo(
            F.softmax(torch.from_numpy(t_logits), dim=1).numpy(), t_labels)
        print(f"-- [{key}]  val acc={v_acc:.4f} f1={v_f1:.4f}   "
              f"test acc={t_acc:.4f} f1={t_f1:.4f}  (raw, T=1)")

        per_member[key] = {
            "val_logits":  v_logits,  "val_labels":  v_labels,  "val_sigs":  v_sigs,
            "test_logits": t_logits,  "test_labels": t_labels,  "test_sigs": t_sigs,
        }
    print()

    # ── Verify alignment of test/val metadata across members ───────────────
    ref_key = needed_keys[0]
    ref_test_sigs   = per_member[ref_key]["test_sigs"]
    ref_test_labels = per_member[ref_key]["test_labels"]
    ref_val_sigs    = per_member[ref_key]["val_sigs"]
    ref_val_labels  = per_member[ref_key]["val_labels"]
    for key in needed_keys[1:]:
        assert per_member[key]["test_sigs"] == ref_test_sigs, \
            f"Test metadata order mismatch between '{ref_key}' and '{key}'."
        assert np.array_equal(per_member[key]["test_labels"], ref_test_labels), \
            f"Test label order mismatch between '{ref_key}' and '{key}'."
        assert per_member[key]["val_sigs"] == ref_val_sigs, \
            f"Val metadata order mismatch between '{ref_key}' and '{key}'."
        assert np.array_equal(per_member[key]["val_labels"], ref_val_labels), \
            f"Val label order mismatch between '{ref_key}' and '{key}'."
    print(f"Alignment OK  —  val={len(ref_val_sigs)}  test={len(ref_test_sigs)}")
    print()

    # ── Fit per-member temperatures on val NLL ──────────────────────────────
    temperatures: dict[str, float] = {}
    print("Fitting per-member temperature (LBFGS on val NLL):")
    for key in needed_keys:
        m = per_member[key]
        t0 = time.time()
        T = fit_temperature(m["val_logits"], m["val_labels"], device)
        elapsed = time.time() - t0

        # Pre/post NLL for diagnostic
        with torch.no_grad():
            lg = torch.from_numpy(m["val_logits"].astype(np.float32)).to(device)
            lb = torch.from_numpy(m["val_labels"].astype(np.int64)).to(device)
            nll_pre  = float(F.cross_entropy(lg, lb).item())
            nll_post = float(F.cross_entropy(lg / T, lb).item())

        # Save T for this member
        run_dir = base_dir / MODEL_REGISTRY[key]["run_name"]
        (run_dir / "temperature.txt").write_text(
            f"{T:.6f}\n", encoding="utf-8")

        temperatures[key] = T
        direction = "softens (over-confident)" if T > 1 else (
            "sharpens (under-confident)" if T < 1 else "unchanged")
        print(f"  {key:<10}  T = {T:.4f}   val NLL  {nll_pre:.4f} -> "
              f"{nll_post:.4f}   ({direction})   ({elapsed:.2f}s)")
    print()

    # ── Calibrate test softmaxes (T applied to test logits) ─────────────────
    cal_test_probs: dict[str, np.ndarray] = {}
    for key in needed_keys:
        m = per_member[key]
        T = temperatures[key]
        cal_test_probs[key] = temperature_softmax(m["test_logits"], T)

        # Solo metrics post-calibration (argmax invariant — sanity check)
        acc, f1 = evaluate_solo(cal_test_probs[key], m["test_labels"])
        print(f"  [{key}] solo test post-calibration   acc={acc:.4f}  "
              f"f1={f1:.4f}   (argmax invariant under T scaling — sanity)")
    print()

    # ── Build each ensemble (equal-weight soft voting on calibrated probs) ──
    leaderboard = []
    for run_name, members in ENSEMBLES:
        out_run_name = f"{run_name}_temperature"
        out_dir = base_dir / out_run_name

        print("=" * 70)
        print(f"ENSEMBLE: {out_run_name}  (members={members}, "
              f"strategy=equal-weight on calibrated probs)")
        print("=" * 70)

        out_dir.mkdir(parents=True, exist_ok=True)

        probs_stack = np.stack([cal_test_probs[m] for m in members], axis=0)
        mean_probs  = probs_stack.mean(axis=0)

        # Persist per-member calibrated test softmax + labels for reproducibility
        for m in members:
            np.save(out_dir / f"softmax_calibrated_{m}.npy", cal_test_probs[m])
        np.save(out_dir / "labels.npy", ref_test_labels)

        # Save the temperatures used
        pd.DataFrame({
            "member": members,
            "T":      [temperatures[m] for m in members],
        }).to_csv(out_dir / "temperatures.csv", index=False)

        strat_label = ("soft voting (equal-weight mean of "
                       "temperature-calibrated softmaxes)")

        cfg_lines = [
            f"RUN NAME         : {out_run_name}",
            f"Created          : {datetime.now().isoformat(timespec='seconds')}",
            f"Strategy         : {strat_label}",
            f"Members          : {members}",
            "",
            "Per-member temperature (fit on val NLL, LBFGS):",
        ]
        for m in members:
            cfg_lines.append(f"  {m:<10}  T = {temperatures[m]:.4f}")
        cfg_lines.append("")
        cfg_lines.append("Member sources:")
        for m in members:
            cfg_lines.append(f"  {m:<10}  run={MODEL_REGISTRY[m]['run_name']}")
            cfg_lines.append(f"             ckpt={MODEL_REGISTRY[m]['checkpoint']}")
        (out_dir / "ensemble_config.txt").write_text(
            "\n".join(cfg_lines), encoding="utf-8")

        # Need test metadata for write_outputs — reconstruct from sigs
        ref_test_md = [{"ch1_path": s[0], "col_index": s[1], "seg_idx": s[2]}
                        for s in ref_test_sigs]

        result = write_outputs(
            run_name=out_run_name, members=members,
            mean_probs=mean_probs,
            labels=ref_test_labels,
            metadata=ref_test_md,
            class_names=CLASS_NAMES,
            out_dir=out_dir,
            strategy_label=strat_label,
        )
        leaderboard.append(result)

        print(f"  acc      = {result['acc']:.4f}")
        print(f"  macro_f1 = {result['macro_f1']:.4f}")
        print(f"  wrong    = {result['n_wrong']} / {len(ref_test_labels)}")
        for k, v in result["per_class_f1"].items():
            print(f"    {k:<22} F1={v:.4f}")
        print(f"  outputs -> {out_dir}")
        print()

    # ── Leaderboard ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("LEADERBOARD  (calibrated solos + temperature-calibrated ensembles)")
    print("=" * 70)
    rows = []
    for key in needed_keys:
        m = per_member[key]
        probs = cal_test_probs[key]
        preds = probs.argmax(1)
        acc = accuracy_score(m["test_labels"], preds)
        p, r, f, sup = precision_recall_fscore_support(
            m["test_labels"], preds, labels=list(range(len(CLASS_NAMES))),
            zero_division=0)
        macro_f1 = f.mean()
        rows.append({
            "name":     f"(solo, T={temperatures[key]:.3f}) "
                        f"{MODEL_REGISTRY[key]['run_name']}",
            "T":        temperatures[key],
            "acc":      acc,
            "macro_f1": macro_f1,
            **{f"F1_{c}": f[i] for i, c in enumerate(CLASS_NAMES)},
        })
    for res in leaderboard:
        rows.append({
            "name":     res["run_name"],
            "T":        np.nan,
            "acc":      res["acc"],
            "macro_f1": res["macro_f1"],
            **{f"F1_{c}": v for c, v in res["per_class_f1"].items()},
        })
    lb_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(lb_df.to_string(index=False))

    lb_path = base_dir / "ensemble_leaderboard_temperature.csv"
    lb_df.to_csv(lb_path, index=False)
    print(f"\nLeaderboard written -> {lb_path}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
