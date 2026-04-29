"""
ENSEMBLE EVALUATION v3 — PER-CLASS VAL-F1 WEIGHTED SOFT VOTING (raw probs)
=========================================================================
Pure ablation of *per-class val-F1 weighting* applied to raw (uncalibrated)
softmax. No temperature calibration.

Pipeline
--------
1. For each member, load val_logits + test_logits + labels (compute_logits.py).
2. Compute per-class val F1 = F1(argmax(softmax(val_logits)), val_labels).
3. Weight matrix: w[i, c] = val_F1[i, c] / Σⱼ val_F1[j, c]  (column-stochastic
   over members; one column per class).
4. For each ensemble: weighted soft vote on softmax(test_logits) with w
   restricted to that ensemble's members; renormalise per-sample to sum to 1.

Usage:
    python compute_logits.py                          # generates inputs
    python ensemble_evaluate_v3_perclass_f1.py
"""
from __future__ import annotations

import sys
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
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support)

SCRIPTS_4 = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_4))

from ensemble_evaluate_v1 import (
    MODEL_REGISTRY,
    ENSEMBLES,
    OUTPUT_BASE,
    CLASS_NAMES,
    perclass_val_f1,
    weights_perclass,
    weighted_soft_vote,
    write_outputs,
)
from ensemble_evaluate_v2_temperature import load_split_logits


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """CPU softmax of (N, C) float32 logits, returned as float64."""
    return (F.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1)
            .numpy().astype(np.float64))


def main():
    print("=" * 70)
    print("ENSEMBLE EVALUATION v3 — PER-CLASS VAL-F1 WEIGHTED SOFT VOTING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    base_dir = Path(OUTPUT_BASE)
    needed_keys = sorted({m for _, members in ENSEMBLES for m in members})
    print(f"Members required : {needed_keys}\n")

    # ── Load val + test logits/labels per member ────────────────────────────
    per_member = {}
    for key in needed_keys:
        v_logits, v_labels, v_sigs = load_split_logits(key, base_dir, "val")
        t_logits, t_labels, t_sigs = load_split_logits(key, base_dir, "test")
        per_member[key] = {
            "val_logits":  v_logits,  "val_labels":  v_labels,  "val_sigs":  v_sigs,
            "test_logits": t_logits,  "test_labels": t_labels,  "test_sigs": t_sigs,
        }

    # Alignment
    ref_key = needed_keys[0]
    ref_test_sigs   = per_member[ref_key]["test_sigs"]
    ref_test_labels = per_member[ref_key]["test_labels"]
    ref_val_sigs    = per_member[ref_key]["val_sigs"]
    ref_val_labels  = per_member[ref_key]["val_labels"]
    for key in needed_keys[1:]:
        assert per_member[key]["test_sigs"]  == ref_test_sigs, \
            f"Test metadata mismatch '{ref_key}' vs '{key}'."
        assert np.array_equal(per_member[key]["test_labels"], ref_test_labels), \
            f"Test label mismatch '{ref_key}' vs '{key}'."
        assert per_member[key]["val_sigs"]   == ref_val_sigs, \
            f"Val metadata mismatch '{ref_key}' vs '{key}'."
        assert np.array_equal(per_member[key]["val_labels"],  ref_val_labels), \
            f"Val label mismatch '{ref_key}' vs '{key}'."
    print(f"Alignment OK — val={len(ref_val_sigs)} test={len(ref_test_sigs)}\n")

    # ── Compute per-member raw test softmax + per-class val F1 ──────────────
    test_probs:        dict[str, np.ndarray] = {}
    val_f1_per_member: dict[str, np.ndarray] = {}
    print("Per-class val F1 per member:")
    print("  " + f"{'member':<10}  " + "  ".join(f"{c:<18}" for c in CLASS_NAMES))
    for key in needed_keys:
        m = per_member[key]
        v_probs = softmax_np(m["val_logits"])
        t_probs = softmax_np(m["test_logits"])
        f1 = perclass_val_f1(v_probs, m["val_labels"], len(CLASS_NAMES))
        val_f1_per_member[key] = f1
        test_probs[key]        = t_probs
        print(f"  {key:<10}  " + "  ".join(f"{x:<18.4f}" for x in f1)
              + f"   (macro={f1.mean():.4f})")
    print()

    # ── Build each ensemble (per-class val-F1 weighted) ─────────────────────
    leaderboard = []
    for run_name, members in ENSEMBLES:
        out_run_name = f"{run_name}_perclass_f1"
        out_dir = base_dir / out_run_name

        print("=" * 70)
        print(f"ENSEMBLE: {out_run_name}  (members={members})")
        print("=" * 70)

        out_dir.mkdir(parents=True, exist_ok=True)
        probs_stack  = np.stack([test_probs[m]        for m in members], axis=0)
        val_f1_stack = np.stack([val_f1_per_member[m] for m in members], axis=0)
        weights_mat  = weights_perclass(val_f1_stack)            # (M, C)
        mean_probs   = weighted_soft_vote(probs_stack, weights_mat)

        # Reproducibility artefacts
        for m in members:
            np.save(out_dir / f"softmax_{m}.npy", test_probs[m])
        np.save(out_dir / "labels.npy", ref_test_labels)
        pd.DataFrame(val_f1_stack, index=members, columns=CLASS_NAMES) \
            .to_csv(out_dir / "val_per_class_f1.csv")
        pd.DataFrame(weights_mat, index=members, columns=CLASS_NAMES) \
            .to_csv(out_dir / "weights.csv")

        strat_label = "soft voting (per-class val-F1 weighted, raw probs)"

        cfg_lines = [
            f"RUN NAME         : {out_run_name}",
            f"Created          : {datetime.now().isoformat(timespec='seconds')}",
            f"Strategy         : {strat_label}",
            f"Members          : {members}",
            "",
            "Member sources:",
        ]
        for m in members:
            cfg_lines.append(f"  {m:<10}  run={MODEL_REGISTRY[m]['run_name']}")
            cfg_lines.append(f"             ckpt={MODEL_REGISTRY[m]['checkpoint']}")
        cfg_lines.append("")
        cfg_lines.append("Per-class val F1 (rows=members, cols=classes):")
        cfg_lines.append("  " + f"{'member':<10}  " +
                         "  ".join(f"{c:<18}" for c in CLASS_NAMES))
        for m, row in zip(members, val_f1_stack):
            cfg_lines.append(f"  {m:<10}  " +
                             "  ".join(f"{x:<18.4f}" for x in row))
        cfg_lines.append("")
        cfg_lines.append("Member weights per class (column-stochastic, sum=1 over members):")
        cfg_lines.append("  " + f"{'member':<10}  " +
                         "  ".join(f"{c:<18}" for c in CLASS_NAMES))
        for m, row in zip(members, weights_mat):
            cfg_lines.append(f"  {m:<10}  " +
                             "  ".join(f"{x:<18.4f}" for x in row))
        (out_dir / "ensemble_config.txt").write_text(
            "\n".join(cfg_lines), encoding="utf-8")

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
    print("LEADERBOARD (v3 perclass_f1 only, raw probs)")
    print("=" * 70)
    rows = []
    for key in needed_keys:
        m = per_member[key]
        probs = test_probs[key]
        preds = probs.argmax(1)
        acc = accuracy_score(m["test_labels"], preds)
        p, r, f, sup = precision_recall_fscore_support(
            m["test_labels"], preds, labels=list(range(len(CLASS_NAMES))),
            zero_division=0)
        rows.append({
            "name":     f"(solo) {MODEL_REGISTRY[key]['run_name']}",
            "acc":      acc,
            "macro_f1": f.mean(),
            **{f"F1_{c}": f[i] for i, c in enumerate(CLASS_NAMES)},
        })
    for res in leaderboard:
        rows.append({
            "name":     res["run_name"],
            "acc":      res["acc"],
            "macro_f1": res["macro_f1"],
            **{f"F1_{c}": v for c, v in res["per_class_f1"].items()},
        })
    lb_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(lb_df.to_string(index=False))

    lb_path = base_dir / "ensemble_leaderboard_perclass_f1.csv"
    lb_df.to_csv(lb_path, index=False)
    print(f"\nLeaderboard written -> {lb_path}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
