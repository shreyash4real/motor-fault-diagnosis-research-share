"""
STEP 6 — ENSEMBLE EVALUATION (4-class)
======================================
Soft-voting ensemble of previously trained single-representation models.

WHAT IT DOES
------------
1. Loads each requested model's saved best_model.pt plus its matching
   test.pt features.
2. Runs inference once per model -> (N_test, 4) softmax probability tensor.
3. Checks that sample ordering (ch1_path, col_index, seg_idx, label)
   matches across all representations so the softmaxes align.
4. For each requested ENSEMBLE combination, averages the member softmaxes,
   argmaxes, and writes the usual artefacts:
     summary.txt, per_class_metrics.csv, confusion_matrix.png,
     misclassified_samples.csv, ensemble_config.txt, softmax_<member>.npy.

MODEL REGISTRY
--------------
  stft              E4_mod_alexnet_227           STFT     ModifiedAlexNet (3ch, 227x227)
  dwt               DWT_multibranch_plain_v1     DWT      DWTMultiBranchCNN (plain)
  clarke            CLARKE_mod_alexnet_v1        Clarke   ModifiedAlexNet (1ch, 128x128)
  envelope          ENVELOPE_resnet_v3_reg       Envelope EnvelopeResNet1D (regularized v3, 940k)
  envelope_dilated  ENVELOPE_dilated1d_v1        Envelope EnvelopeDilated1D — legacy, archival only

Model classes are imported from their respective train_*.py files. The
two ModifiedAlexNet classes (STFT and Clarke) have identical names but
different configurations; we import them with aliases to keep them
separate.

Usage
-----
    Edit the ENSEMBLES / MODEL_REGISTRY blocks, then:
    python ensemble_evaluate_v1.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout so unicode chars in prints survive piping through `tee`
# on Windows (default stdout encoding under a pipe is cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              f1_score, accuracy_score)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── import sibling trainers so we can reuse their nn.Module classes ─────────
SCRIPTS_4 = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_4))

# STFT and Clarke both define a class called ModifiedAlexNet. Different
# configurations; import with aliases.
from train_stft_cnn_nobpfo100 import ModifiedAlexNet as StftAlexNet
from train_dwt_cnn_nobpfo100 import build_model as build_dwt_model
from train_envelope_cnn_dilated_nobpfo100 import EnvelopeDilated1D
from train_envelope_cnn_reg_nobpfo100 import EnvelopeResNet1D as EnvelopeResNetV3

# ═══════════════════════════════════════════════════════════════════════════
#  ENSEMBLE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

# Each entry: (ensemble_run_name, list_of_member_keys_from_MODEL_REGISTRY)
ENSEMBLES = [
    ("ENSEMBLE_stft_dwt",          ["stft", "dwt"]),
    ("ENSEMBLE_stft_dwt_envelope", ["stft", "dwt", "envelope_dilated"]),
    ("ENSEMBLE_stft_dwt_envelope_v3", ["stft", "dwt", "envelope"]),
]



SEED      = 42
USE_AMP   = True
USE_MMAP  = True
BATCH     = 64

CLASS_NAMES = ["healthy 1", "stator short 1", "bearing bpfo 3", "broken rotor bar"]

# ─── model registry ─────────────────────────────────────────────────────────
# Each model entry provides:
#   features_dir : where test.pt lives
#   checkpoint   : best_model.pt path
#   kind         : "stft" | "dwt" | "clarke" | "envelope"
#                  (determines how to load features and run inference)


OUTPUT_BASE = r"C:\Project Work\Outputs_nobpfo100\training"
BASE_V2     = r"C:\Project Work\Outputs_nobpfo100"


MODEL_REGISTRY = {
    "stft": {
        "run_name":     "E4_mod_alexnet_227",
        "features_dir": fr"{BASE_V2}\features\stft",
        "checkpoint":   fr"{OUTPUT_BASE}\E4_mod_alexnet_227\best_model.pt",
        "kind":         "stft",
    },
    "dwt": {
        "run_name":     "DWT_multibranch_plain_v1",
        "features_dir": fr"{BASE_V2}\features\dwt",
        "checkpoint":   fr"{OUTPUT_BASE}\DWT_multibranch_plain_v1\best_model.pt",
        "kind":         "dwt",
    },
    "envelope": {
        # Canonical envelope member — regularized 940k ResNet1D (v3).
        # Solo bpfo-3 F1 = 0.823, the highest single-branch result in the project.
        "run_name":     "ENVELOPE_resnet_v3_reg",
        "features_dir": fr"{BASE_V2}\features\envelope",
        "checkpoint":   fr"{OUTPUT_BASE}\ENVELOPE_resnet_v3_reg\best_model.pt",
        "kind":         "envelope_v3",
    },
    "envelope_dilated": {
        # Archived — original dilated 1D envelope CNN. Kept reachable so future
        # comparison ensembles can include it; not in default ENSEMBLES.
        "run_name":     "ENVELOPE_dilated1d_v1",
        "features_dir": fr"{BASE_V2}\features\envelope",
        "checkpoint":   fr"{OUTPUT_BASE}\ENVELOPE_dilated1d_v1\best_model.pt",
        "kind":         "envelope",
    },
}
# ═══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers: load test features + metadata
# ──────────────────────────────────────────────────────────────────────────────

def _sig(md_entry: dict) -> tuple:
    """A per-sample identity tuple used to check alignment across reps."""
    return (md_entry["ch1_path"], int(md_entry["col_index"]),
            int(md_entry["seg_idx"]))


def load_split_data(kind: str, features_dir: Path,
                    split: str = "test") -> dict:
    """Returns dict with 'features', 'labels', 'metadata', 'class_names'.

    For DWT the 'features' field is a list of per-level tensors. `split` is
    one of "train" / "val" / "test" and selects <split>.pt under features_dir.
    """
    path = features_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    data = torch.load(path, weights_only=False, mmap=USE_MMAP)

    out = {
        "labels":      data["labels"],
        "metadata":    data["metadata"],
        "class_names": data["class_names"],
    }
    if kind == "dwt":
        out["features_per_level"] = data["features_per_level"]
        out["level_lengths"]      = data["level_lengths"]
    else:
        out["features"] = data["features"]
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers: rebuild models from checkpoints
# ──────────────────────────────────────────────────────────────────────────────

def build_model(kind: str, ckpt: dict, test_data: dict):
    """Reconstruct the nn.Module for a given representation and load state."""
    if kind == "stft":
        model = StftAlexNet(num_classes=len(CLASS_NAMES), dropout=0.5)
    elif kind == "clarke":
        # Clarke: 1-channel native 128x128, channel widths from the ckpt
        channels = (32, 64, 96, 96, 64)   # matches CLARKE_mod_alexnet_v1 config
        model = ClarkeAlexNet(num_classes=len(CLASS_NAMES), dropout=0.5,
                              channels=channels, in_channels=1)
    elif kind == "dwt":
        model = build_dwt_model(
            model_type="multibranch_plain",
            num_classes=len(CLASS_NAMES),
            level_lengths=test_data["level_lengths"],
            branch_channels=(24, 48),
            head_dropout=0.4,
        )
    elif kind == "envelope":
        model = EnvelopeDilated1D()
    elif kind == "envelope_v3":
        model = EnvelopeResNetV3(num_classes=len(CLASS_NAMES))
    else:
        raise ValueError(f"unknown kind: {kind}")

    model.load_state_dict(ckpt["model_state"])
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Per-model inference -> (N_test, num_classes) softmax
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_softmax(kind: str, model, test_data: dict, device: torch.device,
                   use_amp: bool) -> np.ndarray:
    """Run the model on test features; return (N, C) softmax probs as numpy."""
    model.eval().to(device)

    labels = test_data["labels"]
    n = int(labels.shape[0])
    all_probs = []

    if kind == "dwt":
        feats_per_level = test_data["features_per_level"]
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
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
    else:
        feats = test_data["features"]
        for i in range(0, n, BATCH):
            x = feats[i:i + BATCH].to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(x)
            else:
                logits = model(x)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers: per-class val F1, weights matrix, weighted soft voting
# ──────────────────────────────────────────────────────────────────────────────

def perclass_val_f1(val_softmax: np.ndarray, val_labels: np.ndarray,
                    num_classes: int) -> np.ndarray:
    """Per-class F1 of one model on its val set. Returns shape (C,)."""
    preds = val_softmax.argmax(1)
    return f1_score(val_labels, preds,
                    labels=list(range(num_classes)),
                    average=None, zero_division=0)


def weights_perclass(val_f1_stack: np.ndarray) -> np.ndarray:
    """val_f1_stack: (M, C). Returns column-stochastic (M, C) weights —
    column c sums to 1 over members."""
    eps = 1e-8
    w = val_f1_stack + eps
    return w / w.sum(axis=0, keepdims=True)


def weighted_soft_vote(probs_stack: np.ndarray,
                       weights: np.ndarray) -> np.ndarray:
    """probs_stack: (M, N, C); weights: (M, C). Returns (N, C) renormalized
    so each row is a valid probability distribution."""
    weighted = (probs_stack * weights[:, None, :]).sum(axis=0)
    return weighted / weighted.sum(axis=1, keepdims=True).clip(min=1e-12)


def combine_probs(probs_stack: np.ndarray, strategy: str,
                  weights: np.ndarray | None) -> np.ndarray:
    if strategy == "equal":
        return probs_stack.mean(axis=0)
    if strategy == "perclass_f1":
        if weights is None:
            raise ValueError("perclass_f1 requires a (M, C) weights matrix")
        return weighted_soft_vote(probs_stack, weights)
    raise ValueError(f"unknown strategy: {strategy}")


def load_val_outputs(member_key: str, base_dir: Path,
                     ) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Load saved val softmax / labels / metadata sigs for a member."""
    cfg = MODEL_REGISTRY[member_key]
    run_dir = base_dir / cfg["run_name"]
    sm_path = run_dir / "val_softmax.npy"
    lb_path = run_dir / "val_labels.npy"
    sg_path = run_dir / "val_meta_sigs.csv"
    if not (sm_path.exists() and lb_path.exists() and sg_path.exists()):
        raise FileNotFoundError(
            f"Missing val outputs for member '{member_key}' under {run_dir}.\n"
            f"Run: python compute_val_softmax.py")
    softmax = np.load(sm_path)
    labels  = np.load(lb_path)
    sigs_df = pd.read_csv(sg_path)
    sigs = [(str(r["ch1_path"]), int(r["col_index"]), int(r["seg_idx"]))
            for _, r in sigs_df.iterrows()]
    return softmax, labels, sigs


# ──────────────────────────────────────────────────────────────────────────────
#  Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion(cm, class_names, out_path: Path, title: str):
    n = len(class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=10, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_outputs(run_name: str, members: list[str],
                   mean_probs: np.ndarray,    # (N, C) already-combined probs
                   labels: np.ndarray,
                   metadata: list,
                   class_names: list,
                   out_dir: Path,
                   strategy_label: str) -> dict:
    """Compute metrics from already-combined ensemble probs and write artefacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = mean_probs.argmax(axis=1)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    p, r, f, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))),
        zero_division=0)

    # per-class metrics csv
    metrics_df = pd.DataFrame({
        "class":     class_names,
        "precision": p.round(4),
        "recall":    r.round(4),
        "f1":        f.round(4),
        "support":   support,
    })
    metrics_df.to_csv(out_dir / "per_class_metrics.csv", index=False)

    # confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names,
                    out_dir / "confusion_matrix.png",
                    title=f"{run_name}: confusion matrix (rows normalized)")

    # misclassified
    wrong_mask = preds != labels
    n_wrong = int(wrong_mask.sum())
    if n_wrong > 0:
        rows = []
        for i in np.where(wrong_mask)[0]:
            md = metadata[int(i)]
            rows.append({
                "test_sample_idx": int(i),
                "true_class":      class_names[int(labels[i])],
                "pred_class":      class_names[int(preds[i])],
                "pred_prob":       float(mean_probs[i, int(preds[i])]),
                "true_prob":       float(mean_probs[i, int(labels[i])]),
                "ch1_path":        md["ch1_path"],
                "col_index":       md["col_index"],
                "seg_idx":         md["seg_idx"],
                "speed_pct":       md.get("speed_pct", ""),
            })
        pd.DataFrame(rows).to_csv(
            out_dir / "misclassified_samples.csv", index=False)

    # summary
    summary_lines = [
        f"Run: {run_name}",
        f"Ensemble members ({strategy_label}): {members}",
        f"Test samples: {len(labels)}",
        f"Test accuracy: {acc:.4f}",
        f"Test macro-F1: {macro_f1:.4f}",
        f"Misclassified: {n_wrong} / {len(labels)}  ({100*n_wrong/len(labels):.2f}%)",
        f"",
        f"Per-class F1:",
    ]
    for i, name in enumerate(class_names):
        summary_lines.append(
            f"  {name:<22} P={p[i]:.3f}  R={r[i]:.3f}  F1={f[i]:.3f}  "
            f"(n={support[i]})")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "run_name": run_name,
        "members":  members,
        "acc":      acc,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(class_names, f.tolist())),
        "n_wrong":  n_wrong,
    }


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ENSEMBLE EVALUATION — 4-class")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp_effective = USE_AMP and device.type == "cuda"
    print(f"Device           : {device}")
    print(f"AMP              : {use_amp_effective}")

    # ── Collect the union of model keys we actually need ────────────────────
    needed_keys = sorted({m for _, members in ENSEMBLES for m in members})
    print(f"Members required : {needed_keys}")
    print()

    # ── Load each model's test features + checkpoint; run inference once ────
    per_model = {}          # key -> {"probs": (N,C), "labels": (N,), "meta_sigs": [...]}
    for key in needed_keys:
        cfg = MODEL_REGISTRY[key]
        kind = cfg["kind"]
        print(f"-- [{key}] run={cfg['run_name']} kind={kind}")
        print(f"   features : {cfg['features_dir']}")
        print(f"   ckpt     : {cfg['checkpoint']}")

        t0 = time.time()
        test_data = load_split_data(kind, Path(cfg["features_dir"]),
                                    split="test")
        ckpt = torch.load(cfg["checkpoint"], map_location=device,
                           weights_only=False)
        model = build_model(kind, ckpt, test_data)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   loaded ckpt (val_f1 in ckpt = {ckpt.get('val_f1', 'n/a')}), "
              f"{n_params:,} params")

        probs = infer_softmax(kind, model, test_data, device, use_amp_effective)
        elapsed = time.time() - t0

        per_model[key] = {
            "probs":     probs,                            # (N, C)
            "labels":    test_data["labels"].numpy(),      # (N,)
            "metadata":  test_data["metadata"],
            "meta_sigs": [_sig(m) for m in test_data["metadata"]],
        }
        # Quick sanity check on this model alone
        solo_preds = probs.argmax(1)
        solo_acc = accuracy_score(per_model[key]["labels"], solo_preds)
        solo_f1  = f1_score(per_model[key]["labels"], solo_preds,
                            average="macro", zero_division=0)
        print(f"   solo test  acc={solo_acc:.4f}  macro_f1={solo_f1:.4f}  "
              f"({elapsed:.1f}s)")
        print()

        # Free GPU memory between models
        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Verify sample alignment across all models ───────────────────────────
    print("Checking sample alignment across representations...")
    ref_key = needed_keys[0]
    ref_sigs   = per_model[ref_key]["meta_sigs"]
    ref_labels = per_model[ref_key]["labels"]
    for key in needed_keys[1:]:
        assert per_model[key]["meta_sigs"] == ref_sigs, \
            (f"Metadata order mismatch between '{ref_key}' and '{key}'. "
             "Cannot align ensembles; re-generate features from the same splits.csv.")
        assert np.array_equal(per_model[key]["labels"], ref_labels), \
            f"Label order mismatch between '{ref_key}' and '{key}'."
    print(f"  OK — {len(ref_sigs)} samples aligned across {len(needed_keys)} reps")
    print()

    # ── Build each ensemble (equal-weight soft voting) ──────────────────────
    leaderboard = []
    ensembles_root = Path(OUTPUT_BASE)
    for run_name, members in ENSEMBLES:
        out_dir = ensembles_root / run_name

        print("=" * 70)
        print(f"ENSEMBLE: {run_name}  (members={members})")
        print("=" * 70)

        out_dir.mkdir(parents=True, exist_ok=True)
        probs_stack = np.stack([per_model[m]["probs"] for m in members], axis=0)
        mean_probs  = probs_stack.mean(axis=0)

        # Save per-member test softmax for reproducibility
        for m in members:
            np.save(out_dir / f"softmax_{m}.npy", per_model[m]["probs"])
        np.save(out_dir / "labels.npy", ref_labels)

        strat_label = "soft voting (equal-weight mean of softmaxes)"

        cfg_lines = [
            f"RUN NAME         : {run_name}",
            f"Created          : {datetime.now().isoformat(timespec='seconds')}",
            f"Strategy         : {strat_label}",
            f"Members          : {members}",
            "",
            "Member sources:",
        ]
        for m in members:
            cfg_lines.append(f"  {m:<10} run={MODEL_REGISTRY[m]['run_name']}")
            cfg_lines.append(f"             ckpt={MODEL_REGISTRY[m]['checkpoint']}")
        (out_dir / "ensemble_config.txt").write_text(
            "\n".join(cfg_lines), encoding="utf-8")

        result = write_outputs(
            run_name=run_name, members=members,
            mean_probs=mean_probs,
            labels=ref_labels,
            metadata=per_model[ref_key]["metadata"],
            class_names=CLASS_NAMES,
            out_dir=out_dir,
            strategy_label=strat_label,
        )
        leaderboard.append(result)

        print(f"  acc      = {result['acc']:.4f}")
        print(f"  macro_f1 = {result['macro_f1']:.4f}")
        print(f"  wrong    = {result['n_wrong']} / {len(ref_labels)}")
        for k, v in result["per_class_f1"].items():
            print(f"    {k:<22} F1={v:.4f}")
        print(f"  outputs -> {out_dir}")
        print()

    # ── Leaderboard ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("LEADERBOARD  (single models + ensembles; sorted by macro_f1)")
    print("=" * 70)
    rows = []
    for key in needed_keys:
        m = per_model[key]
        preds = m["probs"].argmax(1)
        acc = accuracy_score(m["labels"], preds)
        p, r, f, sup = precision_recall_fscore_support(
            m["labels"], preds, labels=list(range(len(CLASS_NAMES))),
            zero_division=0)
        macro_f1 = f.mean()
        rows.append({
            "name":     f"(solo) {MODEL_REGISTRY[key]['run_name']}",
            "acc":      acc,
            "macro_f1": macro_f1,
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
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(lb_df.to_string(index=False))

    lb_path = ensembles_root / "ensemble_leaderboard.csv"
    lb_df.to_csv(lb_path, index=False)
    print(f"\nLeaderboard written -> {lb_path}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
