"""
STEP 5 — ENVELOPE-SPECTRUM 1D CNN TRAINING (4-class)
=====================================================
Sibling of train_stft_cnn_e4_modalex_227.py and train_dwt_cnn.py.
Trains a small 1D CNN on the envelope-spectrum features produced by
precompute_envelope_v1.py.

WHY THIS MODEL
--------------
Envelope-spectrum features have two properties that steer the architecture:

  (a) Peaks are NARROW
      - Bearing-fault harmonics in the envelope spectrum occupy
        1-5 frequency bins at 1 Hz resolution.
      - Small kernels (k=3-7) are the right tool for local-peak detection.

  (b) Harmonic SPACING is much larger than peak width
      - BPFO lives in 44-87 Hz region depending on speed.
      - Successive BPFO harmonics are separated by ~44-87 bins.
      - A useful receptive field must span at least one full BPFO period
        to "see" two consecutive harmonics.

A stack of dilated 1D convs is the natural fit: small kernels for local
peak sensitivity, exponentially growing dilation to reach harmonic-scale
receptive fields cheaply. Dilation rates 1,2,4,8 with k=7 give RF~91 bins
at the 4th conv -- enough to span one BPFO cycle at 100% speed.

Final pooling concatenates AdaptiveAvgPool1d + AdaptiveMaxPool1d. Max
captures the strongest per-filter peak (what we care about); mean keeps
overall activation level (baseline comparison). This deviates from the
GAP-only convention in the STFT / DWT trainers -- done on purpose for
the spectral-peak nature of the input.

MODEL SIZE
----------
~77k parameters on (3, 501) input. ~13 params per training sample --
well under the safe zone, smallest of all models trained in this repo.

EXPERIMENTS SO FAR
------------------
  ENVELOPE_v1   dilated 1D CNN       ~77k params     ?

Usage
-----
    Edit the RUN CONFIG block, then:
    python train_envelope_cnn_v1.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              f1_score, accuracy_score)

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
#  RUN CONFIG
# ═══════════════════════════════════════════════════════════════════════════

RUN_NAME   = "ENVELOPE_dilated1d_v1"

FEATURES_DIR = r"C:\Project Work\Outputs\4class\v2_speed_strat\envelope\features"
OUTPUT_BASE  = r"C:\Project Work\Outputs\4class\training"

HEALTHY_SUBSAMPLE_N = None          # confirmed harmful; keep disabled
HEALTHY_CLASS_NAME  = "healthy 1"
USE_CLASS_WEIGHTS   = False         # default per prior runs

# Architecture — see file header for design justification
CONV_CHANNELS  = (32, 32, 64, 64, 128)   # 5 conv layers
CONV_KERNEL    = 7                       # odd; k-1 must be divisible by 2
CONV_DILATIONS = (1, 2, 4, 8, 1)         # exponential growth then cap
HEAD_DROPOUT   = 0.4

WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.0

EPOCHS        = 40
BATCH_SIZE    = 128      # input is tiny (3, 501); fits easily on 6 GB
LEARNING_RATE = 1e-3
PATIENCE      = 8
MIN_EPOCHS    = 12

NUM_WORKERS   = 0
SEED          = 42
USE_AMP       = True
USE_MMAP      = True
# ═══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset — same schema as the STFT .pt
# ──────────────────────────────────────────────────────────────────────────────

class EnvelopeTensorDataset(Dataset):
    def __init__(self, features, labels, metadata, class_names, source_file):
        assert features.shape[0] == labels.shape[0] == len(metadata)
        self.features    = features
        self.labels      = labels
        self.metadata    = metadata
        self.class_names = class_names
        self.source_file = source_file

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], idx


def load_pt_dataset(name: str, features_dir: Path,
                    subsample_healthy_n: int | None = None) -> EnvelopeTensorDataset:
    path = features_dir / f"{name}.pt"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    print(f"  Loading {name}.pt ...", end=" ", flush=True)
    t0 = time.time()
    data = torch.load(path, weights_only=False, mmap=USE_MMAP)
    elapsed = time.time() - t0

    features    = data["features"]
    labels      = data["labels"]
    metadata    = data["metadata"]
    class_names = data["class_names"]

    n_before   = features.shape[0]
    subsampled = False
    if subsample_healthy_n is not None and HEALTHY_CLASS_NAME in class_names:
        healthy_idx = class_names.index(HEALTHY_CLASS_NAME)
        is_healthy  = (labels == healthy_idx)
        n_healthy   = int(is_healthy.sum().item())
        if n_healthy > subsample_healthy_n:
            rng = np.random.default_rng(SEED)
            healthy_positions    = np.where(is_healthy.numpy())[0]
            keep_healthy         = rng.choice(healthy_positions,
                                              size=subsample_healthy_n,
                                              replace=False)
            nonhealthy_positions = np.where(~is_healthy.numpy())[0]
            keep_positions       = np.sort(np.concatenate(
                [nonhealthy_positions, keep_healthy]))
            features = features[keep_positions].clone()
            labels   = labels[keep_positions].clone()
            metadata = [metadata[i] for i in keep_positions.tolist()]
            subsampled = True

    size_mb = path.stat().st_size / (1024 ** 2)
    mode = "mmap" if USE_MMAP and not subsampled else "RAM"
    n_after = features.shape[0]
    if subsampled:
        print(f"{n_before} -> {n_after} samples (healthy subsampled), "
              f"{size_mb:.0f} MB, {mode} ({elapsed:.1f}s)")
    else:
        print(f"{n_after} samples, {size_mb:.0f} MB, {mode} ({elapsed:.1f}s)")

    return EnvelopeTensorDataset(
        features=features, labels=labels, metadata=metadata,
        class_names=class_names, source_file=name)


# ──────────────────────────────────────────────────────────────────────────────
#  Model — dilated 1D CNN with mean+max pooling head
# ──────────────────────────────────────────────────────────────────────────────

class EnvelopeDilated1D(nn.Module):
    """
    Dilated 1D CNN for envelope-spectrum classification.

    Input  : (B, 3, 501) float32   z-scored log1p envelope spectrum
    Output : (B, num_classes)

    Receptive-field progression with k=7 and dilations (1,2,4,8,1):
        conv1 RF =  7
        conv2 RF = 19
        conv3 RF = 43
        conv4 RF = 91    (covers one full BPFO period at 100% speed)
        conv5 RF = 93    (k=3 is a small integrator after the dilated stack)

    Pool: AdaptiveAvgPool1d(1) + AdaptiveMaxPool1d(1), concatenated -> 256 dim.
    Head: Dropout -> Linear(256, num_classes).
    """

    def __init__(self,
                 in_channels:  int   = 3,
                 num_classes:  int   = 4,
                 channels:     tuple = CONV_CHANNELS,
                 kernel_size:  int   = CONV_KERNEL,
                 dilations:    tuple = CONV_DILATIONS,
                 head_dropout: float = HEAD_DROPOUT):
        super().__init__()
        assert len(channels) == len(dilations), \
            "CONV_CHANNELS and CONV_DILATIONS must be the same length"
        assert kernel_size % 2 == 1, "kernel_size must be odd so 'same' padding works"

        # Build conv stack. Last conv uses k=3, d=1 as a small integrator.
        layers = []
        prev_c = in_channels
        n_convs = len(channels)
        for i, (out_c, d) in enumerate(zip(channels, dilations)):
            # Last layer: force k=3, d=1
            if i == n_convs - 1:
                k_here, d_here = 3, 1
            else:
                k_here, d_here = kernel_size, d
            pad = (k_here - 1) * d_here // 2
            layers += [
                nn.Conv1d(prev_c, out_c, kernel_size=k_here,
                          dilation=d_here, padding=pad, bias=False),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
            ]
            prev_c = out_c
        self.features = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        feature_dim = channels[-1] * 2   # avg || max
        self.classifier = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self.feature_dim = feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                    # (B, C_last, 501)
        x_avg = self.avg_pool(x).flatten(1)     # (B, C_last)
        x_max = self.max_pool(x).flatten(1)     # (B, C_last)
        return torch.cat([x_avg, x_max], dim=1)  # (B, 2*C_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


def build_model(num_classes: int) -> nn.Module:
    return EnvelopeDilated1D(
        in_channels=3,
        num_classes=num_classes,
        channels=CONV_CHANNELS,
        kernel_size=CONV_KERNEL,
        dilations=CONV_DILATIONS,
        head_dropout=HEAD_DROPOUT,
    )


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.numpy(), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Train / eval  (mirrors train_stft_cnn_e4_modalex_227.py)
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device,
              train: bool, epoch_label: str, n_batches: int, use_amp: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_indices = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    print(f"  >> {epoch_label}")
    t_phase = time.time()

    with ctx:
        for batch_i, (x, y, idx) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with autocast('cuda'):
                    logits = model(x)
                    loss   = criterion(logits, y)
                if train:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                logits = model(x)
                loss   = criterion(logits, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            all_indices.append(idx.numpy())

            interval = max(1, n_batches // 10)
            if batch_i % interval == 0 or batch_i == n_batches:
                elapsed = time.time() - t_phase
                rate    = batch_i / elapsed if elapsed > 0 else 0
                eta     = (n_batches - batch_i) / rate if rate > 0 else 0
                print(f"     batch {batch_i:>4}/{n_batches}  "
                      f"loss={loss.item():.3f}  "
                      f"({rate:.1f} batch/s, ETA {eta:.0f}s)")

    preds   = np.concatenate(all_preds)
    labels  = np.concatenate(all_labels)
    indices = np.concatenate(all_indices)
    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, preds, labels, indices


def plot_confusion(cm, class_names, out_path: Path):
    n = len(class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (rows normalized; raw counts shown)")
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=10, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_training_curves(log_df, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(log_df["epoch"], log_df["train_loss"], label="train", linewidth=1.5)
    ax1.plot(log_df["epoch"], log_df["val_loss"],   label="val",   linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(log_df["epoch"], log_df["train_f1"], label="train", linewidth=1.5)
    ax2.plot(log_df["epoch"], log_df["val_f1"],   label="val",   linewidth=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1")
    ax2.set_title("Macro-F1"); ax2.set_ylim(0, 1.02)
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fmt_time(seconds):
    if seconds < 60:   return f"{seconds:.0f}s"
    if seconds < 3600: return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"ENVELOPE 1D-CNN TRAINING - RUN: {RUN_NAME}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp_effective = USE_AMP and device.type == "cuda"

    print(f"Device             : {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Mixed precision    : {use_amp_effective}")
    print(f"Features dir       : {FEATURES_DIR}")
    print(f"Conv channels      : {CONV_CHANNELS}")
    print(f"Conv kernel        : {CONV_KERNEL}   (last layer hardcoded to k=3)")
    print(f"Conv dilations     : {CONV_DILATIONS}")
    print(f"Head dropout       : {HEAD_DROPOUT}")
    print(f"Healthy subsample  : {HEALTHY_SUBSAMPLE_N}")
    print(f"Class weighting    : {USE_CLASS_WEIGHTS}")
    print(f"Weight decay       : {WEIGHT_DECAY}")
    print(f"Label smoothing    : {LABEL_SMOOTHING}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Max epochs         : {EPOCHS}  (patience={PATIENCE}, min={MIN_EPOCHS})")
    print()

    out_dir = Path(OUTPUT_BASE) / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading precomputed features:")
    features_dir = Path(FEATURES_DIR)
    train_ds = load_pt_dataset("train", features_dir,
                               subsample_healthy_n=HEALTHY_SUBSAMPLE_N)
    val_ds   = load_pt_dataset("val",   features_dir)
    test_ds  = load_pt_dataset("test",  features_dir)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\n{num_classes} classes: {class_names}")
    print(f"Train: {len(train_ds)}   Val: {len(val_ds)}   Test: {len(test_ds)}")

    train_counts = np.bincount(train_ds.labels.numpy(), minlength=num_classes)
    print("\nTrain class distribution:")
    for i, name in enumerate(class_names):
        pct = 100 * train_counts[i] / len(train_ds)
        print(f"  {name:<22} {train_counts[i]:>5}  ({pct:5.1f}%)")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    model = build_model(num_classes).to(device)
    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EnvelopeDilated1D")
    print(f"  total parameters     : {n_params:,}")
    print(f"  trainable parameters : {n_trainable:,}")
    print(f"  params per train sample: {n_params / len(train_ds):.1f}")

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_ds.labels, num_classes).to(device)
        print(f"\nClass weights (inverse frequency, mean=1):")
        for name, w in zip(class_names, class_weights.cpu().numpy()):
            print(f"  {name:<22} {w:6.3f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                         label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler('cuda', enabled=use_amp_effective)

    config_lines = [
        f"RUN NAME          : {RUN_NAME}",
        f"started           : {datetime.now().isoformat(timespec='seconds')}",
        f"device            : {device}",
        f"mixed precision   : {use_amp_effective}",
        f"model type        : EnvelopeDilated1D",
        f"features dir      : {FEATURES_DIR}",
        f"classes           : {class_names}",
        f"train / val / test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}",
        f"conv channels     : {CONV_CHANNELS}",
        f"conv kernel       : {CONV_KERNEL} (last layer k=3)",
        f"conv dilations    : {CONV_DILATIONS}",
        f"head dropout      : {HEAD_DROPOUT}",
        f"total params      : {n_params:,}",
        f"params/sample     : {n_params / len(train_ds):.1f}",
        f"healthy subsample : {HEALTHY_SUBSAMPLE_N}",
        f"class weights     : {USE_CLASS_WEIGHTS}",
        f"weight decay      : {WEIGHT_DECAY}",
        f"label smoothing   : {LABEL_SMOOTHING}",
        f"batch size        : {BATCH_SIZE}",
        f"max epochs        : {EPOCHS}",
        f"early stop        : patience={PATIENCE}, min_epochs={MIN_EPOCHS}",
        f"optimizer         : AdamW  lr={LEARNING_RATE}",
        f"scheduler         : CosineAnnealingLR",
        f"seed              : {SEED}",
    ]
    (out_dir / "config.txt").write_text("\n".join(config_lines), encoding="utf-8")

    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)
    n_test_batches  = len(test_loader)

    print()
    print("-" * 80)
    print(f"{'epoch':>5}  {'tr_loss':>7} {'tr_acc':>6} {'tr_f1':>5}  | "
          f" {'va_loss':>7} {'va_acc':>6} {'va_f1':>5}    {'lr':>9}  {'time':>6}")
    print("-" * 80)

    log_rows     = []
    best_val_f1  = -1.0
    epochs_since_improve = 0
    best_path    = out_dir / "best_model.pt"
    t_train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, epoch_label=f"Epoch {epoch}/{EPOCHS} - training",
            n_batches=n_train_batches, use_amp=use_amp_effective)
        va_loss, va_acc, va_f1, _, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, epoch_label=f"Epoch {epoch}/{EPOCHS} - validating",
            n_batches=n_val_batches, use_amp=use_amp_effective)
        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        marker = ""
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            epochs_since_improve = 0
            torch.save({
                "model_state":    model.state_dict(),
                "class_names":    class_names,
                "model_type":     "EnvelopeDilated1D",
                "conv_channels":  CONV_CHANNELS,
                "conv_kernel":    CONV_KERNEL,
                "conv_dilations": CONV_DILATIONS,
                "head_dropout":   HEAD_DROPOUT,
                "epoch":          epoch,
                "val_f1":         va_f1,
            }, best_path)
            marker = " *"
        else:
            epochs_since_improve += 1

        print(f"{epoch:>5}  {tr_loss:7.4f} {tr_acc:6.3f} {tr_f1:5.3f}  | "
              f" {va_loss:7.4f} {va_acc:6.3f} {va_f1:5.3f}    "
              f"{lr_now:9.6f}  {fmt_time(elapsed):>6}{marker}")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss":   va_loss, "val_acc":   va_acc, "val_f1":   va_f1,
            "lr":         lr_now,  "epoch_time_s": elapsed,
        })

        if epoch >= MIN_EPOCHS and epochs_since_improve >= PATIENCE:
            print(f"\nEarly stop: val F1 hasn't improved in {PATIENCE} epochs.")
            break

    train_total = time.time() - t_train_start
    print("-" * 80)
    print(f"Training done in {fmt_time(train_total)}.  Best val F1 = {best_val_f1:.4f}")

    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    plot_training_curves(pd.DataFrame(log_rows), out_dir / "training_curves.png")

    print(f"\nLoading best model for test evaluation...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    te_loss, te_acc, te_f1, te_preds, te_labels, te_indices = run_epoch(
        model, test_loader, criterion, optimizer, scaler, device,
        train=False, epoch_label="TEST evaluation",
        n_batches=n_test_batches, use_amp=use_amp_effective)
    print(f"\nTEST RESULTS")
    print(f"  loss     = {te_loss:.4f}")
    print(f"  accuracy = {te_acc:.4f}")
    print(f"  macro-F1 = {te_f1:.4f}")

    p, r, f, support = precision_recall_fscore_support(
        te_labels, te_preds, labels=list(range(num_classes)), zero_division=0)
    metrics_df = pd.DataFrame({
        "class":     class_names,
        "precision": p.round(4),
        "recall":    r.round(4),
        "f1":        f.round(4),
        "support":   support,
    })
    metrics_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    print("\nPer-class test metrics:")
    print(metrics_df.to_string(index=False))

    cm = confusion_matrix(te_labels, te_preds, labels=list(range(num_classes)))
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    wrong_mask = te_preds != te_labels
    n_wrong = int(wrong_mask.sum())
    print(f"\nMisclassified test samples: {n_wrong} / {len(te_labels)} "
          f"({100*n_wrong/len(te_labels):.1f}%)")
    if n_wrong > 0:
        wrong_rows = []
        for i in np.where(wrong_mask)[0]:
            sample_i = int(te_indices[i])
            md = test_ds.metadata[sample_i]
            wrong_rows.append({
                "test_sample_idx": sample_i,
                "true_class":      class_names[int(te_labels[i])],
                "pred_class":      class_names[int(te_preds[i])],
                "ch1_path":        md["ch1_path"],
                "col_index":       md["col_index"],
                "seg_idx":         md["seg_idx"],
                "speed_pct":       md.get("speed_pct", ""),
            })
        pd.DataFrame(wrong_rows).to_csv(
            out_dir / "misclassified_samples.csv", index=False)

    summary = [
        f"Run: {RUN_NAME}",
        f"Model: EnvelopeDilated1D",
        f"Total params: {n_params:,}",
        f"Test accuracy: {te_acc:.4f}",
        f"Test macro-F1: {te_f1:.4f}",
        f"Best val F1:   {best_val_f1:.4f}",
        f"",
        f"Per-class F1:",
    ]
    for i, name in enumerate(class_names):
        summary.append(f"  {name:<22} P={p[i]:.3f}  R={r[i]:.3f}  F1={f[i]:.3f}  "
                       f"(n={support[i]})")
    (out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"DONE - outputs in: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    FEATURES_DIR = r"C:\Project Work\Outputs_nobpfo100\features\envelope"
    OUTPUT_BASE  = r"C:\Project Work\Outputs_nobpfo100\training"

    main()

