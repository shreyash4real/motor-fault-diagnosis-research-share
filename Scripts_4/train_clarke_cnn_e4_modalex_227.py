"""
STEP 5 — STFT CNN TRAINING (configurable architecture)
=======================================================
Two model families available via MODEL_TYPE:

  1. "vgg_style"      — the 5-block VGG CNN from Run A, configurable
                        channel widths via CHANNELS.

  2. "modified_alexnet" — faithful implementation of Alotaibi et al.'s
                        Modified AlexNet (Sensors 2023, paper id
                        sensors-23-07764):
                          * standard AlexNet conv stack unchanged:
                            conv1 11x11 stride 4, conv2 5x5, conv3-5 3x3
                          * channel progression 96, 256, 384, 384, 256
                          * BatchNorm after every conv layer
                          * all three FC layers replaced by GAP
                          * single Linear(256 -> num_classes) head

                        The paper uses 227x227 input (grayscale vibration
                        images). Our native STFT tensors are (3, 154, 149).
                        The model bilinearly upsamples to 227x227 inside
                        forward() so .pt files don't need regeneration.

                        Total params: ~3.75M. That's ~640 params per
                        train sample — well above the typical safe zone
                        for from-scratch training.

EXPERIMENTS SO FAR
------------------
  A_v2_baseline      vgg_style  (12, 24, 48, 96, 96)    340k params   F1=0.903
  B_v2_healthy_sub   same, healthy subsampled to 855    340k params   F1=0.866
  (new)              modified_alexnet @227x227          ~3.75M params  ?

Usage
-----
    Edit the RUN CONFIG block, then:
    python train_stft_cnn.py
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

RUN_NAME   = "CLARKE_mod_alexnet_v1"
MODEL_TYPE = "modified_alexnet"

# Used for both vgg_style and modified_alexnet — scaled to fit Clarke's
# (1, 128, 128) input and ~5.7k train samples. ~250k params, ~43 per sample.
CHANNELS = (32, 64, 96, 96, 64)

FEATURES_DIR = r"C:\Project Work\outputs\4class\v2_speed_strat\clarke_features"
OUTPUT_BASE  = r"C:\Project Work\outputs\4class\training"

HEALTHY_SUBSAMPLE_N = None
HEALTHY_CLASS_NAME  = "healthy 1"
USE_CLASS_WEIGHTS   = False

DROPOUT_P1      = 0.5    # single dropout before the final Linear in ModifiedAlexNet
DROPOUT_P2      = 0.3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.0

EPOCHS        = 40
BATCH_SIZE    = 64       # drop to 32 if you hit OOM on the 227x227 upsample
LEARNING_RATE = 1e-3
PATIENCE      = 8
MIN_EPOCHS    = 12

NUM_WORKERS   = 0
SEED          = 42
USE_AMP       = True
USE_MMAP      = True
# ═══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────────────

class STFTTensorDataset(Dataset):
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
                    subsample_healthy_n: int | None = None) -> STFTTensorDataset:
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

    return STFTTensorDataset(
        features=features, labels=labels, metadata=metadata,
        class_names=class_names, source_file=name)


# ──────────────────────────────────────────────────────────────────────────────
#  Model 1 — VGG-style (Run A baseline, configurable)
# ──────────────────────────────────────────────────────────────────────────────

class STFTCNN(nn.Module):
    def __init__(self, num_classes: int = 4,
                 channels: tuple = (12, 24, 48, 96, 96),
                 dropout1: float = 0.4, dropout2: float = 0.3):
        super().__init__()
        assert len(channels) == 5
        self.channels = channels
        self.feature_dim = channels[-1]

        def block(in_c, out_c, pool=True):
            layers = [
                nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        c1, c2, c3, c4, c5 = channels
        self.b1 = block(1,   c1)
        self.b2 = block(c1,  c2)
        self.b3 = block(c2,  c3)
        self.b4 = block(c3,  c4)
        self.b5 = block(c4,  c5, pool=False)
        self.gap = nn.AdaptiveAvgPool2d(1)

        head_hidden = max(32, self.feature_dim // 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(self.feature_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(head_hidden, num_classes),
        )

    def encode(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.b4(x); x = self.b5(x); x = self.gap(x)
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
#  Model 2 — Modified AlexNet (Alotaibi et al. 2023), faithful at 227x227
# ──────────────────────────────────────────────────────────────────────────────

class ModifiedAlexNet(nn.Module):
    """
    Faithful implementation of the Modified AlexNet from
    Alotaibi et al. (Sensors 2023, sensors-23-07764).

    The paper's two modifications to standard AlexNet:
      (a) BatchNorm added after every conv layer
      (b) All three FC layers replaced by a single GAP -> Linear head

    Everything else — filter sizes, strides, channel counts — matches
    the original AlexNet exactly.

    Feature-map progression on 227x227 input:
        227x227  -> conv1 11x11 stride 4         -> 55x55   (96 ch)
                 -> maxpool 3x3 stride 2         -> 27x27
                 -> conv2 5x5 pad 2              -> 27x27   (256 ch)
                 -> maxpool 3x3 stride 2         -> 13x13
                 -> conv3 3x3 pad 1              -> 13x13   (384 ch)
                 -> conv4 3x3 pad 1              -> 13x13   (384 ch)
                 -> conv5 3x3 pad 1              -> 13x13   (256 ch)
                 -> maxpool 3x3 stride 2         -> 6x6
                 -> GAP                          -> 1x1
                 -> Dropout -> Linear(256, K)

    Our STFT tensors are natively (3, 154, 149). They are upsampled to
    (3, 227, 227) inside forward() via bilinear interpolation. This keeps
    the precomputed .pt files unchanged while presenting the model with
    exactly the input size the paper used.

    Total parameters: ~3.75M (the paper reports 3,752,704).
    """
    TARGET_HW = None   # Clarke: no upsample, use native 128x128

    def __init__(self, num_classes: int = 4, dropout: float = 0.5,
                 channels: tuple = (32, 64, 96, 96, 64),
                 in_channels: int = 1):
        super().__init__()
        c1, c2, c3, c4, c5 = channels

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=11, stride=4, padding=0)
        self.bn1   = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(c1,  c2, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(c2,  c3, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(c3)

        self.conv4 = nn.Conv2d(c3,  c4, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(c4)

        self.conv5 = nn.Conv2d(c4,  c5, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(c5)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c5, num_classes),
        )

        self.feature_dim = c5

    def _upsample_to_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.TARGET_HW is None:
            return x
        if x.shape[-2:] != self.TARGET_HW:
            x = F.interpolate(x, size=self.TARGET_HW,
                              mode="bilinear", align_corners=False)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._upsample_to_target(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.gap(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


def build_model(model_type: str, num_classes: int, channels: tuple,
                dropout1: float, dropout2: float) -> nn.Module:
    if model_type == "vgg_style":
        return STFTCNN(num_classes=num_classes, channels=channels,
                       dropout1=dropout1, dropout2=dropout2)
    elif model_type == "modified_alexnet":
        # Single FC head uses one dropout value. Ignore dropout2.
        return ModifiedAlexNet(num_classes=num_classes, dropout=dropout1,
                               channels=channels, in_channels=1)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.numpy(), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Train / eval
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
    print(f"STFT CNN TRAINING — RUN: {RUN_NAME}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp_effective = USE_AMP and device.type == "cuda"

    print(f"Device             : {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Mixed precision    : {use_amp_effective}")
    print(f"Model type         : {MODEL_TYPE}")
    if MODEL_TYPE == "vgg_style":
        print(f"Channels           : {CHANNELS}")
    if MODEL_TYPE == "modified_alexnet":
        print(f"Input upsampling   : native (3, 154, 149) -> (3, 227, 227) bilinear")
    print(f"Features dir       : {FEATURES_DIR}")
    print(f"Healthy subsample  : {HEALTHY_SUBSAMPLE_N}")
    print(f"Class weighting    : {USE_CLASS_WEIGHTS}")
    print(f"Dropout            : {DROPOUT_P1} / {DROPOUT_P2}")
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

    model = build_model(MODEL_TYPE, num_classes, CHANNELS,
                        DROPOUT_P1, DROPOUT_P2).to(device)
    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {MODEL_TYPE}")
    print(f"  total parameters    : {n_params:,}")
    print(f"  trainable parameters: {n_trainable:,}")
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
        f"model type        : {MODEL_TYPE}",
        f"features dir      : {FEATURES_DIR}",
        f"classes           : {class_names}",
        f"train / val / test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}",
        f"total params      : {n_params:,}",
        f"params/sample     : {n_params / len(train_ds):.1f}",
        f"healthy subsample : {HEALTHY_SUBSAMPLE_N}",
        f"class weights     : {USE_CLASS_WEIGHTS}",
        f"dropout           : {DROPOUT_P1} / {DROPOUT_P2}",
        f"weight decay      : {WEIGHT_DECAY}",
        f"label smoothing   : {LABEL_SMOOTHING}",
        f"batch size        : {BATCH_SIZE}",
        f"max epochs        : {EPOCHS}",
        f"early stop        : patience={PATIENCE}, min_epochs={MIN_EPOCHS}",
        f"optimizer         : AdamW  lr={LEARNING_RATE}",
        f"scheduler         : CosineAnnealingLR",
        f"seed              : {SEED}",
    ]
    if MODEL_TYPE == "vgg_style":
        config_lines.insert(6, f"channels          : {CHANNELS}")
    if MODEL_TYPE == "modified_alexnet":
        config_lines.insert(6, f"input upsampling  : (154, 149) -> (227, 227) bilinear")
    (out_dir / "config.txt").write_text("\n".join(config_lines), encoding="utf-8")

    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)
    n_test_batches  = len(test_loader)

    print()
    print("─" * 80)
    print(f"{'epoch':>5}  {'tr_loss':>7} {'tr_acc':>6} {'tr_f1':>5}  | "
          f" {'va_loss':>7} {'va_acc':>6} {'va_f1':>5}    {'lr':>9}  {'time':>6}")
    print("─" * 80)

    log_rows     = []
    best_val_f1  = -1.0
    epochs_since_improve = 0
    best_path    = out_dir / "best_model.pt"
    t_train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, epoch_label=f"Epoch {epoch}/{EPOCHS} — training",
            n_batches=n_train_batches, use_amp=use_amp_effective)
        va_loss, va_acc, va_f1, _, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, epoch_label=f"Epoch {epoch}/{EPOCHS} — validating",
            n_batches=n_val_batches, use_amp=use_amp_effective)
        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        marker = ""
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            epochs_since_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "model_type":  MODEL_TYPE,
                "channels":    CHANNELS if MODEL_TYPE == "vgg_style" else None,
                "epoch":       epoch,
                "val_f1":      va_f1,
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
    print("─" * 80)
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
        f"Model: {MODEL_TYPE}",
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
    print(f"DONE — outputs in: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
