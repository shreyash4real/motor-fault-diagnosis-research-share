"""
STEP 5 — DWT CNN TRAINING (multi-branch 1D, native resolution)
==============================================================
Two model families available via MODEL_TYPE:

  1. "multibranch_plain"   — one 1D conv stack per DWT sub-band,
                             concatenated at the head. No residuals.
                             ~170k params.

  2. "multibranch_resnet"  — same topology but each branch uses a pair
                             of residual blocks. Cleaner gradients at
                             depth; residual connections are essentially
                             free in param count. ~225k params.

The representation is a list of 10 sub-band tensors per segment at
their native DWT coefficient lengths. cD2 has 5011 coefs, cA10 has 34.
Each branch receives its native input length (NO resampling, NO
padding to common length), which is the whole point of using DWT
rather than a uniform-grid transform like STFT.

Stem kernel/stride per branch is chosen from the native length so
coarse bands don't get pooled to zero:

    L_j > 2000   (cD2, cD3)           stem: k=15, s=4
    500 < L_j <= 2000 (cD4, cD5)      stem: k=11, s=3
    100 < L_j <=  500 (cD6, cD7)      stem: k=7,  s=2
    L_j <= 100 (cD8..cD10, cA10)      stem: k=3,  s=1

After the stem each branch runs either a plain 2-conv block
(multibranch_plain) or two small residual blocks (multibranch_resnet),
then GAP to a 48-d per-branch vector. The head concatenates all 10
branch vectors (480-d) and classifies.

Outputs (mirrors STFT/Clarke runs):
  config.txt, training_log.csv, training_curves.png,
  per_class_metrics.csv, confusion_matrix.png,
  misclassified_samples.csv, best_model.pt, summary.txt

Usage
-----
    Edit the RUN CONFIG block, then:
    python train_dwt_cnn.py
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

RUN_NAME   = "DWT_multibranch_plain_v1"
MODEL_TYPE = "multibranch_plain"        # or "multibranch_plain"

# Per-branch channel progression. (c_stem, c_mid) applied identically
# to every sub-band. Tuned for ~220k total params at 10 branches.
BRANCH_CHANNELS = (24, 48)

FEATURES_DIR = r"C:\Project Work\outputs\4class\v2_speed_strat\dwt\features"
OUTPUT_BASE  = r"C:\Project Work\outputs\4class\training"

HEALTHY_SUBSAMPLE_N = None
HEALTHY_CLASS_NAME  = "healthy 1"
USE_CLASS_WEIGHTS   = False

HEAD_DROPOUT    = 0.4
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.0

EPOCHS        = 40
BATCH_SIZE    = 128      # safe on 6 GB VRAM for this model; raise to 256 if you want
LEARNING_RATE = 1e-3
PATIENCE      = 8
MIN_EPOCHS    = 12

NUM_WORKERS   = 4        # Ryzen 7 6800HS has 16 threads, 4 workers is comfortable
SEED          = 42
USE_AMP       = True
USE_MMAP      = True
# ═══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DWTTensorDataset(Dataset):
    """
    Per-sample item: (bands_tuple, label, idx)
      bands_tuple: tuple of N_LEVELS tensors, each (3, L_j)
      label:       torch.int64 scalar
      idx:         python int (for traceability)

    PyTorch's default_collate walks tuples element-wise, so the DataLoader
    will produce (stacked_bands_tuple, stacked_labels, stacked_idx) where
    stacked_bands_tuple is a tuple of N_LEVELS tensors each (B, 3, L_j).
    """
    def __init__(self, features_per_level, labels, metadata,
                 class_names, level_names, level_bands_hz, level_lengths,
                 source_file):
        n = labels.shape[0]
        for t in features_per_level:
            assert t.shape[0] == n, "Inconsistent sample count across levels"
        assert len(metadata) == n
        self.features_per_level = features_per_level
        self.labels             = labels
        self.metadata           = metadata
        self.class_names        = class_names
        self.level_names        = level_names
        self.level_bands_hz     = level_bands_hz
        self.level_lengths      = level_lengths
        self.source_file        = source_file

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        bands = tuple(self.features_per_level[j][idx]
                      for j in range(len(self.features_per_level)))
        return bands, self.labels[idx], idx


def load_pt_dataset(name: str, features_dir: Path,
                    subsample_healthy_n: int | None = None) -> DWTTensorDataset:
    path = features_dir / f"{name}.pt"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    print(f"  Loading {name}.pt ...", end=" ", flush=True)
    t0 = time.time()
    data = torch.load(path, weights_only=False, mmap=USE_MMAP)
    elapsed = time.time() - t0

    features_per_level = data["features_per_level"]
    labels             = data["labels"]
    metadata           = data["metadata"]
    class_names        = data["class_names"]
    level_names        = data["level_names"]
    level_bands_hz     = data["level_bands_hz"]
    level_lengths      = data["level_lengths"]

    n_before   = labels.shape[0]
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
            features_per_level = [t[keep_positions].clone()
                                   for t in features_per_level]
            labels   = labels[keep_positions].clone()
            metadata = [metadata[i] for i in keep_positions.tolist()]
            subsampled = True

    size_mb = path.stat().st_size / (1024 ** 2)
    mode = "mmap" if USE_MMAP and not subsampled else "RAM"
    n_after = labels.shape[0]
    if subsampled:
        print(f"{n_before} -> {n_after} samples (healthy subsampled), "
              f"{size_mb:.0f} MB, {mode} ({elapsed:.1f}s)")
    else:
        print(f"{n_after} samples, {size_mb:.0f} MB, {mode} ({elapsed:.1f}s)")

    return DWTTensorDataset(
        features_per_level=features_per_level,
        labels=labels, metadata=metadata,
        class_names=class_names,
        level_names=level_names,
        level_bands_hz=level_bands_hz,
        level_lengths=level_lengths,
        source_file=name)


# ──────────────────────────────────────────────────────────────────────────────
#  Per-branch stem config
# ──────────────────────────────────────────────────────────────────────────────

def stem_config(length: int) -> tuple[int, int]:
    """Return (kernel, stride) for the stem conv based on native length."""
    if length > 2000:
        return 15, 4
    if length > 500:
        return 11, 3
    if length > 100:
        return 7, 2
    return 3, 1


# ──────────────────────────────────────────────────────────────────────────────
#  Model 1 — multibranch_plain
# ──────────────────────────────────────────────────────────────────────────────

class PlainBranch(nn.Module):
    """
    Stem -> Conv -> BN -> GELU -> Conv -> BN -> GELU -> GAP.
    Outputs (B, c_mid).
    """
    def __init__(self, in_channels: int, c_stem: int, c_mid: int,
                 stem_k: int, stem_s: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c_stem,
                      kernel_size=stem_k, stride=stem_s,
                      padding=stem_k // 2, bias=False),
            nn.BatchNorm1d(c_stem),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(c_stem, c_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c_mid),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(c_mid, c_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c_mid),
            nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x).flatten(1)
        return x


# ──────────────────────────────────────────────────────────────────────────────
#  Model 2 — multibranch_resnet
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    Basic residual block for 1D signals. Projection-shortcut when
    in_channels != out_channels or stride > 1.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                                kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                                kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


class ResNetBranch(nn.Module):
    """
    Stem -> ResBlock(c_stem -> c_stem) -> ResBlock(c_stem -> c_mid, stride 2)
    -> GAP. Outputs (B, c_mid).
    """
    def __init__(self, in_channels: int, c_stem: int, c_mid: int,
                 stem_k: int, stem_s: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c_stem,
                      kernel_size=stem_k, stride=stem_s,
                      padding=stem_k // 2, bias=False),
            nn.BatchNorm1d(c_stem),
            nn.GELU(),
        )
        self.res1 = ResBlock1D(c_stem, c_stem, stride=1)
        self.res2 = ResBlock1D(c_stem, c_mid,  stride=2)
        self.gap  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.gap(x).flatten(1)
        return x


# ──────────────────────────────────────────────────────────────────────────────
#  Full multi-branch model
# ──────────────────────────────────────────────────────────────────────────────

class DWTMultiBranchCNN(nn.Module):
    """
    One 1D encoder per DWT sub-band, concat at the head.

    forward() expects a tuple/list of N_LEVELS tensors, each (B, 3, L_j).
    The order must match level_lengths (which came from precompute).
    """
    def __init__(self, num_classes: int, level_lengths: list[int],
                 branch_type: str = "resnet",
                 branch_channels: tuple = (24, 48),
                 head_dropout: float = 0.4,
                 in_channels: int = 3):
        super().__init__()
        assert branch_type in ("plain", "resnet")
        self.branch_type   = branch_type
        self.level_lengths = list(level_lengths)
        self.n_levels      = len(level_lengths)
        self.c_stem, self.c_mid = branch_channels
        self.in_channels   = in_channels

        branch_cls = PlainBranch if branch_type == "plain" else ResNetBranch
        branches = []
        for L in level_lengths:
            k, s = stem_config(L)
            branches.append(branch_cls(
                in_channels=in_channels,
                c_stem=self.c_stem, c_mid=self.c_mid,
                stem_k=k, stem_s=s))
        self.branches = nn.ModuleList(branches)

        fused_dim = self.n_levels * self.c_mid
        self.feature_dim = fused_dim
        self.head_hidden = 128

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, self.head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(self.head_hidden, num_classes),
        )

    def encode(self, bands):
        """
        bands: tuple/list of n_levels tensors, each (B, C, L_j).
        Returns (B, fused_dim).
        """
        feats = [branch(x) for branch, x in zip(self.branches, bands)]
        return torch.cat(feats, dim=1)

    def forward(self, bands):
        return self.classifier(self.encode(bands))


def build_model(model_type: str, num_classes: int,
                level_lengths: list[int],
                branch_channels: tuple,
                head_dropout: float) -> nn.Module:
    if model_type == "multibranch_plain":
        return DWTMultiBranchCNN(num_classes=num_classes,
                                  level_lengths=level_lengths,
                                  branch_type="plain",
                                  branch_channels=branch_channels,
                                  head_dropout=head_dropout)
    elif model_type == "multibranch_resnet":
        return DWTMultiBranchCNN(num_classes=num_classes,
                                  level_lengths=level_lengths,
                                  branch_type="resnet",
                                  branch_channels=branch_channels,
                                  head_dropout=head_dropout)
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

def bands_to_device(bands, device):
    """Move a tuple/list of tensors to `device` with non-blocking=True."""
    return [b.to(device, non_blocking=True) for b in bands]


def run_epoch(model, loader, criterion, optimizer, scaler, device,
              train: bool, epoch_label: str, n_batches: int, use_amp: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_indices = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    print(f"  >> {epoch_label}")
    t_phase = time.time()

    with ctx:
        for batch_i, (bands, y, idx) in enumerate(loader, 1):
            bands = bands_to_device(bands, device)
            y     = y.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with autocast('cuda'):
                    logits = model(bands)
                    loss   = criterion(logits, y)
                if train:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                logits = model(bands)
                loss   = criterion(logits, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            batch_n = y.size(0)
            total_loss += loss.item() * batch_n
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
    print(f"DWT CNN TRAINING — RUN: {RUN_NAME}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp_effective = USE_AMP and device.type == "cuda"

    print(f"Device             : {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM               : {vram_gb:.1f} GB")
    print(f"Mixed precision    : {use_amp_effective}")
    print(f"Model type         : {MODEL_TYPE}")
    print(f"Branch channels    : stem={BRANCH_CHANNELS[0]}  mid={BRANCH_CHANNELS[1]}")
    print(f"Features dir       : {FEATURES_DIR}")
    print(f"Healthy subsample  : {HEALTHY_SUBSAMPLE_N}")
    print(f"Class weighting    : {USE_CLASS_WEIGHTS}")
    print(f"Head dropout       : {HEAD_DROPOUT}")
    print(f"Weight decay       : {WEIGHT_DECAY}")
    print(f"Label smoothing    : {LABEL_SMOOTHING}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Num workers        : {NUM_WORKERS}")
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

    class_names   = train_ds.class_names
    num_classes   = len(class_names)
    level_names   = train_ds.level_names
    level_lengths = train_ds.level_lengths
    level_bands   = train_ds.level_bands_hz

    print(f"\n{num_classes} classes: {class_names}")
    print(f"Sub-bands: {len(level_names)}  ({level_names})")
    print("Per-branch stem config (derived from native length):")
    print(f"  {'band':<6} {'length':<8} {'k':>3} {'s':>3}  band (Hz)")
    for name, L, band in zip(level_names, level_lengths, level_bands):
        k, s = stem_config(L)
        print(f"  {name:<6} {L:<8} {k:>3} {s:>3}  "
              f"{band[0]:>7.2f} - {band[1]:<8.2f}")
    print(f"\nTrain: {len(train_ds)}   Val: {len(val_ds)}   Test: {len(test_ds)}")

    train_counts = np.bincount(train_ds.labels.numpy(), minlength=num_classes)
    print("\nTrain class distribution:")
    for i, name in enumerate(class_names):
        pct = 100 * train_counts[i] / len(train_ds)
        print(f"  {name:<22} {train_counts[i]:>5}  ({pct:5.1f}%)")

    pin = (device.type == "cuda")
    persistent = (NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persistent)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persistent)

    model = build_model(MODEL_TYPE, num_classes,
                        level_lengths=level_lengths,
                        branch_channels=BRANCH_CHANNELS,
                        head_dropout=HEAD_DROPOUT).to(device)
    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {MODEL_TYPE}")
    print(f"  total parameters      : {n_params:,}")
    print(f"  trainable parameters  : {n_trainable:,}")
    print(f"  params per train smpl : {n_params / len(train_ds):.1f}")

    # Per-branch parameter breakdown (useful sanity print)
    print("  parameters per branch :")
    for name, branch in zip(level_names, model.branches):
        bp = sum(p.numel() for p in branch.parameters())
        print(f"    {name:<6} {bp:>8,}")
    head_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"    {'HEAD':<6} {head_params:>8,}")

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
        f"branch channels   : stem={BRANCH_CHANNELS[0]} mid={BRANCH_CHANNELS[1]}",
        f"features dir      : {FEATURES_DIR}",
        f"classes           : {class_names}",
        f"sub-bands         : {level_names}",
        f"sub-band lengths  : {level_lengths}",
        f"train / val / test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}",
        f"total params      : {n_params:,}",
        f"params/sample     : {n_params / len(train_ds):.1f}",
        f"healthy subsample : {HEALTHY_SUBSAMPLE_N}",
        f"class weights     : {USE_CLASS_WEIGHTS}",
        f"head dropout      : {HEAD_DROPOUT}",
        f"weight decay      : {WEIGHT_DECAY}",
        f"label smoothing   : {LABEL_SMOOTHING}",
        f"batch size        : {BATCH_SIZE}",
        f"num workers       : {NUM_WORKERS}",
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
                "model_state":    model.state_dict(),
                "class_names":    class_names,
                "model_type":     MODEL_TYPE,
                "branch_channels": BRANCH_CHANNELS,
                "level_lengths":  level_lengths,
                "level_names":    level_names,
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
