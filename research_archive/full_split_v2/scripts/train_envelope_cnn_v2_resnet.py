"""
STEP 5 — ENVELOPE-SPECTRUM 1D RESNET TRAINING (v2 Refinement)
=============================================================
Refined iteration of train_envelope_cnn_v1.py. 

CHANGES IN V2:
  1. CAPACITY: Increased channels (64-256) to reach ~450k params (~75 params/sample).
  2. ARCHITECTURE: Switched to a Residual 1D structure to allow deeper learning.
  3. ATTENTION: Added Squeeze-and-Excitation (SE) blocks to allow the model to
     dynamically weight frequency bands (crucial for BPFO detection).
  4. STEM: Multi-scale entry (k=3 and k=11) to capture both sharp peaks 
     and broader harmonic clusters simultaneously.

WHY THESE CHANGES:
  V1 (77k params) plateaued at 0.72 recall for BPFO-3. The "thinner" model 
  likely lacked the expressive power to distinguish subtle BPFO harmonics 
  from background noise across different speeds. SE blocks specifically 
  help the model "ignore" the 50Hz line noise and focus on the 44-87Hz 
  fault regions.

Usage
-----
    python train_envelope_cnn_v2_resnet.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

from pipeline_paths import legacy_path

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

RUN_NAME   = "ENVELOPE_resnet_v2"

FEATURES_DIR = legacy_path(r"C:\Project Work\Outputs\4class\v2_speed_strat\envelope\features")
OUTPUT_BASE  = legacy_path(r"C:\Project Work\Outputs\4class\training")

# Model Hyperparams
CHANNELS       = (64, 128, 256)    # Wider than v1
DROPOUT_P      = 0.4
WEIGHT_DECAY   = 5e-4              # Slightly higher for larger model
LABEL_SMOOTHING = 0.05             # Added mild smoothing

EPOCHS        = 50
BATCH_SIZE    = 64                 # Smaller batch for more updates per epoch
LEARNING_RATE = 1e-3
PATIENCE      = 10
MIN_EPOCHS    = 15

NUM_WORKERS   = 0
SEED          = 42
USE_AMP       = True
USE_MMAP      = True
# ═══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset (identical to v1)
# ──────────────────────────────────────────────────────────────────────────────

class EnvelopeTensorDataset(Dataset):
    def __init__(self, features, labels, metadata, class_names, source_file):
        self.features    = features
        self.labels      = labels
        self.metadata    = metadata
        self.class_names = class_names

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], idx


def load_pt_dataset(name: str, features_dir: Path) -> EnvelopeTensorDataset:
    path = features_dir / f"{name}.pt"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    print(f"  Loading {name}.pt ...", end=" ", flush=True)
    t0 = time.time()
    data = torch.load(path, weights_only=False, mmap=USE_MMAP)
    print(f"done ({time.time()-t0:.1f}s)")

    return EnvelopeTensorDataset(
        features=data["features"], 
        labels=data["labels"], 
        metadata=data["metadata"],
        class_names=data["class_names"], 
        source_file=name
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Model Architecture - 1D ResNet + SE Blocks
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation for 1D: channel-wise attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=-1) # Global Average Pool
        y = self.fc(y).view(b, c, 1)
        return x * y


class ResBlock1d(nn.Module):
    """Residual block: [Conv-BN-GELU-Conv-BN] + SE + Shortcut."""
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=7, stride=stride, 
                               padding=3*dilation, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_c)
        self.se    = SEBlock1d(out_c)
        self.gelu  = nn.GELU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.gelu(out)


class EnvelopeResNet1D(nn.Module):
    """
    Refined 1D ResNet for Envelope Spectrum.
    ~450k parameters.
    """
    def __init__(self, num_classes=4, channels=CHANNELS):
        super().__init__()
        
        # Multi-scale Stem: capture different peak widths
        # k=3 for sharp harmonics, k=11 for broader spectral features
        self.stem_k3  = nn.Conv1d(3, channels[0]//2, kernel_size=3, padding=1, bias=False)
        self.stem_k11 = nn.Conv1d(3, channels[0]//2, kernel_size=11, padding=5, bias=False)
        self.stem_bn  = nn.BatchNorm1d(channels[0])
        self.gelu     = nn.GELU()

        # ResNet Body
        # dilations help span the BPFO harmonics (~44-87 bins apart)
        self.layer1 = ResBlock1d(channels[0], channels[0], dilation=1)
        self.layer2 = ResBlock1d(channels[0], channels[1], stride=2, dilation=2)
        self.layer3 = ResBlock1d(channels[1], channels[1], dilation=4)
        self.layer4 = ResBlock1d(channels[1], channels[2], stride=2, dilation=1)

        # Head: Mean + Max Pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(channels[2] * 2, channels[2]),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(channels[2], num_classes)
        )

    def encode(self, x):
        # Stem
        x3 = self.stem_k3(x)
        x11 = self.stem_k11(x)
        x = torch.cat([x3, x11], dim=1)
        x = self.gelu(self.stem_bn(x))

        # Body
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pool
        x_avg = self.avg_pool(x).flatten(1)
        x_max = self.max_pool(x).flatten(1)
        return torch.cat([x_avg, x_max], dim=1)

    def forward(self, x):
        return self.classifier(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
#  Train / Eval Loop (standard for this project)
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train, use_amp):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_indices = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for x, y, idx in loader:
            x, y = x.to(device), y.to(device)
            with autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss   = criterion(logits, y)
            
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_indices.append(idx.numpy())

    avg_loss = total_loss / len(loader.dataset)
    labels = np.concatenate(all_labels)
    preds  = np.concatenate(all_preds)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, preds, labels, np.concatenate(all_indices)


def main():
    print("=" * 70)
    print(f"ENVELOPE v2 RESNET TRAINING - RUN: {RUN_NAME}")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(OUTPUT_BASE) / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_ds = load_pt_dataset("train", Path(FEATURES_DIR))
    val_ds   = load_pt_dataset("val",   Path(FEATURES_DIR))
    test_ds  = load_pt_dataset("test",  Path(FEATURES_DIR))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = EnvelopeResNet1D(num_classes=len(train_ds.class_names)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler('cuda', enabled=USE_AMP)

    best_val_f1 = -1.0
    patience_counter = 0
    log_rows = []

    print(f"\n{'Epoch':>5} | {'Tr Loss':>8} {'Tr F1':>6} | {'Val Loss':>8} {'Val F1':>6} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True, USE_AMP)
        va_loss, va_acc, va_f1, _, _, _ = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, USE_AMP)
        
        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        status = ""
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({"model_state": model.state_dict(), "class_names": train_ds.class_names}, out_dir / "best_model.pt")
            status = "*"
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"{epoch:5d} | {tr_loss:8.4f} {tr_f1:6.3f} | {va_loss:8.4f} {va_f1:6.3f} | {lr_now:8.6f} {status}")
        
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1, "val_loss": va_loss, "val_f1": va_f1})
        
        if epoch >= MIN_EPOCHS and patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final Test
    print("\nEvaluating best model on test set...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc, te_f1, te_preds, te_labels, _ = run_epoch(model, test_loader, criterion, optimizer, scaler, device, False, USE_AMP)
    
    print(f"Test Accuracy: {te_acc:.4f} | Test F1: {te_f1:.4f}")
    
    # Per-class Metrics
    p, r, f, s = precision_recall_fscore_support(te_labels, te_preds, zero_division=0)
    summary = [f"Run: {RUN_NAME}", f"Params: {n_params:,}", f"Test F1: {te_f1:.4f}\n", "Per-Class F1:"]
    for i, name in enumerate(train_ds.class_names):
        summary.append(f"  {name:<20}: P={p[i]:.3f} R={r[i]:.3f} F1={f[i]:.3f}")
    
    (out_dir / "summary.txt").write_text("\n".join(summary))
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    
    # Confusion Matrix
    cm = confusion_matrix(te_labels, te_preds)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig(out_dir / "confusion_matrix.png")
    
    print(f"Done. Results in {out_dir}")

if __name__ == "__main__":
    main()
