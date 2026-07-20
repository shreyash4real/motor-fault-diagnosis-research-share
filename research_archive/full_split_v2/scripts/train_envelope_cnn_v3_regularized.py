"""
STEP 5 — ENVELOPE-SPECTRUM 1D RESNET TRAINING (v3 + Full Research Artifacts)
===========================================================================
Refined v3 training with full artifact export for research paper standards.
Matches the structure and artifact output of DWT/STFT branches.

CHANGES IN V3 (Final Edition):
  1. REGULARIZATION: D=0.5, WD=1e-3, LS=0.1, Class Weights=True.
  2. ARCHITECTURE: ResNet1D + SE Blocks + Multi-scale Stem.
  3. ENSEMBLE EXPORTS: .npy logits, labels, softmax, and temperature.txt.
  4. PLOTS: training_curves.png and confusion_matrix.png.
  5. METRICS: per_class_metrics.csv and verbose summary.txt.
  6. METADATA: val/test_meta_sigs.csv and misclassified_samples_envelope.csv.
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

RUN_NAME   = "ENVELOPE_resnet_v3_reg"

FEATURES_DIR = legacy_path(r"C:\Project Work\Outputs\4class\v2_speed_strat\envelope\features")
OUTPUT_BASE  = legacy_path(r"C:\Project Work\Outputs\4class\training")

# v3 Parameters
CHANNELS        = (64, 128, 256)
DROPOUT_P       = 0.5
WEIGHT_DECAY    = 1e-3
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True

EPOCHS        = 60
BATCH_SIZE    = 64
LEARNING_RATE = 2e-3
PATIENCE      = 12
MIN_EPOCHS    = 15

NUM_WORKERS   = 0
SEED          = 42
USE_AMP       = True
USE_MMAP      = True
# ═══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EnvelopeTensorDataset(Dataset):
    def __init__(self, features, labels, metadata, class_names):
        self.features    = features
        self.labels      = labels
        self.metadata    = metadata
        self.class_names = class_names
    def __len__(self): return self.features.shape[0]
    def __getitem__(self, idx): return self.features[idx], self.labels[idx], idx

def load_pt_dataset(name: str, features_dir: Path) -> EnvelopeTensorDataset:
    path = features_dir / f"{name}.pt"
    if not path.exists(): sys.exit(1)
    data = torch.load(path, weights_only=False, mmap=USE_MMAP)
    return EnvelopeTensorDataset(data["features"], data["labels"], data["metadata"], data["class_names"])


# ──────────────────────────────────────────────────────────────────────────────
#  Model Architecture
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock1d(nn.Module):
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
        y = x.mean(dim=-1)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ResBlock1d(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=7, stride=stride, padding=3*dilation, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_c)
        self.se    = SEBlock1d(out_c)
        self.gelu  = nn.GELU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_c))
    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.gelu(out)

class EnvelopeResNet1D(nn.Module):
    def __init__(self, num_classes=4, channels=CHANNELS):
        super().__init__()
        self.stem_k3  = nn.Conv1d(3, channels[0]//2, kernel_size=3, padding=1, bias=False)
        self.stem_k11 = nn.Conv1d(3, channels[0]//2, kernel_size=11, padding=5, bias=False)
        self.stem_bn  = nn.BatchNorm1d(channels[0])
        self.gelu     = nn.GELU()
        self.layer1 = ResBlock1d(channels[0], channels[0], dilation=1)
        self.layer2 = ResBlock1d(channels[0], channels[1], stride=2, dilation=2)
        self.layer3 = ResBlock1d(channels[1], channels[1], dilation=4)
        self.layer4 = ResBlock1d(channels[1], channels[2], stride=2, dilation=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[2] * 2, channels[2]),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(channels[2], num_classes)
        )
    def encode(self, x):
        x = self.gelu(self.stem_bn(torch.cat([self.stem_k3(x), self.stem_k11(x)], dim=1)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)
    def forward(self, x): return self.classifier(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
#  Train / Eval / Utils
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train, use_amp):
    model.train() if train else model.eval()
    total_loss, all_logits, all_labels, all_indices = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for x, y, idx in loader:
            x, y = x.to(device), y.to(device)
            with autocast('cuda', enabled=use_amp):
                logits = model(x); loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * x.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.cpu().numpy()); all_indices.append(idx.numpy())
    
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds  = logits.argmax(1)
    return total_loss / len(loader.dataset), accuracy_score(labels, preds), f1_score(labels, preds, average="macro", zero_division=0), logits, labels, np.concatenate(all_indices)


def calibrate_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    print("  Optimizing temperature...", end=" ", flush=True)
    logits_t, labels_t = torch.from_numpy(logits), torch.from_numpy(labels)
    T = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def eval_nll():
        opt.zero_grad(); loss = F.cross_entropy(logits_t / T, labels_t); loss.backward(); return loss
    opt.step(eval_nll)
    print(f"done (T={T.item():.3f})")
    return T.item()


def plot_training_curves(log_df, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(log_df["epoch"], log_df["tr_loss"], label="train"); ax1.plot(log_df["epoch"], log_df["va_loss"], label="val")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(log_df["epoch"], log_df["tr_f1"], label="train"); ax2.plot(log_df["epoch"], log_df["va_f1"], label="val")
    ax2.set_title("Macro-F1"); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()


def plot_confusion(cm, class_names, out_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"ENVELOPE v3 FINAL - RUN: {RUN_NAME}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(OUTPUT_BASE) / RUN_NAME; out_dir.mkdir(parents=True, exist_ok=True)
    
    train_ds = load_pt_dataset("train", Path(FEATURES_DIR)); val_ds = load_pt_dataset("val", Path(FEATURES_DIR)); test_ds = load_pt_dataset("test", Path(FEATURES_DIR))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True); val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False); test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = EnvelopeResNet1D(num_classes=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    
    counts = np.bincount(train_ds.labels.numpy())
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS); scaler = GradScaler('cuda', enabled=USE_AMP)

    # 1. Verbose Config
    config = [f"RUN: {RUN_NAME}", f"Params: {n_params:,}", f"Channels: {CHANNELS}", f"Dropout: {DROPOUT_P}", f"Weight Decay: {WEIGHT_DECAY}", f"Smoothing: {LABEL_SMOOTHING}", f"Class Weights: {weights.cpu().numpy()}", f"LR: {LEARNING_RATE}", f"Batch: {BATCH_SIZE}", f"Seed: {SEED}"]
    (out_dir / "config.txt").write_text("\n".join(config))

    log_rows, best_val_f1, patience = [], -1.0, 0
    print(f"\n{'Epoch':>5} | {'Tr F1':>6} | {'Val F1':>6}")
    print("-" * 30)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True, USE_AMP)
        va_loss, va_acc, va_f1, _, _, _ = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, USE_AMP)
        scheduler.step()
        log_rows.append({"epoch": epoch, "tr_loss": tr_loss, "tr_f1": tr_f1, "va_loss": va_loss, "va_f1": va_f1})
        status = "*" if va_f1 > best_val_f1 else ""
        if status: best_val_f1 = va_f1; torch.save({"model_state": model.state_dict()}, out_dir / "best_model.pt"); patience = 0
        else: patience += 1
        print(f"{epoch:5d} | {tr_f1:6.3f} | {va_f1:6.3f} {status}")
        if epoch >= MIN_EPOCHS and patience >= PATIENCE: break

    # 2. Results & Artifacts
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    plot_training_curves(pd.DataFrame(log_rows), out_dir / "training_curves.png")
    
    model.load_state_dict(torch.load(out_dir / "best_model.pt")["model_state"])
    
    # Val artifacts (temp cal)
    _, _, _, va_logits, va_labels, va_indices = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, USE_AMP)
    T = calibrate_temperature(va_logits, va_labels)
    (out_dir / "temperature.txt").write_text(f"{T:.6f}")
    np.save(out_dir / "val_logits.npy", va_logits); np.save(out_dir / "val_labels.npy", va_labels)
    np.save(out_dir / "val_softmax.npy", F.softmax(torch.from_numpy(va_logits) / T, dim=1).numpy())
    pd.DataFrame([val_ds.metadata[i] for i in va_indices]).to_csv(out_dir / "val_meta_sigs.csv", index=False)

    # Test artifacts
    te_loss, te_acc, te_f1, te_logits, te_labels, te_indices = run_epoch(model, test_loader, criterion, optimizer, scaler, device, False, USE_AMP)
    np.save(out_dir / "test_logits.npy", te_logits); np.save(out_dir / "test_labels.npy", te_labels)
    pd.DataFrame([test_ds.metadata[i] for i in te_indices]).to_csv(out_dir / "test_meta_sigs.csv", index=False)

    # Per-class & Confusion
    te_preds = te_logits.argmax(1)
    p, r, f, s = precision_recall_fscore_support(te_labels, te_preds, zero_division=0)
    metrics_df = pd.DataFrame({"class": train_ds.class_names, "precision": p, "recall": r, "f1": f, "support": s})
    metrics_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    plot_confusion(confusion_matrix(te_labels, te_preds), train_ds.class_names, out_dir / "confusion_matrix.png")

    # Summary & Misclassified
    summary = [f"Run: {RUN_NAME}", f"Params: {n_params}", f"Test Acc: {te_acc:.4f}", f"Test F1: {te_f1:.4f}\n", "Per-Class:"]
    for i, name in enumerate(train_ds.class_names): summary.append(f"  {name:<20}: P={p[i]:.3f} R={r[i]:.3f} F1={f[i]:.3f}")
    (out_dir / "summary.txt").write_text("\n".join(summary))

    wrong = te_preds != te_labels
    if wrong.any():
        wrong_rows = []
        for i in np.where(wrong)[0]:
            md = test_ds.metadata[int(te_indices[i])]
            wrong_rows.append({"idx": int(te_indices[i]), "true": train_ds.class_names[te_labels[i]], "pred": train_ds.class_names[te_preds[i]], "col": md["col_index"], "seg": md["seg_idx"], "speed": md["speed_pct"]})
        pd.DataFrame(wrong_rows).to_csv(out_dir / "misclassified_samples_envelope.csv", index=False)

    print(f"All Done. Artifacts saved in {out_dir}")

if __name__ == "__main__": main()
