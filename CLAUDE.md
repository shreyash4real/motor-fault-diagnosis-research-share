# CLAUDE.md — Motor Fault Detection Research Project

## What This Project Is

A machine-learning research pipeline for **motor fault diagnosis using 3-phase electric current signals** from a centrifugal pump testbench. Goal: a research paper comparing signal representations (STFT, DWT, Mel, Clarke) for 4-class fault classification. The pipeline is fully operational and producing results; current focus is deeper analysis of the two leading representations (STFT and DWT, both ~95% test accuracy) for the paper.

Dataset: *Motor Current and Vibration Monitoring Dataset for Fault Detection in E-motor-driven Centrifugal Pump*, S. Bruinsma et al. — CC0.

---

## Hardware

| | |
|---|---|
| Machine | ASUS ROG Zephyrus G15 GA503RM |
| CPU | AMD Ryzen 7 6800HS — 16 threads @ 3.2 GHz |
| RAM | 16 GB |
| GPU | NVIDIA RTX 3060 Laptop — **6 GB VRAM** |
| OS | Windows 11 Home 64-bit |

All training scripts detect CUDA automatically. Batch sizes in scripts are tuned for 6 GB VRAM — do not raise them without checking VRAM headroom first.

---

## Repository Layout

- `Dataset\` — READ-ONLY source CSVs (electric + vibration). Never write here.
- `Denoised\` — bandpass-filtered copies, regenerable via `denoise_all.py`.
- `Outputs\main\` — PC setup + dataset-wide validation; `validated_manifest.csv` is the source of truth.
- `Outputs\4class\v2_speed_strat\` — canonical 4-class split + features (`features\`, `mel_v1\features\`, `clarke_features\`, `dwt\features\`, `stft_images\`, `dwt\images\`).
- `Outputs\4class\v1\` — old random split, kept for comparison only.
- `Outputs\4class\training\<RUN_NAME>\` — one folder per training run.
- `Outputs\vibration\` — complete and archived.
- `Scripts\` — Steps 0–2 (setup, validation, denoising). Dataset-wide.
- `Scripts_4\` — Steps 3–5 (splits, features, training). 4-class-specific. New experiments go here.
- `Appendices\`, `DxDiag.txt`, `README.txt` — reference docs.

---

## The 4-Class Problem

| Index | Label | Fault |
|-------|-------|-------|
| 0 | `healthy 1` | Baseline healthy |
| 1 | `stator short 1` | Stator winding short |
| 2 | `bearing bpfo 3` | Outer-race bearing defect — severity 3 |
| 3 | `broken rotor bar` | Rotor cage fracture |

This set is **locked**. All four labels are present at 50 %, 75 %, and 100 % VFD speeds. Expansion to more classes is exploratory only.

---

## Dataset Constraints

- **`Dataset\`** is read-only. No script writes to it — ever.
- **`Denoised\`** is always regenerable by re-running `denoise_all.py`. Do not edit it manually.
- Electric: 231 files, 26 fault labels (except `stator short 2` which has only 6 files), 9 files per label otherwise, 3 channels per file, 300 000 samples @ 20 kHz, 15 s recordings.
- Each CSV column `0` is the time axis; signal data starts at column index `1`. Scripts read signals as `df.iloc[:, col_idx + 1]`.
- All 231 electric files passed validation (100 % PASS).

---

## Pipeline Steps

### Step 0 — PC Setup (`Scripts\setup_pc.py`)
Walks `Dataset\Electric\Motor-2`, builds `manifest.csv` + `validated_manifest.csv`, writes `folder_tree.txt` and `setup_summary.txt` into `Outputs\main\`. Run this first on a new machine.

### Step 1 — Validation (`Scripts\validate_dataset.py`)
Checks each CSV: can pandas open it, row count == 300 000, no all-NaN columns, no flat-zero columns (motor not running), no clipped columns (>98 % saturation), RMS in 0.001–50 A. Writes PASS/WARN/FAIL verdict per file into `validated_manifest.csv` and a human-readable `validation_report.txt`.

### Step 2 — Denoising (`Scripts\denoise_all.py`)
4th-order Butterworth bandpass **5–5000 Hz**, zero-phase `filtfilt`. Skips FAIL files; processes PASS and WARN. Mirrors the input folder tree under `Denoised\Motor-2\`. The vibration variant uses SOS form for numerical stability at 0.95 Nyquist (6–9500 Hz).

### Step 3 — Split Generation (`Scripts_4\generate_splits_v2.py`)

**v2 is canonical.** Splits at **column level** (not segment level) to prevent data leakage: 75 % overlap segmentation means adjacent segments share 15 000 / 20 000 samples, so a segment-level split would contaminate test with train data.

Stratification: each `(class, speed)` group is split independently at 70 / 15 / 15. For small groups (n=3): exactly 1 train, 1 val, 1 test. This guarantees every class has train/val/test coverage at every speed.

Each class gets its own `np.random.default_rng(SEED + hash(cls) % 10_000)`, so splits are class-independent but fully reproducible.

**Why v1 failed for bearing bpfo 3**: v1 pooled all speeds before splitting, which by chance put all bearing bpfo 3 val columns at 50 % speed and all test columns at 75/100 % speed. The model validated on 50 %-speed features and was tested on unseen speeds → val F1 looked fine but test F1 cratered to 0.564. v2 prevents this.

v1 outputs are kept in `Outputs\4class\v1\` for comparison only.

Outputs: `Outputs\4class\v2_speed_strat\splits.csv` + `split_report.txt`. Seed: **42** everywhere.

### Step 4 — Feature Precomputation

All precompute scripts read from `Denoised\Motor-2\`, write `.pt` + PNGs to `Outputs\4class\v2_speed_strat\`.

**Segmentation** (shared by all representations):
- Window: 20 000 samples (1 s)
- Stride: `(300 000 − 20 000) // (57 − 1) = 5 000` samples (0.25 s) → **75 % overlap**
- Segments per column: **57**

#### 4a. Linear STFT (`precompute_stft_v2.py`)

Locked parameters:
```
nperseg  = 1024
noverlap = 896          (87.5% overlap within STFT)
window   = hann
boundary = None
padded   = False
freq cutoff = 3000 Hz → 154 frequency bins
output shape = (3, 154, 149) float32
```

Two representations diverge intentionally from the same STFT data:
- **PNG** — raw log-magnitude dB, jet colormap, range [−50, +10] dB, 154 × 149 px, no axes/title. Matches conventional MCSA literature rendering.
- **Training tensor** — per-image z-score across all 3 channels together (preserves relative inter-phase magnitudes). Stored in `.pt`.

PNG tree: `stft_images/<split>/<class>/<speed>/colXX_segYYY_ch{1,2,3}.png` — 3 PNGs per segment.

#### 4b. Mel STFT (`precompute_mel_stft_v2.py`)

Same STFT base, then librosa Slaney-norm mel filterbank over 0–3000 Hz → 96 mel bins. Output shape: (3, 96, 149). Features in `mel_v1\features\`.

#### 4c. Clarke Transform (`precompute_clarke_v2.py`)

Amplitude-invariant α/β transform:
```
i_α = (2/3)(i_a − 0.5·i_b − 0.5·i_c)
i_β = (2/3)((√3/2)·i_b − (√3/2)·i_c)
```
128 × 128 log1p density histogram, per-sample symmetric max extent (scale-invariant). Output shape: (1, 128, 128). Features in `clarke_features\`.

#### 4d. DWT (`precompute_dwt_v1.py`)

```
Wavelet:         db8
Decomposition:   level 10 (max for 20 000 samples with db8)
Boundary:        symmetric
Dropped:         cD1 — covers 5–10 kHz, above the 5 kHz denoise cutoff
```

**10 kept sub-bands** (stored in this order):

| Band | Freq (Hz) | Length | Notes |
|------|-----------|--------|-------|
| cD2 | 2500–5000 | 5011 | |
| cD3 | 1250–2500 | 2513 | |
| cD4 | 625–1250 | 1264 | |
| cD5 | 312–625 | 639 | |
| cD6 | 156–312 | 327 | |
| cD7 | 78–156 | 171 | upper BPFO sideband @ 100% |
| cD8 | 39–78 | 93 | line fundamental + BPFO |
| cD9 | 20–39 | 54 | BPFO @ 50%, lower sidebands |
| cD10 | 10–20 | 34 | shaft rotation |
| cA10 | 0–10 | 34 | slip-modulated content |

**Normalization**: per-level z-score computed over all `3 × L_j` values in that sub-band. Per-level (not global) because cD8 energy exceeds cD2 energy by orders of magnitude — a global z-score would suppress the low-energy bands entirely.

**.pt format**: dict with `features_per_level` (10 tensors, each `(N, 3, L_j)`), plus `level_names`, `level_bands_hz`, `level_lengths`, `labels`, `class_names`, `metadata`, `dwt_params`. Key name `features_per_level` differs from the STFT/Clarke convention.

PNG: one diagnostic panel per segment (10 rows × abs envelope of phase-1 cD_j), `dwt\images\<split>\<class>\<speed>\colXX_segYYY.png`.

### Step 5 — Training

#### STFT / Mel / Clarke CNNs (`train_stft_cnn_*.py`, `train_clarke_cnn_*.py`)

Two model families available via `MODEL_TYPE`:

**`vgg_style`** (runs A, B, C — baseline):
- 5 × [Conv-BN-ReLU, Conv-BN-ReLU, MaxPool] blocks
- Channels: (12, 24, 48, 96, 96), no pool on block 5
- GAP → Dropout → Linear(96, 48) → ReLU → Dropout → Linear(48, 4)
- ~337k params

**`modified_alexnet`** (runs E4 — paper implementation):
- Faithful Alotaibi et al. (Sensors 2023, sensors-23-07764)
- AlexNet conv stack: 11×11 s4, 5×5, 3×3 × 3 with channel progression 96→256→384→384→256
- BatchNorm after every conv (the paper's modification)
- All 3 FC layers replaced by GAP → Dropout → Linear(256, 4) (the paper's modification)
- Input: (3, 154, 149) bilinearly upsampled to (3, 227, 227) inside `forward()` — `.pt` files unchanged
- ~3.75M params

#### DWT multi-branch CNN (`train_dwt_cnn.py`)

One 1D encoder branch per sub-band at **native coefficient length** — no resampling or padding.

Stem kernel/stride selected by length bucket:
```
L > 2000  (cD2, cD3)             k=15, s=4
500 < L ≤ 2000  (cD4, cD5)       k=11, s=3
100 < L ≤ 500   (cD6, cD7)        k=7, s=2
L ≤ 100   (cD8–cD10, cA10)        k=3, s=1
```

**`multibranch_plain`**: Stem → Conv-BN-GELU → Conv-BN-GELU → GAP(1d) → 48-d vector, per branch.

**`multibranch_resnet`**: Stem → ResBlock(c_stem→c_stem) → ResBlock(c_stem→c_mid, stride 2) → GAP → 48-d vector, per branch.

Head: concat all 10 branches (480-d) → Linear(480, 128) → GELU → Dropout(0.4) → Linear(128, 4).

DataLoader collates bands as a tuple of tensors — `__getitem__` returns `(tuple_of_10_tensors, label, idx)`.

#### Common training config (all scripts)

```
Optimizer : AdamW, lr=1e-3, weight_decay=1e-4
Scheduler : CosineAnnealingLR(T_max=EPOCHS)
AMP       : torch.cuda.amp autocast + GradScaler (CUDA only)
Early stop: patience=8, min_epochs=12 (monitors val macro-F1)
Seed      : torch.manual_seed(42), np.random.seed(42)
mmap      : torch.load(..., mmap=True) keeps GPU VRAM pressure low
```

Every run writes to `Outputs\4class\training\<RUN_NAME>\`:
`best_model.pt`, `config.txt`, `summary.txt`, `training_log.csv`, `training_curves.png`, `confusion_matrix.png`, `per_class_metrics.csv`, `misclassified_samples.csv`.

---

## All Model Results (v2 speed-stratified split)

| Run | Representation | Test Acc | Macro-F1 | Params | bpfo-3 F1 |
|-----|----------------|----------|----------|--------|-----------|
| `E4_mod_alexnet_227` | STFT → AlexNet 227×227 | **95.39%** | 0.9393 | 3.75 M | 0.793 |
| `E4_mod_alexnet_mel96` | Mel → AlexNet | 95.18% | 0.9383 | 3.75 M | 0.791 |
| `DWT_multibranch_plain_v1` | DWT plain | 95.10% | 0.9358 | 174 k | 0.782 |
| `A_v2_baseline` | STFT → VGG-style | 92.84% | 0.9033 | 337 k | 0.669 |
| `DWT_multibranch_resnet_v1` | DWT ResNet | 92.91% | 0.9001 | 222 k | 0.683 |
| `B_v2_healthy_sub` | STFT (subsampled) | 87.79% | 0.8660 | 337 k | 0.564 |
| `E5_mod_resnet` | STFT → ResNet | 85.31% | 0.8247 | 2.80 M | 0.422 |
| `CLARKE_mod_alexnet_v1` | Clarke → AlexNet | 81.14% | 0.7904 | 250 k | 0.441 |
| `C_v1_healthy_sub` | STFT v1-split sub | 83.55% | 0.8257 | 337 k | 0.439 |

**Focus for the paper**: STFT and DWT — both ~95%. Clarke is a clear third; healthy subsampling consistently hurts.

**Consistent weakness**: `bearing bpfo 3` — low recall (56–73%) across all models. All other classes hit F1 ≥ 0.93. This is the hardest class and likely a key finding for the paper.

**Key insight from runs B/C**: healthy subsampling reduces performance. The imbalance (healthy has ~96 columns, fault classes have 15–21) does not hurt the model — the imbalance is real and the model learns from it correctly. Do not subsample.

---

## Ensemble Leaderboard Perspective

Best current result: `ENSEMBLE_stft_dwt_envelope_temperature` with **96.49% test accuracy** and **0.9524 macro-F1** on the canonical v2 speed-stratified split.

| Run | Test Acc | Macro-F1 | bpfo-3 F1 | Notes |
|-----|----------|----------|-----------|-------|
| `ENSEMBLE_stft_dwt_envelope_temperature` | **96.49%** | **0.9524** | 0.837 | Best overall; STFT + DWT + envelope, temperature calibrated |
| `ENSEMBLE_stft_dwt_temperature` | 96.05% | 0.9470 | 0.819 | Strong two-representation ensemble |
| `ENSEMBLE_stft_dwt_clarke_temperature` | 95.98% | 0.9449 | 0.811 | Clarke does not improve the best ensemble |
| `(solo, T=2.400) E4_mod_alexnet_227` | 95.39% | 0.9393 | 0.793 | Best solo STFT model |
| `(solo, T=0.743) DWT_multibranch_plain_v1` | 95.03% | 0.9347 | 0.778 | Best solo DWT model |
| `(solo, T=1.172) ENVELOPE_dilated1d_v1` | 93.27% | 0.9132 | 0.792 | Weaker solo, useful in the best ensemble |
| `(solo, T=2.849) CLARKE_mod_alexnet_v1` | 81.21% | 0.7903 | 0.439 | Not useful in the best ensemble |

Remaining errors are still dominated by `bearing bpfo 3`, especially the 100% speed test column with `col_index=5`. Diagnostic summary: for BPFO-3 errors, `col5@100%` accounts for 48/50 E4 STFT errors, 40/51 DWT plain errors, and 43/48 envelope errors. The test split contains only `col_index=5` for `bearing bpfo 3` at 100% speed.

## Share Package

`ProjectShare\` is the clean private-repo export folder for sharing code and results without exposing raw or regenerated bulk data. Friends should inspect:

- `index.html` for the result dashboard, leaderboard, diagnostics, and links.
- `code_graph_explainer.html` plus `graphify-out\GRAPH_REPORT.md` for codebase structure.
- `Scripts\`, `Scripts_4\`, `requirements.txt`, and project guidance docs for reproducibility.
- `Outputs\4class\training\` for configs, checkpoints, logs, metrics, plots, leaderboards, and misclassification CSVs.
- `Outputs\4class\diagnostics\` and `Outputs\4class\v2_speed_strat\{splits.csv,split_report.txt}` for the canonical split and BPFO-3 diagnostic evidence.
- `sample_gallery\` for exactly 48 representative images: 4 representations x 4 classes x 3 speeds.

Current state: `ProjectShare\` was created on 2026-04-30 as a clean local Git repo with a committed export. The remote private GitHub repo `motor-fault-diagnosis-research-share` still needs to be created/pushed from that folder when GitHub CLI or credentials are available.

Intentionally excluded from `ProjectShare\`: `Dataset\`, `Denoised\`, full feature `.pt` tensors from `Outputs\4class\v2_speed_strat\`, full STFT/DWT/envelope/Clarke image forests, and `.code-review-graph\` internal databases. `.rtk\filters.toml` is only a local Claude/RTK terminal-output filter stub, not a research artifact.

Code-review-graph status for the share package: the existing graph reports 32 files, 417 nodes, 7,815 edges, and 0 embeddings, last updated 2026-04-26 01:29:30. A full rebuild attempt on 2026-04-30 timed out after the MCP 120s limit, so public architecture claims should rely on the included caveat and graphify report until the graph is rebuilt successfully.

---

## Vibration Pipeline

Complete and archived in `Outputs\vibration\`. Achieved 99.91 % test accuracy (5-channel STFT → VGG-style, 45 segments/column). **Not being extended** — do not write new vibration scripts.

---

## Python Dependencies

No `requirements.txt`. Scripts use: `torch` (CUDA + AMP + mmap), `numpy`, `pandas`, `scipy` (`stft`, `butter`, `filtfilt`, `sosfilt`), `pywavelets` (`wavedec`), `librosa` (Mel filterbank, Slaney norm), `matplotlib` (Agg backend), `scikit-learn` (metrics), `pathlib`.

---

## Hard Invariants — Do Not Break

1. **`Dataset\` is read-only.** No script ever writes into it.
2. **Splits are column-level.** Segment-level splits cause ~20 pp inflated accuracy due to the 75 % overlap leakage. Do not change this.
3. **v2 speed-stratified splits are canonical.** Every (class × speed) group must appear in train, val, and test.
4. **Seed 42 everywhere** — `torch.manual_seed(42)`, `np.random.seed(42)`, and per-class `np.random.default_rng(SEED + hash(cls) % 10_000)` for splits.
5. **FAIL files are skipped** in all steps. PASS and WARN are processed.
6. **DWT cD1 is always dropped** — its 5–10 kHz band sits above the 5 kHz denoising cutoff and carries only filter leakage.
7. **DWT normalization is per-level** (not global per-segment). Required because sub-band energies span orders of magnitude.
8. **STFT PNGs and training tensors use different scales intentionally** — PNGs are raw dB for human inspection; `.pt` files are z-scored for training convergence. Do not "fix" this divergence.
9. **Vibration is done.** No new vibration scripts.

---

## Mistakes to Avoid

- **Declaring results before running** — write the script, run it, then report numbers.
- **Vague file/folder names** — use structured, descriptive names: `DWT_multibranch_plain_v1`, `precompute_stft_v2.py`. Not `test_model`, `run_new`, `script2`.
- **Breaking existing script conventions** — before writing a new script in `Scripts_4\`, read a peer script (e.g. `precompute_stft_v2.py` before a new precompute script). Match its structure: manifest loading, verdict filtering, per-file logging, output-dir argument.
- **Healthy subsampling** — hurts performance. Do not reintroduce it unless explicitly testing that hypothesis.
- **Raising batch size without checking VRAM** — 6 GB VRAM. `E4_mod_alexnet_227` at batch 64 is already pushing it on 227×227; comment in script says "drop to 32 if OOM".
- **Using segment-level splits** — see invariant 2 above.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
