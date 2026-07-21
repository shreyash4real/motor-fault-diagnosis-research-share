# Comprehensive Technical Report: Multi-Representation Motor Fault Diagnosis

## 1. Project Overview & Objectives
The objective of this research pipeline is to develop a robust machine-learning system for **motor fault diagnosis using controlled 3-phase electric-current recordings**.
The system is designed to classify signals into four canonical categories across three variable frequency drive (VFD) speeds (50%, 75%, 100%):
1. `healthy 1` (Baseline)
2. `stator short 1` (Stator winding short)
3. `bearing bpfo 3` (Outer-race bearing defect, severity 3)
4. `broken rotor bar` (Rotor cage fracture)

The central thesis of the pipeline is that no single signal representation captures all fault physics optimally. Therefore, the architecture extracts three distinct physical perspectives—**Time-Frequency (STFT), Time-Scale (DWT), and Demodulated Low-Frequency (Envelope)**—trains bespoke parallel deep-learning models for each, and fuses them via temperature-calibrated soft voting.

## 2. Evaluation corpus & foundational constraints
* **Recordings:** Controlled three-phase motor-current measurements. The raw recording corpus is intentionally not distributed in this share package.
* **Signal Specs:** 15-second recordings, sampled at **20 kHz** ($F_s = 20,000$). Each recording contains 300,000 samples across 3 simultaneous electric phases.
* **Volume:** 231 total electric files.

## 3. Pipeline Step-by-Step Architecture

### Step 1: Data Validation 
* **Process:** Each raw CSV is systematically scanned to ensure physical validity.
* **Parameters:** The validation script verifies exactly 300,000 rows, the absence of all-NaN or flat-zero columns, signal clipping/saturation at < 98%, and RMS current bounds strictly between 0.001 A and 50 A.
* **Output:** A `validated_manifest.csv` containing a `PASS`, `WARN`, or `FAIL` verdict. Failed files are permanently excluded from all downstream steps.

### Step 2: Preprocessing & Denoising
* **Process:** High-frequency noise and extreme low-frequency drift are removed from the raw signals.
* **Parameters:** A **4th-order Butterworth bandpass filter** with cutoff frequencies at **5 Hz and 5,000 Hz**. It is applied using `scipy.signal.filtfilt` to ensure zero-phase distortion.
* **Output:** Filtered signal arrays stored in a parallel, non-destructive `Denoised/` directory structure.

### Step 3: Data Splitting Strategy (Canonical v2)
* **Process:** A deterministic, speed-stratified split routing. 
* **Parameters:** Data is split 70% Train / 15% Validation / 15% Test independently for every `(class, speed)` combination. For groups with only 3 files, exactly 1 file goes to each split.
* **Critical Design Choice:** Splits are explicitly performed at the full 15-second column level *before* segmentation. Segment-level splits cause severe data leakage (artificially inflating accuracy by ~20%) due to overlapping segment windows.

### Step 4: Segmentation & Feature Precomputation
All branches apply the same uniform sliding-window segmentation to the 300,000-sample columns.
* **Window Size:** 20,000 samples (1.0 second).
* **Stride:** 5,000 samples (0.25 seconds) -> **75% Overlap**.
* **Yield:** Exactly 57 segments per 15-second column.

From here, the pipeline branches into three parallel feature extraction methodologies:

#### Representation A: Linear STFT 
* **Physics:** Captures the broad time-frequency spectrum.
* **Parameters:** `scipy.signal.stft`, Hann window, `nperseg = 1024`, `noverlap = 896` (87.5% inner overlap). Frequencies are truncated to a **3,000 Hz cutoff**.
* **Output:** Tensors of shape `(3, 154, 149)`. Normalization is a per-image Z-score computed across all 3 phases simultaneously to preserve inter-phase relative magnitude.

#### Representation B: Discrete Wavelet Transform (DWT)
* **Physics:** Captures multi-resolution transient phenomena at native sequence lengths.
* **Parameters:** Wavelet `db8`, Decomposition Level 10, Symmetric boundary mode.
* **Filtering:** The highest detail band `cD1` (5–10 kHz) is intentionally dropped because it exceeds the 5 kHz denoising cutoff.
* **Output:** 10 sub-bands (`cD2` through `cD10`, plus `cA10`). Because sub-band energies vary by orders of magnitude (e.g., the 50 Hz fundamental in `cD8` dominates), normalization is performed via a **per-level Z-score**. 

#### Representation C: Envelope Time Domain
* **Physics:** Strips the dominant 50 Hz line carrier to expose low-frequency bearing fault (BPFO) modulations.
* **Parameters:** The signal is detrended -> passed through a Hilbert transform to extract the analytic envelope -> absolute magnitude taken -> detrended again -> multiplied by a Hann window -> transformed via Real FFT. 
* **Filtering:** The spectrum is strictly cropped to **0–500 Hz** and compressed using `log1p`.
* **Output:** Tensors of shape `(3, 501)`. Normalized via per-image Z-score.

## 4. Model Architectures & Training Configurations (Step 5)

Each representation is trained on a bespoke deep learning architecture optimized for its specific dimensional characteristics. 

#### Model A: STFT Classifier (Modified AlexNet)
* **Architecture:** Based on Alotaibi et al. (Sensors 2023). Five convolutional layers (`11x11`, `5x5`, `3x3`, `3x3`, `3x3`) progressing from 96 to 256 channels, with BatchNorm applied after every convolution. The fully connected head is replaced by Global Average Pooling (GAP) -> Dropout (0.5) -> Linear classifier.
* **Data Flow:** The `(3, 154, 149)` STFT tensors are bilinearly upsampled to `(3, 227, 227)` inside the `forward()` pass. 
* **Parameters:** ~3.75 Million.

#### Model B: DWT Multi-Branch 1D CNN
* **Architecture:** 10 parallel 1D convolutional branches (one for each wavelet sub-band) operating at the **native sequence length** of the band. 
* **Stem Configuration:** The initial kernel size (`k`) and stride (`s`) scale based on the band length. Large bands (e.g., >2000 length) use `k=15, s=4`. Small bands (e.g., <100 length) use `k=3, s=1`.
* **Fusion:** Each branch reduces to a 48-dimensional vector via GAP. The 10 branches are concatenated (480-d) -> Dense(128) -> GELU -> Dropout(0.4) -> Output.

#### Model C: Envelope 1D CNN variants
* **Historical full-split baseline:** A regularized 1D ResNet v3 using Squeeze-and-Excitation (SE) blocks and dilated convolutions to capture harmonic peaks across the frequency axis. Its configuration includes Dropout (0.5), Weight Decay (1e-3), Label Smoothing (0.1), and Inverse-Frequency Class Weighting.
* **Retrospective scoped analysis:** The selected three-view result instead uses the `envelope_dilated` branch recorded in its ensemble configuration. The full and scoped figures therefore change both evaluation scope and envelope-model selection; they are not a controlled architecture-only comparison.

#### Common Training Hyperparameters
* **Optimizer:** AdamW.
* **Learning Rate:** 1e-3 (or 2e-3 for Envelope), decayed using a `CosineAnnealingLR` schedule.
* **Epochs & Early Stopping:** Max 40-60 epochs, early stopping triggered by validation Macro-F1 plateau (patience = 8 to 12 epochs).

## 5. Fusion & Ensemble Strategy
Rather than hard voting or concatenating features, the pipeline utilizes **Temperature-Calibrated Equal-Weight Soft Voting**.

1. **Logit Extraction:** Pre-softmax logits are extracted for the validation and test sets of all three models.
2. **Temperature Scaling (LBFGS):** For each model independently, a scalar temperature $T$ is optimized to minimize the Cross-Entropy Loss (NLL) on the *validation set*. $T > 1$ softens over-confident models; $T < 1$ sharpens under-confident ones.
3. **Soft Voting:** The calibrated softmax probabilities ($P = \text{softmax}(\text{logits} / T)$) from the STFT, DWT, and Envelope models are averaged with equal weights to produce the final classification.

**Reported ensemble outputs:** The historical full-split ensemble achieves ~96.7% window-level Test Accuracy and ~0.956 Macro-F1 over 24 held-out source columns. A separate retrospective scoped analysis reports 99.85% window-level accuracy and 99.75% Macro-F1 over 23 held-out source columns. Its 1,311 overlapping one-second windows are not 1,311 independent recordings.
