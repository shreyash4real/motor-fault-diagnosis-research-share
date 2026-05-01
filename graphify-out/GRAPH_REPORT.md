# Graph Report - ProjectShare  (2026-05-01)

## Corpus Check
- 39 files · ~264,592 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 606 nodes · 937 edges · 31 communities detected
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 51 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 31|Community 31]]

## God Nodes (most connected - your core abstractions)
1. `ModifiedAlexNet` - 17 edges
2. `EnvelopeDilated1D` - 17 edges
3. `ModifiedAlexNet` - 17 edges
4. `process_split()` - 11 edges
5. `process_split()` - 11 edges
6. `process_split()` - 11 edges
7. `process_split()` - 11 edges
8. `process_split()` - 11 edges
9. `part2_column_signatures()` - 10 edges
10. `process_split()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `build_model()`  [INFERRED]
  Scripts_4\compute_logits.py → Scripts_4\train_stft_freqnet_v2.py
- `main()` --calls--> `build_model()`  [INFERRED]
  Scripts_4\compute_val_softmax.py → Scripts_4\train_stft_freqnet_v2.py
- `process_split()` --calls--> `load_split_data()`  [INFERRED]
  Scripts_4\compute_logits.py → Scripts_4\ensemble_evaluate_v1.py
- `main()` --calls--> `load_split_data()`  [INFERRED]
  Scripts_4\compute_logits.py → Scripts_4\ensemble_evaluate_v1.py
- `main()` --calls--> `load_split_data()`  [INFERRED]
  Scripts_4\compute_val_softmax.py → Scripts_4\ensemble_evaluate_v1.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (52): infer_logits(), main(), process_split(), COMPUTE LOGITS (one-time helper for temperature calibration) ===================, Mirror of infer_softmax but returns raw (N, C) float32 logits., write_meta_sigs(), main(), COMPUTE VAL SOFTMAX (one-time helper) ====================================== For (+44 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (24): bands_to_device(), build_model(), compute_class_weights(), DWTMultiBranchCNN, DWTTensorDataset, fmt_time(), load_pt_dataset(), main() (+16 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (16): BasicBlock, build_model(), compute_class_weights(), fmt_time(), load_pt_dataset(), main(), ModifiedAlexNet, ModifiedResNet (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (28): cosine_sim(), energy_in_band(), envelope_spectrum(), find_misclass_file(), get_column_indices_per_split(), linear_spectrum(), list_solo_runs(), load_misclass_long() (+20 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (20): build_model(), compute_class_weights(), fmt_time(), load_pt_dataset(), main(), plot_confusion(), plot_training_curves(), STEP 5 — STFT CNN TRAINING — STFTFreqNet v2 =================================== (+12 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (13): build_model(), compute_class_weights(), fmt_time(), load_pt_dataset(), main(), ModifiedAlexNet, plot_confusion(), plot_training_curves() (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (11): EnvelopeResNet1D, EnvelopeTensorDataset, load_pt_dataset(), main(), STEP 5 — ENVELOPE-SPECTRUM 1D RESNET TRAINING (v2 Refinement) =================, Squeeze-and-Excitation for 1D: channel-wise attention., Residual block: [Conv-BN-GELU-Conv-BN] + SE + Shortcut., Refined 1D ResNet for Envelope Spectrum.     ~450k parameters. (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (11): calibrate_temperature(), EnvelopeResNet1D, EnvelopeTensorDataset, load_pt_dataset(), main(), plot_confusion(), plot_training_curves(), STEP 5 — ENVELOPE-SPECTRUM 1D RESNET TRAINING (v3 + Full Research Artifacts) == (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (20): clarke_transform(), compute_clarke_density(), fmt_size(), fmt_time(), load_full_three_phases(), main(), process_split(), STEP 4 — PRECOMPUTE CLARKE IMAGES + PNG TREE (4-class) ========================= (+12 more)

### Community 9 - "Community 9"
Cohesion: 0.16
Nodes (12): build_model(), compute_class_weights(), EnvelopeCNN, fmt_time(), load_pt_dataset(), main(), plot_confusion(), plot_training_curves() (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.16
Nodes (19): compute_spectrogram_db(), fmt_size(), fmt_time(), load_full_all_channels(), main(), process_split(), STEP 4 — PRECOMPUTE STFT FEATURES + PER-CHANNEL PNG TREE (4-class) =============, bearing bpfo 3' → 'bearing_bpfo_3' (Windows-safe folder name). (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (18): compute_dwt_per_channel(), fmt_size(), fmt_time(), get_level_info(), load_full_three_phases(), main(), process_split(), STEP 4 — PRECOMPUTE DWT COEFFICIENTS + PNG TREE (4-class) ====================== (+10 more)

### Community 12 - "Community 12"
Cohesion: 0.17
Nodes (18): compute_envelope_spectrum(), fmt_size(), fmt_time(), load_full_three_phases(), main(), process_split(), STEP 4 — PRECOMPUTE ENVELOPE-SPECTRUM FEATURES + PER-SEGMENT PNG (4-class) =====, bearing bpfo 3' -> 'bearing_bpfo_3' (Windows-safe folder name). (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (18): compute_spectrogram_db(), fmt_size(), fmt_time(), load_full_three_phases(), main(), process_split(), STEP 4 — PRECOMPUTE STFT FEATURES + PER-CHANNEL PNG TREE (4-class) =============, bearing bpfo 3' → 'bearing_bpfo_3' (Windows-safe folder name). (+10 more)

### Community 14 - "Community 14"
Cohesion: 0.17
Nodes (18): compute_spectrogram_db(), fmt_size(), fmt_time(), load_full_three_phases(), main(), process_split(), STEP 4 — PRECOMPUTE STFT FEATURES + PER-CHANNEL PNG TREE (4-class) =============, bearing bpfo 3' → 'bearing_bpfo_3' (Windows-safe folder name). (+10 more)

### Community 15 - "Community 15"
Cohesion: 0.16
Nodes (11): build_model(), compute_class_weights(), fmt_time(), load_pt_dataset(), main(), plot_confusion(), plot_training_curves(), STEP 5 — STFT CNN TRAINING (configurable architecture) ======================== (+3 more)

### Community 16 - "Community 16"
Cohesion: 0.16
Nodes (12): compute_class_weights(), fmt_time(), load_pt_dataset(), main(), plot_confusion(), plot_training_curves(), STEP 5 (v2) — STFT CNN TRAINING, configurable for multiple runs ===============, Load one .pt file. Optionally subsample the healthy class (train only). (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.16
Nodes (12): compute_class_weights(), fmt_time(), load_pt_dataset(), main(), plot_confusion(), plot_training_curves(), STEP 5 (v2) — STFT CNN TRAINING, configurable for multiple runs ===============, Load one .pt file. Optionally subsample the healthy class (train only). (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (12): compute_class_weights(), fmt_time(), load_pt_dataset(), main(), plot_confusion(), plot_training_curves(), STEP 5 (v2) — STFT CNN TRAINING, configurable for multiple runs ===============, Load one .pt file. Optionally subsample the healthy class (train only). (+4 more)

### Community 19 - "Community 19"
Cohesion: 0.16
Nodes (12): build_model(), compute_class_weights(), fmt_time(), load_pt_dataset(), main(), plot_confusion(), plot_training_curves(), STEP 5 — STFT CNN TRAINING — STFTFreqNet v1 =================================== (+4 more)

### Community 20 - "Community 20"
Cohesion: 0.17
Nodes (11): build_model(), compute_class_weights(), fmt_time(), load_pt_dataset(), main(), plot_confusion(), plot_training_curves(), STEP 5 — STFT CNN TRAINING (configurable architecture) ======================== (+3 more)

### Community 21 - "Community 21"
Cohesion: 0.21
Nodes (11): Dataset, build_model(), compute_class_weights(), EnvelopeTensorDataset, fmt_time(), load_pt_dataset(), main(), plot_confusion() (+3 more)

### Community 22 - "Community 22"
Cohesion: 0.29
Nodes (12): compute_envelope_stft_db(), fmt_size(), fmt_time(), load_full_three_phases(), main(), process_split(), STEP 4 — PRECOMPUTE ENVELOPE STFT FEATURES + PER-CHANNEL PNG TREE (4-class) ===, 1. Detrend signal     2. Hilbert -> Magnitude (Envelope)     3. Detrend Envelo (+4 more)

### Community 23 - "Community 23"
Cohesion: 0.27
Nodes (10): carry_over_verdicts(), count_signal_columns(), find_motor2_root(), get_channel_name(), main(), STEP 0 — PC SETUP (one-time per machine) =======================================, Join new manifest with old validated manifest on (speed_pct, fault_label,     ch, Walk the tree under `root` and return the first folder named Motor-2. (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.29
Nodes (9): build_filter(), denoise_signal(), main(), mirror_path(), process_file(), STEP 2 — DENOISING (manifest-driven) ==================================== Reads, Design the Butterworth bandpass once; reuse across all files., Bandpass a single column. Skip flat/NaN columns (preserve unchanged). (+1 more)

### Community 25 - "Community 25"
Cohesion: 0.29
Nodes (9): count_signal_columns(), get_channel_name(), main(), STEP 1 — VIBRATION DATASET SCANNER =================================== Walks the, Return human-readable summary lines., Extract channel id from filename, e.g. 'Vibration_Motor-2_50_time-healthy 1-ch1., Return number of signal columns (total columns minus timestamp column)., scan() (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.31
Nodes (8): bandpass_filter(), denoise_file(), main(), mirror_path(), STEP 3 — VIBRATION DENOISING (manifest-driven) =================================, Map input_path under INPUT_ROOT to the same relative path under OUTPUT_ROOT., 4th-order Butterworth zero-phase bandpass filter, SOS form.      lowcut  =    6, Read one vibration CSV, bandpass-filter every signal column, save to mirror path

### Community 27 - "Community 27"
Cohesion: 0.38
Nodes (6): allocate_splits_for_group(), build_class_splits(), main(), STEP 3 (v2) — SPEED-STRATIFIED COLUMN SPLITS (4-class) =========================, Given n columns at one (class, speed), return (n_train, n_val, n_test).     Ensu, For one class, split EACH (speed) group independently.

### Community 28 - "Community 28"
Cohesion: 0.5
Nodes (4): main(), STEP 1 — DATASET VALIDATION =========================== Reads manifest.csv produ, Run all checks on a single file; return per-file stats + verdict., validate_file()

### Community 29 - "Community 29"
Cohesion: 0.5
Nodes (4): main(), STEP 2 — VIBRATION DATASET VALIDATION ======================================= Re, Load a vibration CSV and run all quality checks. Returns a dict of results., validate_file()

### Community 31 - "Community 31"
Cohesion: 0.67
Nodes (1): diagnose_envelope_time_domain.py Standalone diagnostic script to visualize the

## Knowledge Gaps
- **140 isolated node(s):** `STEP 2 — DENOISING (manifest-driven) ==================================== Reads`, `Design the Butterworth bandpass once; reuse across all files.`, `Bandpass a single column. Skip flat/NaN columns (preserve unchanged).`, `Map an input file path to the equivalent path under output_root.`, `STEP 0 — PC SETUP (one-time per machine) =======================================` (+135 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 31`** (3 nodes): `main()`, `diagnose_envelope_time_domain.py Standalone diagnostic script to visualize the`, `diagnose_envelope_time_domain.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DWTTensorDataset` connect `Community 1` to `Community 21`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `STFTTensorDataset` connect `Community 2` to `Community 21`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `STFTTensorDataset` connect `Community 4` to `Community 21`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `ModifiedAlexNet` (e.g. with `STEP 6 — ENSEMBLE EVALUATION (4-class) ======================================` and `A per-sample identity tuple used to check alignment across reps.`) actually correct?**
  _`ModifiedAlexNet` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `EnvelopeDilated1D` (e.g. with `STEP 6 — ENSEMBLE EVALUATION (4-class) ======================================` and `A per-sample identity tuple used to check alignment across reps.`) actually correct?**
  _`EnvelopeDilated1D` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `ModifiedAlexNet` (e.g. with `STEP 6 — ENSEMBLE EVALUATION (4-class) ======================================` and `A per-sample identity tuple used to check alignment across reps.`) actually correct?**
  _`ModifiedAlexNet` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `STEP 2 — DENOISING (manifest-driven) ==================================== Reads`, `Design the Butterworth bandpass once; reuse across all files.`, `Bandpass a single column. Skip flat/NaN columns (preserve unchanged).` to the rest of the system?**
  _140 weakly-connected nodes found - possible documentation gaps or missing edges._