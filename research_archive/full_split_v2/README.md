# Full-split v2 research archive

This directory restores the complete historical full-split experiment from commit `ac41fa5` (2026-05-01). It is intentionally separate from `Outputs_nobpfo100/`: the two evaluations answer different questions and must never be combined into one leaderboard.

## What this archive records

- `results/` is the original 276-file result bundle: checkpoints, logits, metrics, plots, split files, diagnostics, and all historical configurations (about 55 MiB).
- `scripts/` is the historical v2 preprocessing, training, logit, ensemble, and diagnosis code. Its paths have been made portable; feature extraction and model logic are otherwise preserved.
- The representative full-split run is `results/training/ENSEMBLE_stft_dwt_envelope_v3_temperature/`.

That representative run evaluates 1,368 overlapping test windows and reports 96.71% accuracy / 95.60% macro-F1. It contains the BPFO-3-at-100% held-out source column (`col_index=5`), which accounts for 44 of its 45 errors. The diagnosis is evidence of a current-sensing boundary at that operating point, not a basis for hiding the data.

## How it relates to the current result

`Outputs_nobpfo100/` is the later declared operating-envelope evaluation: 1,311 windows and 99.85% accuracy / 99.75% macro-F1 for the three-view temperature-calibrated ensemble. It explicitly excludes the measurement-confounded BPFO-3-at-100% test source group. See the root [MODEL_CARD.md](../../MODEL_CARD.md) for the comparison and deployment boundary.

The product frontend displays the historical baseline and the scoped result side by side, but lets a reviewer select only the four configurations that share the scoped evaluation split. The complete historical leaderboard remains available in `results/training/`.

## Portable archived scripts

The original scripts embedded a local Windows project path. They now map those paths through `scripts/pipeline_paths.py`; no data path needs to be edited in source. By default, fresh outputs go under `reproduced_outputs/` rather than overwriting the preserved `results/` evidence.

Set these input roots when reproducing preprocessing or training:

```bash
export MFDS_ARCHIVE_RAW_ROOT=/path/to/Motor-2
export MFDS_ARCHIVE_DENOISED_ROOT=/path/to/Denoised/Motor-2
export MFDS_ARCHIVE_OUTPUT_ROOT=/path/to/full-split-v2-output
```

For example, from `research_archive/full_split_v2/scripts/`:

```bash
python3 precompute_stft_v2.py
python3 precompute_dwt_v1.py
python3 precompute_envelope_v1.py
python3 train_stft_cnn_e4_modalex_227.py
python3 train_dwt_cnn.py
python3 train_envelope_cnn_v3_regularized.py
python3 compute_logits.py
python3 ensemble_evaluate_v2_temperature.py
```

Generating the historical split also needs `MFDS_ARCHIVE_VALIDATED_MANIFEST` to point to the validated manifest. The raw recordings and bulk feature tensors are not in this share package, so a full rerun still requires the authorised source data and a suitable PyTorch environment.

## Historical paths inside evidence files

Some saved CSV metadata and logs still show `C:\\Project Work`. Those files are immutable historical evidence; rewriting their embedded paths would change provenance without making them portable. Use the paths above for runnable code, and treat embedded locations in `results/` as historical recording locations only.
