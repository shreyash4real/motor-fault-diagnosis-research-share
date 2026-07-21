# Motor Current Analytics

**Motor-current diagnostics for the induction motors industry already depends on.**

[Open the product demo](https://shreyash4real.github.io/motor-fault-diagnosis-research-share/)

Motor Current Analytics is an electrical-engineering-led prototype for spotting common induction-motor fault patterns from three-phase current—the signal a plant can often acquire more cheaply and conveniently than dedicated vibration instrumentation. It translates current recordings into complementary spectral views and presents the diagnostic evidence in a plant-friendly product flow.

## The product idea

Induction motors are everywhere, run for long hours, and often operate away from ideal loading or supply conditions. Faults can reduce efficiency, raise maintenance risk, and eventually create costly downtime. Motor Current Analytics is designed as the front end of a practical service:

1. A plant supplies a short three-phase current recording from a motor under observation.
2. The diagnostic pipeline validates, filters, segments, and represents the signal as STFT, DWT, and envelope views.
3. An ensemble returns an evidence-backed indication of healthy operation or a likely stator, bearing, or broken-rotor-bar fault pattern.
4. A maintenance engineer uses that result to decide what to inspect next—not as an automatic replacement for a human diagnosis.

The public demo is deliberately an **evidence explorer**, not a live production inference endpoint. It ships committed, scoped evaluation results and representative signal views so reviewers can inspect exactly what the prototype measured without exposing raw recordings or claiming that a browser upload is being classified.

## What is implemented

- A polished static product demo with a four-step condition-monitoring journey.
- Three current-derived representations: short-time Fourier transform (STFT), discrete wavelet transform (DWT), and envelope spectrum.
- Four classes: healthy baseline, stator winding short, outer-race bearing defect, and broken rotor bar.
- A deliberately separate historical full-split baseline and four comparable scoped ensemble configurations, with per-class metrics, confusion matrices, probability views, and sample-level exploration.
- A reproducible export script that regenerates the frontend evidence bundle entirely from committed evaluation artefacts.

The displayed three-view configuration reports **99.85% accuracy** and **99.75% macro-F1** in a retrospective scoped analysis of **23 held-out source columns** (1,311 overlapping one-second windows). The full original split reports **96.71% accuracy** and **95.60% macro-F1** over 24 held-out source columns. Both belong in the story: the first is a deliberately bounded analysis; the second is the broader controlled evaluation and includes a condition the tested current-only models did not reliably distinguish. Neither result is field or fleet validation. See [MODEL_CARD.md](MODEL_CARD.md).

## Run it locally

The product demo is static. No install is required to view it; serve `frontend/` so the browser can fetch the bundled result JSON.

```bash
python3 -m http.server 8080 --directory frontend
```

Then open `http://localhost:8080`.

To regenerate the committed frontend data bundle from the stored output artefacts:

```bash
python3 Scripts_nobpfo100/export_frontend_results.py
```

To work with the research and training scripts, install the pinned packages:

```bash
python3 -m pip install -r requirements.txt
```

The raw recordings, bulk denoised signals, and feature tensors are intentionally not part of this share package. The historical full-split result archive retains its original checkpoints and logits as reproducibility evidence; it is not a live deployment bundle.

## Repository map

- `frontend/` — GitHub Pages product demo and its committed evaluation bundle.
- `Scripts_nobpfo100/export_frontend_results.py` — deterministic exporter for `frontend/data/results-data.json`.
- `Scripts_nobpfo100/` — training, feature-precomputation, split, and ensemble-evaluation scripts.
- `Outputs_nobpfo100/training/` — committed metrics and prediction artefacts used by the demo.
- `research_archive/full_split_v2/` — historical scripts and the complete full-split v2 evidence bundle, clearly separated from the scoped evaluation.
- `frontend/docs/motor_fault_diagnosis_report.md` — technical methodology and evaluation notes.
- `MODEL_CARD.md` — intended use, evaluated operating scope, and known sensing limitation.
- `DEVPOST.md` — submission copy, demo-video outline, and final checklist.
- `docs/engineering-reference.md` — deeper pipeline and evaluation context; not required for routine product work.

## Built with Codex and GPT-5.6

I am an electrical engineer who framed the motor and fault-physics problem, then used Codex with GPT-5.6 as an implementation partner to turn it into a reviewable product experience. Codex accelerated the work of tracing the research pipeline, validating the evaluation-to-frontend export path, tightening the product claims, and shaping a coherent demo from the existing artifacts.

The key engineering decisions remained explicit: use current rather than add-on vibration sensing as the product input; keep source-level splits ahead of overlapping segmentation to avoid leakage; preserve three complementary signal views; expose held-out evidence rather than hiding behind a single accuracy number; and present the result as decision support, not autonomous maintenance approval.

The repository contains pre-existing research work and Build Week productization work. The specific cutover, observable Codex/GPT-5.6 contribution, verification approach, and required submission evidence are documented in [docs/codex-gpt-5-6-process.md](docs/codex-gpt-5-6-process.md).

See [DEVPOST.md](DEVPOST.md) for the submission-ready project description, video structure, and the final items that must be completed in the Devpost form.
