# CurrentGuard model card

## Intended use

CurrentGuard is a research prototype for maintenance decision support from a short, three-phase induction-motor current recording. It is intended to surface evidence for a healthy baseline, stator short, outer-race bearing defect (BPFO-3), or broken rotor bar. It is not an autonomous maintenance decision-maker.

## Measurement boundary discovered in real data

This project uses real motor-current recordings, not simulated signals. That matters: a physically labelled fault does not guarantee that the available sensing chain exposes a separable signature at every operating point.

In the original full test split, one held-out BPFO-3 source column at 100% speed (`col_index=5`) was consistently predicted as healthy. It contributed 44 of the 45 errors in the three-view temperature-calibrated ensemble. The archived diagnostics show that the same column dominated BPFO errors across multiple representations and model families; it is therefore treated as a measurement-confounded operating point, not silently discarded data.

## CNN capability finding

This is a useful result about the boundary of a vision-style CNN on signal representations. STFT, DWT, and envelope transforms give a CNN different views of the same measured current; a larger model can learn a stronger mapping only when at least one view contains a repeatable discriminative signature. Repeated failure across those views and model families is evidence that the distinction was not reliably observable through this sensing chain at that operating point.

It does **not** mean that bearing faults are universally undetectable from current, or that a CNN is categorically inadequate. It means this particular motor, current measurement setup, and operating condition need either additional calibration data, a different sensing modality, or an abstention/escalation decision rather than a forced four-class label.

## Evaluation results

| Evaluation scope | Test windows | Accuracy | Macro-F1 | BPFO-3 F1 | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| Full original split | 1,368 | 96.71% | 95.60% | 84.9% | Includes the confounded BPFO-3-at-100% source column. |
| Declared bounded operating envelope | 1,311 | 99.85% | 99.75% | 99.1% | Excludes that source column from test and is valid only for this stated sensing/operating scope. |

The second figure is not a replacement for the first. It answers a narrower engineering question: how the current-signal pipeline behaves where this measurement setup can distinguish the operating states.

## Product implication

A production deployment must treat this as an operating-envelope and data-quality problem:

1. Validate that a target motor, speed range, current-transducer setup, and recording quality are represented by the calibration data.
2. Return an **insufficiently separable / escalate for inspection** result when the signal falls outside that validated envelope.
3. Use an additional sensing modality or a targeted inspection when the current signal alone cannot separate fault from healthy operation.
4. Recalibrate and report performance per plant and operating regime before claiming fleet-wide performance.

The public demo currently explores stored results; it does not perform live inference or implement this future abstention gate.

## Reproducibility trail

- The full-split ensemble result, all historical configurations, and BPFO diagnostics are restored under `research_archive/full_split_v2/` from commit `ac41fa5`.
- `research_archive/full_split_v2/README.md` documents the full-split scope and portable archived-script paths without rewriting historical result metadata.
- The scoped evaluation artefacts are under `Outputs_nobpfo100/`.
- `Scripts_nobpfo100/validate_experiment.py` checks source-level split isolation and class/speed test coverage.
- `Scripts_nobpfo100/generate_splits_nobpfo100.py` now requires the scoped BPFO-3-at-100% exclusion to be selected explicitly and writes its scope to `split_manifest.json`.
