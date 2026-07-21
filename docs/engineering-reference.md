# Engineering reference

Use this reference for work on the research pipeline, stored results, or product claims. It is deliberately separate from `AGENTS.md`: most frontend and documentation work does not need this depth.

## System boundary

Motor Current Analytics is a research prototype for maintenance decision support from three-phase induction-motor current. Its public product is a static evidence explorer; it does not run arbitrary uploaded recordings in the browser.

The four labelled states are healthy operation, stator short, BPFO-3 outer-race bearing defect, and broken rotor bar. The product uses complementary STFT, DWT, and envelope views of the same current signal.

## Evaluation record

| Scope | Test windows | Accuracy | Macro-F1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Full original split | 1,368 | 96.71% | 95.60% | Includes the BPFO-3-at-100% source column. |
| Declared current-sensing envelope | 1,311 | 99.85% | 99.75% | Excludes that measurement-confounded held-out source group. |

The scoped result is not a replacement for the full split. In the full split, 44 of 45 errors in the best three-view temperature-calibrated ensemble came from the BPFO-3-at-100% source column (`col_index=5`). Repeated failure across representations and model families indicates a measurement/identifiability boundary, not a claim that bearing faults are universally invisible to current sensing.

## Data and split invariants

- A 15-second recording is segmented into overlapping 1-second windows with a 0.25-second stride (57 windows per source column).
- Source columns must not appear in more than one split. Window-level random splitting would leak nearly identical signal content.
- The scoped exclusion must be explicit. `Scripts_nobpfo100/generate_splits_nobpfo100.py` writes its declared scope; `validate_experiment.py` validates source isolation and class/speed coverage.
- Raw recordings, denoised bulk signals, and feature tensors are intentionally excluded from the public package.

## Working paths

- `Scripts_nobpfo100/run_all_nobpfo100.py` is the portable pipeline runner.
- `Scripts_nobpfo100/export_frontend_results.py` deterministically regenerates `frontend/data/results-data.json` from committed result artifacts.
- `Outputs_nobpfo100/training/` provides the stored metrics and predictions shown by the frontend.
- `research_archive/full_split_v2/` retains the full historical script/output record. Its legacy metadata paths are evidence, while its restored scripts use environment-configurable paths described in that archive’s README.

## Product language

Describe the product as current-based condition-monitoring evidence that helps a maintenance engineer decide what to inspect next. Do not call it autonomous diagnosis, field-wide validation, universal fault detection, or live browser inference. For the complete intended-use and deployment discussion, use `MODEL_CARD.md`.
