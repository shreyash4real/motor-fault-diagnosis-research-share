# CurrentGuard working guide

## Purpose

CurrentGuard is an electrical-engineering-led motor-current diagnostics prototype. It turns short, three-phase induction-motor current recordings into STFT, DWT, and envelope representations for four-class maintenance decision support.

## Start here

1. Read `README.md` for the product and repository map.
2. Read `MODEL_CARD.md` before changing claims, metrics, or evaluation presentation.
3. Read `docs/engineering-reference.md` only for pipeline, data, or reproducibility work.

## Canonical paths

- Product frontend: `frontend/`
- Current research pipeline: `Scripts_nobpfo100/`
- Current stored evaluation: `Outputs_nobpfo100/`
- Historical full-split evidence: `research_archive/full_split_v2/`

Do not treat the historical archive and bounded-scope evaluation as one comparable leaderboard.

## Non-negotiables

- Split at the source-column level before creating overlapping windows.
- The public frontend is a stored-evidence explorer, not live browser inference.
- Report 99.85% / 99.75% only within the declared current-sensing operating envelope.
- Keep the full-split 96.71% / 95.60% baseline visible when discussing the measurement boundary.
- Do not add raw recordings, bulk feature tensors, or unreviewed customer data to the repository.

## Verify a change

```bash
python3 -m compileall -q Scripts_nobpfo100
python3 Scripts_nobpfo100/export_frontend_results.py
python3 -m json.tool frontend/data/results-data.json >/dev/null
```

For frontend-only changes, also verify the JSX transform and serve `frontend/` locally. Keep documentation concise, product-facing, and consistent with the model card.
