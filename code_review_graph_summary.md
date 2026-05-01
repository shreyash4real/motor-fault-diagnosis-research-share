# Code-Review-Graph Summary

Snapshot used for this share package.

| Field | Value |
|-------|-------|
| Files | 32 |
| Total nodes | 417 |
| Total edges | 7,815 |
| Languages | python |
| Last updated | 2026-04-26T01:29:30 |
| Embeddings | 0 nodes embedded |

## Nodes By Kind

| Kind | Count |
|------|-------|
| Class | 35 |
| File | 32 |
| Function | 350 |

## Edges By Kind

| Edge | Count |
|------|-------|
| CALLS | 7,099 |
| CONTAINS | 386 |
| IMPORTS_FROM | 295 |
| INHERITS | 35 |

## Caveat

A full rebuild was attempted on 2026-04-30, but the MCP call timed out after the 120 second tool limit. The numbers above are the existing graph snapshot, not a freshly rebuilt public map.

The 32-file / 417-node snapshot also **predates the envelope branch additions** (`precompute_envelope_v1.py`, `precompute_envelope_stft_v1.py`, `train_envelope_cnn_v1/v2/v3*.py`, `train_envelope_stft_cnn_v1.py`, `diagnose_bpfo3_v1.py`, `diagnose_envelope_time_domain.py`) committed after 2026-04-26. The current `Scripts_4/` directory contains roughly 8 more Python files than this snapshot reports.

Use `graphify-out/GRAPH_REPORT.md` and this caveat together until code-review-graph is rebuilt successfully.
