# Motor Fault Diagnosis Research Share

## Frontend

[Open the cinematic Motor Fault Detection frontend](https://shreyash4real.github.io/motor-fault-diagnosis-research-share/)

Private share package for the 4-class motor fault diagnosis work.

Pushed to `https://github.com/shreyash4real/motor-fault-diagnosis-research-share` on 2026-04-30.

The tool is scoped to three live signal-representation branches: **STFT**, **DWT**, and **Envelope**. Mel and Clarke were evaluated and dropped — they remain in the leaderboard for honesty, not as candidates.

Open `index.html` for the result dashboard and `code_graph_explainer.html` for the code graph overview.

Included:

- `Scripts/` and `Scripts_4/`
- training outputs under `Outputs/4class/training/`
- BPFO-3 diagnostics under `Outputs/4class/diagnostics/`
- canonical split metadata under `Outputs/4class/v2_speed_strat/`
- representative gallery images under `sample_gallery/` (3 reps × 4 classes × 3 speeds = 36 PNGs)
- `graphify-out/GRAPH_REPORT.md` and `code_review_graph_summary.md`

Excluded intentionally:

- raw `Dataset/`
- denoised bulk data in `Denoised/`
- full feature tensors from `Outputs/4class/v2_speed_strat/`
- full generated STFT/DWT/envelope/Clarke image forests
- `.code-review-graph/` internal databases
- `.rtk/` local terminal-output filter config
