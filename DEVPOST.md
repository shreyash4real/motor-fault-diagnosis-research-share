# Motor Current Analytics — Devpost submission kit

This is the single source of truth for the OpenAI Build Week submission. The product is a static evidence explorer, not a browser-based live classifier; keep that scope intact in the form and the video.

## Submission facts

| Field | Final value |
| --- | --- |
| Project title | **Motor Current Analytics** |
| Tagline | **Current-based fault diagnostics for induction motors** |
| Track | **Work and productivity** |
| Live demo | https://shreyash4real.github.io/motor-fault-diagnosis-research-share/ |
| Public repository | https://github.com/shreyash4real/motor-fault-diagnosis-research-share |
| License | MIT — see `LICENSE` |
| Deadline | **Wednesday, July 22, 2026, 5:30 AM IST** (Tuesday, July 21, 5:00 PM PDT) |

The deadline conversion is from the [official challenge page](https://openai.devpost.com/). Do not wait for the last minute: video processing, a form save failure, or a deployment delay should not be the reason a finished project misses submission.

## Paste-ready project description

Induction motors do a huge share of industrial work. They are rugged, but they operate for long hours under non-ideal loading, supply, and environmental conditions. When a fault develops, the cost is not only a repair: it can mean lower efficiency, unplanned downtime, and uncertainty about which machine needs attention first.

I am an electrical engineer, so I started with a practical question: what can a plant measure cheaply and consistently? Three-phase motor current is already present in the system. **Motor Current Analytics** turns a short current recording into three complementary diagnostic views—STFT, DWT, and envelope spectrum—and combines specialized classifiers to surface a healthy baseline or likely stator-winding, bearing, or broken-rotor-bar fault pattern.

The result is deliberately a product-style evidence explorer, not a black-box score. It lets a maintenance team inspect the selected ensemble, stored metrics, confusion matrix, per-class performance, and individual sample probabilities. The public demo uses committed evaluation evidence so reviewers can inspect exactly what the prototype measured; a future plant pilot would ingest calibrated current recordings and return a maintenance report for an engineer to review.

I used Codex with GPT-5.6 to turn an electrical-engineering concept and an existing research pipeline into a polished, runnable product demo. Codex helped me trace the pipeline, connect committed evaluation artefacts to the interface, validate the export path, correct product claims, and shape the work into a coherent product story. The key domain decisions remained mine: use current as the sensing modality; split by source before overlapping segmentation; retain complementary signal representations; expose held-out evidence rather than a single accuracy number; and frame the result as decision support rather than autonomous maintenance approval.

## What makes it credible

- The interface reads committed evaluation artefacts; it does not invent dashboard numbers.
- The declared current-sensing operating envelope reports **99.85% accuracy** and **99.75% macro-F1**.
- The original full split reports **96.71% accuracy** and **95.60% macro-F1**, including a real BPFO-3-at-100% measurement-confounded source group.
- The project exposes that boundary: 44 of 45 errors in the best historic ensemble arose from that source group. A production system must escalate instead of forcing a confident label outside the validated envelope.
- The repository contains the runnable static demo, setup instructions, MIT license, current pipeline, and separated historical evidence archive.

## How Codex and GPT-5.6 were used

Use this answer if the form has a dedicated AI-usage field:

> I used Codex with GPT-5.6 as an implementation partner across the project: it helped me map and review the research repository, make the archived scripts portable, trace evaluation outputs into a deterministic frontend export, build and refine the product interface, test the static demo, and tighten the evidence and limitation language. I directed the electrical-engineering problem framing and the decisions about sensing, source-level evaluation, operating scope, and maintenance workflow. The result is not generated data or a fabricated interface: the demo presents stored outputs from the real current-signal evaluation pipeline.

## Three-minute video script

Record a public YouTube video under three minutes. Show the live demo working and say **Codex** and **GPT-5.6** aloud.

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:22 | You on camera, then an induction-motor image or the Motor Current Analytics landing view | “I’m an electrical engineer. Induction motors run a huge amount of industry, and a fault can quietly become energy loss and downtime. I wanted a lower-friction monitoring path built around the three-phase current already available at the motor.” |
| 0:22–0:48 | The live demo’s hero and product flow | “This is Motor Current Analytics: current-based fault diagnostics for induction motors. This public version is an evidence explorer. A production pilot would accept a calibrated current recording; this demo lets you inspect the committed diagnostic evidence behind the result.” |
| 0:48–1:18 | Acquisition and representation sections | “The pipeline turns three-phase current into STFT, DWT, and envelope-spectrum views. They give the model complementary ways to inspect the same physical signal.” |
| 1:18–1:52 | Results selector, 99.85% / 99.75% configuration, confusion matrix | “Within the declared current-sensing operating envelope, this three-view ensemble reports 99.85% accuracy and 99.75% macro-F1. The interface exposes the configuration, confusion matrix, per-class metrics, and sample probabilities instead of hiding behind one score.” |
| 1:52–2:16 | Measurement-boundary evidence / model card | “The full original split also matters: it includes a real bearing-fault source group at 100% speed that current alone could not reliably separate from healthy. I kept that result visible. In production, that is an escalate-for-inspection case, not a forced diagnosis.” |
| 2:16–2:42 | Repository README, exporter, and product page | “I used Codex with GPT-5.6 to turn the electrical-engineering workflow into this reviewable product: tracing the research pipeline, linking committed outputs to the frontend, validating the exporter, and shaping a credible demo. I made the engineering choices about current sensing, source-level evaluation, and product scope.” |
| 2:42–2:58 | Return to product overview | “The business path is a pilot with plant current data, calibration for the target motor fleet, and a maintenance report that helps engineers decide what to inspect next. This is Motor Current Analytics.” |

Before publishing, play the exported video once and confirm: the project visibly works, the spoken audio says both Codex and GPT-5.6, it is public, and its duration is under 3:00.

## Submission preflight

Run these commands from the repository root before submitting:

```bash
python3 -m compileall -q Scripts_nobpfo100
python3 Scripts_nobpfo100/export_frontend_results.py
python3 -m json.tool frontend/data/results-data.json >/dev/null
python3 -m http.server 8080 --directory frontend
```

Then open `http://localhost:8080` and walk through the product once.

## Final form checklist

- [ ] Select **Work and productivity**.
- [ ] Paste the title, tagline, and project description above.
- [ ] Use the live-demo and repository URLs from the submission facts table.
- [ ] Add the public YouTube video URL after the video is fully processed.
- [ ] Confirm the repository remains public and the MIT `LICENSE` file is visible.
- [ ] Run the preflight commands and inspect the local product flow.
- [ ] Type `/feedback` in the Codex conversation used for the majority of core project work, copy its session ID, and paste that ID in the Devpost form.
- [ ] Re-read the published Devpost preview: it must say **Motor Current Analytics**, never claim live browser inference, and never omit the operating-envelope limitation.
- [ ] Submit before **5:30 AM IST on Wednesday, July 22, 2026**.

The YouTube publication and `/feedback` session ID require the repository owner to perform them; they cannot be generated truthfully from the repository.
