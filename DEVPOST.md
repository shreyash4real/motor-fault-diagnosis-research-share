# Motor Hakeem — Devpost submission kit

## One-line pitch

Motor Hakeem helps maintenance teams read induction-motor health from the three-phase current they can already measure.

## Project description

Induction motors sit behind an enormous share of industrial work. They are rugged, but they still run for long hours under non-ideal loading, supply, and environmental conditions. When faults develop, the cost is not only a repair; it is lost efficiency, unplanned downtime, and uncertainty about which machine needs attention first.

I am an electrical engineer, so I started with the practical measurement question: what can a plant capture cheaply and consistently? Three-phase motor current is already part of the system. Motor Hakeem turns a short current recording into three complementary diagnostic views—STFT, DWT, and envelope spectrum—then combines specialized classifiers to distinguish a healthy baseline from stator winding, bearing, and broken-rotor-bar fault patterns.

The result is a product-style evidence explorer rather than a black-box score: it shows the selected ensemble, stored metrics, confusion matrix, per-class performance, and individual sample probabilities. It is designed for a future pilot workflow where a plant supplies current data and a maintenance engineer receives a clear fault indication plus the evidence needed to decide what to inspect next.

I used Codex to turn an electrical-engineering concept and a research pipeline into a polished, runnable product demo. Codex helped me trace the pipeline, connect committed evaluation artefacts to the interface, validate the export path, correct product claims, and shape the story for a real maintenance audience. I made the important domain decisions: current as the sensing modality, source-level split before overlapping windows, complementary signal representations, and decision support rather than autonomous maintenance action.

## What makes it credible

- The demo uses committed evaluation artefacts, rather than invented UI numbers.
- It reports both the 99.85% scoped result and the 96.71% full-split result; the latter exposes a real, sensor-confounded BPFO-3-at-100% operating point.
- The repository retains the complete historical scripts and output bundle, while the product interface keeps the two evaluation scopes visibly separate.
- The interface labels the product as a controlled, current-sensing evidence explorer—not field-wide validation or live browser inference.
- It exposes errors and per-class performance, including the measurement boundary, instead of presenting a single headline metric.

## Suggested track

**Work and productivity** — this is a maintenance decision-support workflow for industrial teams.

## Three-minute demo video outline

1. **0:00–0:25 — Problem.** “I am an electrical engineer. Induction motors are everywhere, but a fault can quietly become energy loss and downtime. I wanted a lower-friction monitoring route built around three-phase current.”
2. **0:25–0:55 — Product.** Open Motor Hakeem. Load the verified reference case and explain that a production pilot would accept a plant current recording; the public demo uses committed evidence so its claims are inspectable.
3. **0:55–1:35 — Method.** Show the locked acquisition and signal-processing settings. Explain STFT, DWT, and envelope as complementary views of the same current signal.
4. **1:35–2:15 — Evidence.** Select the three-view temperature-calibrated ensemble. Show its stored metrics, then explain the model card: the full split found a real BPFO-3-at-100% sensing ambiguity, so the scoped result is reported as an operating envelope rather than a universal claim.
5. **2:15–2:45 — Codex story.** Show the repository, the export script, and the product interface. Explain how Codex helped turn the engineering workflow into a demo without a separate frontend, backend, ML, and pipeline team.
6. **2:45–3:00 — Business path.** “The next step is a pilot with plant current data, calibration against the target motor fleet, and a report that helps a maintenance engineer decide what to inspect.”

## Before submitting

- [ ] Record and publish the public video (under 3 minutes, with working product shown and audible Codex/GPT-5.6 explanation).
- [ ] Add the video URL to the Devpost form.
- [ ] Add the repository URL and confirm it is public.
- [ ] Add a license selected by the repository owner.
- [ ] Run the local demo and exporter one final time.
- [ ] Capture the required Codex `/feedback` session ID and paste it into the submission.
- [ ] Replace any draft wording in the Devpost form with the final, truthful product scope above.
