# Codex and GPT-5.6 process record

This is an auditable account of how Codex and GPT-5.6 were used to productize Motor Current Analytics during OpenAI Build Week. It deliberately describes observable work, decisions, commits, and verification—not hidden model reasoning or a claim that the product itself calls an OpenAI model at runtime.

## Plain-language model terminology

“GPT-5.6 Terra + Ultra” combines two controls, not a separate model named `Terra Ultra`:

- **Terra** is the GPT-5.6 variant optimized for fast, efficient exploration, read-heavy scans, broad repository review, and parallel work that returns distilled findings.
- **Ultra** is a higher intelligence/reasoning setting. It allocates more deliberate work to difficult, ambiguous, multi-step tasks and can use appropriate subagents proactively.

In practical terms, Terra is useful for quickly locating relevant files, comparing repository state, and generating concise audit findings. Ultra is useful when the decision requires cross-checking evidence, resolving competing constraints, planning a multi-file change, or deciding what must be disclosed rather than merely finding text.

The exact model/setting for an individual historical turn cannot be reconstructed from Git history alone. The authoritative session evidence is the Codex conversation and its `/feedback` session ID, which must be entered in the Devpost form. Public documentation should not imply that a model setting can be proven from a commit timestamp alone.

Official references: [latest model guide](https://developers.openai.com/api/docs/guides/latest-model) and [Codex manual](https://developers.openai.com/codex/codex-manual.md).

## What Codex did—and what it did not do

Codex was used as an implementation and review partner. Its observable contributions included:

- Mapping a large research repository and identifying the canonical product/evidence path.
- Connecting committed evaluation artifacts to a deterministic frontend export bundle.
- Productizing the static experience for GitHub Pages and iterating its visual language, labels, results controls, and evidence views.
- Auditing split scope, metrics, public claims, and reproducibility boundaries.
- Preparing submission copy, a video outline, verification commands, and this provenance record.
- Running targeted checks after changes: source compilation, exporter execution, JSON validation, and split validation.

The project owner supplied the electrical-engineering problem framing and made the substantive domain decisions: current as the sensing modality; source-level splitting before overlapping segmentation; complementary signal representations; the evaluated scope; the decision-support rather than autonomous-diagnosis boundary; and the commercial pilot direction.

The product has **no OpenAI runtime dependency**. It is a static evidence explorer that reads committed results in the browser. Codex/GPT-5.6 assisted development; it is not the inference engine shown to a plant user.

## Auditable decision process, not private chain-of-thought

It would be inaccurate to present a hidden chain-of-thought as project evidence. Instead, the process can be audited through the following visible loop:

1. Translate an engineering requirement into a bounded repository task.
2. Inspect the relevant code, artifacts, and result metadata before changing claims or UI.
3. Make a small, traceable change or preserve a documented limitation.
4. Run a proportionate check—such as exporter, JSON, split, compilation, or deployed-page validation.
5. Record the outcome in commits, documentation, and the final submission story.

This matters here because the review did not treat the strongest metric as automatically trustworthy. It cross-checked the evaluation unit, preserved the wider baseline, and separated a static evidence view from live plant inference.

## Build Week provenance

This repository contains pre-existing research work. The distinction below is intentional and is required context for the Build Week submission.

| Period | Observable repository state | What should be claimed |
| --- | --- | --- |
| 30 April–15 May 2026 | Early research package, pipeline material, results, and the historical/bounded evaluation work. Examples: [`4dcb19f`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/4dcb19f), [`ac41fa5`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/ac41fa5), [`4d15f80`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/4d15f80). | Pre-existing electrical-engineering research and evaluation artifacts; **not** Build Week work. |
| 18 July 2026 | Productization began: cinematic frontend, root deployment, Pages setup, stored-result connection, configuration controls, and evidence samples. Examples: [`686902a`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/686902a), [`3b30672`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/3b30672), [`ea40ca4`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/ea40ca4), [`5e0f3c1`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/5e0f3c1). | Build Week work that turned research artifacts into a reviewable product experience. |
| 21 July 2026 | Product/copy cleanup, scope visibility, archive separation, evidence refinement, Devpost kit, and final brand work. Examples: [`c61efac`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/c61efac), [`6f84302`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/6f84302), [`a5c95bc`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/a5c95bc), [`3a4ffba`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/3a4ffba), [`97ddc4e`](https://github.com/shreyash4real/motor-fault-diagnosis-research-share/commit/97ddc4e). | Build Week refinement, evidence integration, and submission preparation. |

The dated commit history is useful evidence, but it does not replace the required `/feedback` session ID or public demo video.

## How to describe this in the submission

> I am an electrical engineer who brought the motor-fault and sensing constraints. I used Codex with GPT-5.6 to productize and audit an existing research pipeline: trace its artifacts, connect verifiable results to a static frontend, sharpen the limitation language, test the result-export path, and turn the work into a reviewable maintenance-screening story. The project does not run an OpenAI model in production; Codex accelerated the development and review process.

That wording is accurate only if paired with the Build Week evidence above and the session ID from the conversation that contains the core work.

## Submission evidence checklist

- Keep the repository public with its MIT license and working Pages URL.
- Put the public under-three-minute YouTube URL in Devpost and say both “Codex” and “GPT-5.6” in the spoken narration.
- Enter the `/feedback` session ID from the core Codex work in the Devpost form.
- Explain the pre-existing research versus Build Week productization distinction plainly.
- Do not claim that the static browser demo performs live inference or that GPT-5.6 runs inside the product.

