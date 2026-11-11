# Experiment Plan: General-Purpose Video Highlight Extraction

> This document defines **the research experiment**, not how to implement it. No software libraries, model configurations, or code design are specified here.

## 1. Motivation

Long videos often contain a small number of moments viewers consider memorable or important. What constitutes a highlight differs across contexts: sports prioritize events and reactions, interviews prioritize informative speech, games prioritize meaningful progress, and travel footage often prioritizes distinctive scenes. A solution designed for one genre may fail on another.

## 2. Research question

**Can one category-independent, multimodal highlight selection approach generalize across basketball, gaming, podcasts/interviews, and travel videos better than simpler selection strategies?**

The study examines *generalization across tested domains*, not universal understanding of every possible video.

## 3. Objectives

1. Retain important source-video moments within a short duration budget.
2. Prefer segments matching a specific user request when one is supplied.
3. Avoid repetitive or near-identical highlights.
4. Preserve surrounding context so chosen moments remain understandable.
5. Measure quality and computational cost across different video types.

## 4. Hypotheses

- **H1:** A multimodal selection method improves temporal highlight F1 versus an activity-focused baseline at the same duration budget.
- **H2:** A multimodal selection method improves temporal highlight F1 versus a visual-semantic baseline, particularly on speech-heavy videos.
- **H3:** Request-conditioned summaries receive higher human relevance ratings for instruction-specific tasks than generic summaries.
- **H4:** Redundancy-aware selection decreases repeated content while preserving important-event coverage.

These are prospective hypotheses; negative or mixed outcomes are valid findings.

## 5. Categories and sample

| Video type | Reason for inclusion | Typical important content |
| --- | --- | --- |
| Basketball | Fast motion and crowd/audio reactions | Pivotal plays, celebrations |
| Gaming | Computer-generated visuals, dynamic events | Achievements, key encounters |
| Podcast/interview | Primarily linguistic significance | Important explanations, arguments |
| Travel/vlog | Scenic diversity and long-form progression | Unique destinations and activities |

**Initial target:** 3–5 videos per category, varied in duration and creators where possible. Use properly licensed or authorized footage. Record each source, duration, genre, and annotation conditions. SumMe and TVSum may be used as auxiliary benchmarks if compatible.

## 6. Human reference annotations

Have at least two people independently mark important intervals for each video where feasible. Provide consistent instructions about significance, boundaries, allowable lead-in/lead-out context, and duplicate events. For prompted tests, annotators should see exactly the same user request. Record disagreements and report agreement; choose and document a reconciliation procedure rather than silently replacing independent judgments.

Exclude test videos from any parameter tuning or qualitative model selection.

## 7. Compared methods

- **A: Activity baseline.** Prioritizes segments with strong observable activity/excitement.
- **B: Visual-semantic baseline.** Prioritizes distinctive or semantically relevant visual content.
- **C: Proposed multimodal method.** Combines complementary evidence about event significance with relevance, novelty, and context constraints.

All systems use identical test videos, target summary budgets, and comparable evaluation protocols.

## 8. Evaluation methodology

| Measure | Interpretation |
| --- | --- |
| Temporal precision / recall / F1 | Agreement between selected time intervals and human-labeled highlights |
| Important-event coverage | Fraction of independently marked important events represented |
| Redundancy | Frequency of repeated/near-duplicate content |
| Human relevance rating | Perceived usefulness and prompt alignment |
| Context/coherence rating | Whether extracted clips remain understandable |
| Processing time | Runtime per minute of input video |

Report scores by **domain**, along with pooled summaries. Compare multiple highlight duration budgets (for example 30 s, 60 s, 120 s where input length permits). Provide confidence intervals or clearly state limited sample sizes. Review failures, not only average scores.

## 9. Experiments

1. **Baseline comparison:** Evaluate methods A, B, and C on the same held-out inputs.
2. **Category generalization:** Measure differences between sports, gaming, interview, and travel videos.
3. **Prompt relevance:** Compare generic against request-guided summaries for several preselected prompts.
4. **Redundancy:** Compare selection with versus without a diversity constraint.
5. **Modality contribution:** Remove one information source at a time and evaluate the change.
6. **Efficiency:** Measure summary quality against processing time and target duration.

## 10. Validity and limitations

- “Importance” is subjective; inter-annotator disagreement is expected.
- Four test domains cannot establish universal generalization.
- Video length, creator, language, and production style are confounding factors.
- Some ground truth labels may favor one type of summary over another.
- Small samples limit statistical power; report this transparently.
- Avoid category-specific ground-truth cues that unfairly favor the proposed method.
- Use the same source conditions across models and respect dataset licensing.

## 11. Success criteria

A successful course project provides a reproducible comparison, measurable evaluation, and defensible analysis—including if the proposed multimodal method does **not** outperform simpler baselines. No minimum score is asserted before experimentation.

Related documents: [Project overview](../README.md), [Implementation design](IMPLEMENTATION.md), [Architecture](ARCHITECTURE.md).
