# Experiment Plan: General-Purpose Video Highlight Extraction

> This document defines **the research experiment**, not how to implement it. No software libraries, model configurations, or code design are specified here.

## 1. Motivation

Long videos often contain a small number of moments viewers consider memorable or important. What constitutes a highlight differs across contexts: sports prioritize events and reactions, how-to videos prioritize the key steps, and spoken content prioritizes informative speech. A solution designed for one kind of video may fail on another. Much recent work answers this with trained deep networks; this study asks how far measurements taken directly from the pixels go, and whether they add to learned features.

## 2. Research question

**Does blending classical computer-vision cues with learned multimodal features select highlights that agree with human annotators better than either kind of evidence alone, across the video categories of public benchmarks?**

The study examines *agreement with annotators on the tested benchmarks*, not universal understanding of every possible video.

## 3. Objectives

1. Retain important source-video moments within a short duration budget.
2. Prefer segments matching a specific user request when one is supplied (demonstrated, not measured).
3. Avoid repetitive or near-identical highlights.
4. Preserve surrounding context so chosen moments remain understandable.
5. Measure quality and computational cost across different video types.

## 4. Hypotheses

- **H1:** A multimodal selection method improves temporal highlight F1 versus an activity-focused baseline at the same duration budget.
- **H2:** A multimodal selection method improves temporal highlight F1 versus a visual-semantic baseline.
- **H3:** Redundancy-aware selection decreases repeated content while preserving important-event coverage.
- **H4:** A method using only classical computer-vision cues scores above evenly spaced clips, and a blend of classical and learned evidence scores above either alone.

These are prospective hypotheses; negative or mixed outcomes are valid findings.

## 5. Data

| Dataset | Videos | Human labels | Role |
| --- | --- | --- | --- |
| SumMe | 25 short user videos | 15–18 people each selected a summary of the video | Development set: already run once, and used to fit fusion weights |
| TVSum | 50 web videos, 5 in each of 10 categories | 20 people each scored every 2-second shot from 1 (not important) to 5 (very important) | Held-out test set and the source of per-category results |

Both are public research datasets with existing annotations, so no videos are collected and no annotators are recruited. Record the source and licence of each dataset and do not redistribute the videos.

## 6. Reference annotations and data separation

The datasets' own annotations are used as published. Scores are computed against each annotator separately and then averaged, and agreement between annotators is reported as a ceiling.

SumMe results have already been looked at, so SumMe cannot serve as an untouched test set. Any fitting or method selection uses SumMe only. TVSum annotations are not consulted until the methods and weights are frozen.

## 7. Compared methods

- **A: Activity baseline.** Prioritizes segments with strong observable activity/excitement.
- **B: Visual-semantic baseline.** Prioritizes distinctive or semantically relevant visual content.
- **C: Proposed multimodal method.** Combines complementary evidence about event significance with relevance, novelty, and context constraints.
- **D: Classical computer-vision method.** Uses only hand-built cues measured directly from pixels, with no pretrained models.
- **Blend.** Combines the classical and learned evidence with weights fitted on SumMe.
- **Uniform reference.** Evenly spaced clips that ignore the content, showing what a score looks like with no skill.

All systems use identical test videos, target summary budgets, and comparable evaluation protocols.

## 8. Evaluation methodology

| Measure | Interpretation |
| --- | --- |
| Temporal precision / recall / F1 | Agreement between selected time intervals and human-labeled highlights |
| Important-event coverage | Fraction of independently marked important events represented |
| Redundancy | Frequency of repeated/near-duplicate content |
| Rank correlation (Kendall's τ, Spearman's ρ) | Whether a method orders moments by importance the way annotators do; computed on TVSum, which has importance scores |
| Processing time | Runtime per minute of input video |

Temporal F1 alone is a weak measure on these benchmarks: Otani et al. (CVPR 2019) showed that random selections score close to published methods, which is why the uniform reference and rank correlation are both reported. Worked examples and failure cases accompany the scores.

Report scores by **TVSum category**, along with pooled summaries. Use a summary budget of 15% of each video's length, the convention for both benchmarks. Provide confidence intervals or clearly state limited sample sizes. Review failures, not only average scores.

## 9. Experiments

1. **Baseline comparison:** Evaluate methods A, B, C, and the uniform reference on the same inputs.
2. **Classical versus learned:** Evaluate method D against B and C, then the blend against each.
3. **Weight fitting:** Fit the blend's weights on SumMe, freeze them, and evaluate on TVSum.
4. **Category differences:** Report TVSum results for each of its ten categories.
5. **Redundancy:** Compare selection with versus without a diversity constraint.
6. **Modality contribution:** Remove one information source at a time and evaluate the change.
7. **Efficiency:** Measure summary quality against processing time.
8. **Prompt demonstration (not measured):** Show a few request-guided reels beside generic ones.

## 10. Validity and limitations

- “Importance” is subjective; inter-annotator disagreement is expected.
- Two benchmarks of short web and user videos cannot establish universal generalization.
- Weights fitted on 25 SumMe videos may not transfer to TVSum, whose videos and annotation style differ.
- Published deep-learning results on these benchmarks often use a different scoring protocol, so they are not directly comparable with the numbers reported here.
- Video length, creator, language, and production style are confounding factors.
- Some ground truth labels may favor one type of summary over another.
- Small samples limit statistical power; report this transparently.
- Avoid category-specific ground-truth cues that unfairly favor the proposed method.
- Use the same source conditions across models and respect dataset licensing.

## 11. Success criteria

A successful course project provides a reproducible comparison, measurable evaluation, and defensible analysis—including if the proposed multimodal method does **not** outperform simpler baselines. No minimum score is asserted before experimentation.

## 12. Published reference results

| Benchmark and measure | Published result | Source |
| --- | --- | --- |
| SumMe, F-measure averaged over annotators | 0.143 uniform sampling; 0.163 colour-histogram clustering; 0.234 hand-built interestingness features; 0.311 between annotators | Gygli et al., ECCV 2014 |
| SumMe, Kendall's τ | 0.096 zero-shot from pretrained features; 0.246 supervised (CSTA); 0.205 human | Pang et al. 2024; Son et al. 2024 |
| TVSum, Kendall's τ | 0.000 random; 0.042 dppLSTM; 0.135 zero-shot from pretrained features; 0.194 supervised (CSTA); 0.177 human | Otani et al., CVPR 2019; Pang et al. 2024; Son et al. 2024 |

The methods in this study train no network, so the targets are 0.234 on SumMe and the zero-shot τ of 0.135 on TVSum; supervised results are listed for context. Measured results are in [SUMME_RESULTS.md](../experiments/SUMME_RESULTS.md) and [TVSUM_RESULTS.md](../experiments/TVSUM_RESULTS.md). Full tables and citations are in [DATASETS.md](DATASETS.md#published-results).

Related documents: [Project overview](../README.md), [Implementation design](IMPLEMENTATION.md), [Architecture](ARCHITECTURE.md).
