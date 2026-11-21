# SumMe results (development set)

Run on 2026-11-21. Methods A to D and C's ablations use the fixed weights in `configs/default.yaml`.
The blend uses weights fitted on SumMe itself (`configs/fitted_summe.yaml`), so SumMe is the
development set and not a test of the blend; its held-out figure below comes from leave-one-video-out
fitting. Raw output is in `experiments/results/summe/` (git-ignored); reproduce it with the commands in
[README.md](README.md#summe-benchmark).

**Setup.** All 25 SumMe videos (67 minutes, 15 to 18 annotators each), summary budget 15% of each
video's length, no text request. Models: OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), Whisper `base`,
`all-MiniLM-L6-v2`, on CPU.

## Result

Mean over the 25 videos (± sample standard deviation). F1 is overlap with each annotator's summary;
Kendall's τ is rank correlation with each annotator's per-second selections.

| Method | F1 | Kendall's τ | Spearman's ρ | Redundancy | Analysis seconds per source minute |
| --- | --- | --- | --- | --- | --- |
| Uniform reference (evenly spaced 5 s clips) | 0.172 ± 0.079 | n/a | n/a | n/a | 0 |
| A: activity (motion + audio) | 0.195 ± 0.110 | 0.066 ± 0.093 | 0.082 | 0.841 | 1.7 |
| B: semantic (CLIP visual only) | 0.165 ± 0.108 | 0.067 ± 0.103 | 0.084 | 0.763 | 2.3 |
| C: multimodal | 0.207 ± 0.112 | 0.102 ± 0.102 | 0.127 | 0.769 | 4.1 |
| D: classical CV (motion, colour, edges, change) | 0.186 ± 0.124 | 0.046 ± 0.126 | 0.058 | 0.856 | 1.8 |
| Blend, weights fitted on all 25 videos (in-sample) | 0.212 ± 0.107 | 0.103 ± 0.093 | 0.128 | 0.787 | 4.3 |
| Blend, leave-one-video-out (held out) | 0.193 | 0.080 | | | |
| Agreement between human annotators (mean pairwise) | 0.311 | 0.229 | | | |

Fitted blend weights: audio 0.29, novelty 0.20, visual 0.19, motion 0.14, edges 0.10, change 0.07,
speech 0 and colour 0.

Paired differences across videos (10,000-sample bootstrap interval; Wilcoxon signed-rank test, not
corrected for multiple comparisons), from `experiments/compare_methods.py`. The blend rows use the
in-sample blend.

| Comparison | Measure | Mean difference | 95% interval | Videos where first is better | p |
| --- | --- | --- | --- | --- | --- |
| C minus B | F1 | +0.042 | +0.011 to +0.079 | 17 of 25 | 0.037 |
| C minus A | F1 | +0.012 | −0.016 to +0.053 | 12 of 25 | 0.98 |
| C minus D | F1 | +0.021 | −0.010 to +0.057 | 12 of 25 | 0.48 |
| C minus uniform | F1 | +0.034 | −0.012 to +0.086 | 16 of 25 | 0.22 |
| D minus uniform | F1 | +0.013 | −0.040 to +0.072 | 11 of 25 | 0.90 |
| Blend minus C | F1 | +0.005 | −0.046 to +0.052 | 13 of 25 | 0.47 |
| Blend minus D | F1 | +0.026 | −0.030 to +0.081 | 14 of 25 | 0.17 |
| C minus D | τ | +0.056 | +0.015 to +0.096 | 18 of 25 | 0.013 |
| C minus B | τ | +0.035 | +0.012 to +0.059 | 19 of 25 | 0.008 |
| C minus A | τ | +0.037 | −0.003 to +0.075 | 17 of 25 | 0.13 |
| Blend minus C | τ | +0.001 | −0.024 to +0.023 | 15 of 25 | 0.60 |

Ablations of C (F1 / τ): without audio 0.186 / 0.092, without speech 0.198 / 0.103, without motion
0.201 / 0.096, without visual 0.203 / 0.087, without novelty 0.203 / 0.096, without diversity
0.204 / 0.102.

## Reading

- The multimodal method C has the highest F1 among the fixed-weight methods. The differences the data
  can distinguish are that C beats the visual-only baseline B on both measures (H2) and beats the
  classical method D on rank correlation.
- C is **not** distinguishable from the activity baseline A (H1 not supported here) or from evenly
  spaced clips on F1.
- The classical method D is above evenly spaced clips on average, but not distinguishably, and it is
  the weakest method on rank correlation (first half of H4 not supported here).
- Fitting the blend did not help. In-sample it is level with C; on held-out videos it is below C
  (0.193 against 0.207 F1, 0.080 against 0.102 τ). With 25 videos the fitted weights do not transfer
  even within SumMe (second half of H4 not supported here).
- The fit gave the classical colour cue and speech no weight, and gave edges and frame change a
  little.
- The diversity term lowers redundancy (0.769 with it, 0.806 without) and does not change F1 (H3:
  redundancy down, coverage unchanged).
- Removing audio hurts F1 the most; the other single-modality removals are within noise.
- All methods are well below human agreement.

## Tuning the multimodal weights

Two ways of choosing C's five weights on SumMe were tried (`python -m scenesift fit --method
multimodal`). "Held out" scores each video with weights chosen on the other 24.

| Weights for C | F1, all 25 videos | F1, held out | τ, held out |
| --- | --- | --- | --- |
| Untuned defaults | 0.207 | not fitted | not fitted |
| Least-squares fit (visual 0.21, motion 0.21, audio 0.34, speech 0, novelty 0.24) | 0.197 | 0.192 | 0.090 |
| Grid search for best F1 (visual 0.3, motion 0.2, audio 0.3, speech 0.1, novelty 0.1) | 0.228 | 0.180 | 0.068 |

The grid search finds weights that score 0.228 on the videos it searched over, but on held-out videos
it does worse than the untuned defaults: with 25 videos the search fits noise. Neither tuned set is
adopted; `configs/default.yaml` keeps the untuned weights and `configs/tuned_multimodal.yaml` records
the grid-search result. Their effect on TVSum is in [TVSUM_RESULTS.md](TVSUM_RESULTS.md).

## Against published results

Gygli et al. (ECCV 2014) report results on SumMe with the same protocol (F-measure averaged over
annotators, 15% summary length):

| Method | F-measure |
| --- | --- |
| Uniform sampling (Gygli et al.) | 0.143 |
| Colour-histogram clustering (Gygli et al.) | 0.163 |
| Visual attention (Gygli et al.) | 0.167 |
| Hand-built interestingness features (Gygli et al.) | 0.234 |
| D: classical CV, this run | 0.186 |
| C: multimodal, this run | 0.207 |
| Agreement between annotators (Gygli et al.; reproduced here) | 0.311 |

Both the classical and the multimodal method are above the simple published baselines and below the
SumMe paper's own hand-built method. The uniform reference here (0.172) scores higher than theirs (0.143); the two are
built differently, so differences of a few hundredths between the two sets of numbers should not be
over-read. Sources are listed in [docs/DATASETS.md](../docs/DATASETS.md#published-results).

## Limits of this run

- SumMe is short, unedited user video without categories or requests. It says nothing about
  requests.
- The fixed weights are untuned guesses, and the blend's weights were fitted on these same videos.
  SumMe is therefore the development set; TVSum is the test set.
- 25 videos give wide intervals. Treat every number above as descriptive.

## Checks on other footage (no annotations, judged by eye)

Five openly licensed clips from Wikimedia Commons were run at a 30-second budget with one of the
interface's preset requests. They test that the pipeline runs on real footage; they are not measurements.

| Clip (Commons file title) | Licence | Content | Observation |
| --- | --- | --- | --- |
| بازی بسکتبال مهرام تهران و شهرداری گرگان از دریچه دوربین ایسنا.webm | CC BY 3.0 | Basketball, no audio | C picked one play plus hoop, bench and crowd cutaways; A picked five clips of live play |
| Bob Cousy- The OG Celtics Floor General - Ultimate Highlight Reel.webm | Public domain | Basketball, no audio | C picked the one colour clip and a watermarked clip in a black-and-white reel; A picked play |
| FEZ trial gameplay HD.webm | CC BY-SA 3.0 | Gaming, no speech | C and A both picked the closing logo card alongside gameplay; B picked two near-identical frames |
| Mathieu Asselin Lensculture interview 2016.ogv | CC BY-SA 4.0 | Interview | 38 utterances transcribed; C picked the opening statement and four later points, lowest redundancy of the three |
| Travel vlog Bled & Bohinj - Slovenia Travel Optician.webm | CC BY 3.0 | Travel, no audio | C picked four different scenes; B picked three near-identical boat shots |

Problems these tests exposed and that were fixed: Theora video could not be decoded; reels fell short
of the budget when context padding overlapped; a black end frame was selected. Still open: unusual but
unimportant content (logo cards, watermarks, odd-one-out footage) is rated highly.
