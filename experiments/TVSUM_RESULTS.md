# TVSum results (held-out test set)

Run once on 2026-11-21, after the methods and the blend's weights were fixed on SumMe
(`configs/fitted_summe.yaml`). No setting was changed after seeing these numbers. Raw output is in
`experiments/results/tvsum/` (git-ignored); reproduce it with the commands in
[README.md](README.md#tvsum-benchmark).

**Setup.** All 50 TVSum videos (210 minutes, 20 annotators each, 10 categories of 5 videos), no text
request. Models: OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), Whisper `base`, `all-MiniLM-L6-v2`, on CPU.

## Result

Mean over the 50 videos (± sample standard deviation). Kendall's τ and Spearman's ρ are rank
correlations between a method's score over time and each annotator's score for every 2-second shot.
F1 is overlap between a 15% summary and each annotator's 15% reference summary.

| Method | Kendall's τ | Spearman's ρ | F1 | Redundancy | Analysis seconds per source minute |
| --- | --- | --- | --- | --- | --- |
| Uniform reference (evenly spaced 5 s clips) | n/a | n/a | 0.143 ± 0.034 | n/a | 0 |
| A: activity (motion + audio) | **0.097 ± 0.113** | 0.127 | 0.176 ± 0.078 | 0.779 | 1.2 |
| B: semantic (CLIP visual only) | −0.057 ± 0.141 | −0.076 | 0.144 ± 0.086 | 0.540 | 2.0 |
| C: multimodal | −0.009 ± 0.130 | −0.011 | 0.150 ± 0.065 | 0.559 | 4.3 |
| D: classical CV (motion, colour, edges, change) | 0.088 ± 0.127 | 0.114 | **0.180 ± 0.093** | 0.762 | 1.3 |
| Blend, weights fitted on SumMe | 0.028 ± 0.124 | 0.036 | 0.147 ± 0.065 | 0.597 | 4.4 |
| Agreement between human annotators (mean pairwise) | 0.178 | | 0.250 | | |

Paired differences across videos (10,000-sample bootstrap interval; Wilcoxon signed-rank test, not
corrected for multiple comparisons), from `experiments/compare_methods.py`:

| Comparison | Measure | Mean difference | 95% interval | Videos where first is better | p |
| --- | --- | --- | --- | --- | --- |
| D minus B | τ | +0.145 | +0.097 to +0.195 | 42 of 50 | < 0.001 |
| D minus C | τ | +0.096 | +0.059 to +0.134 | 40 of 50 | < 0.001 |
| D minus A | τ | −0.009 | −0.033 to +0.015 | 23 of 50 | 0.61 |
| A minus C | τ | +0.105 | +0.067 to +0.144 | 40 of 50 | < 0.001 |
| C minus B | τ | +0.049 | +0.027 to +0.072 | 32 of 50 | < 0.001 |
| Blend minus C | τ | +0.037 | +0.014 to +0.059 | 35 of 50 | 0.002 |
| Blend minus D | τ | −0.059 | −0.093 to −0.027 | 15 of 50 | < 0.001 |
| Blend minus A | τ | −0.069 | −0.100 to −0.038 | 14 of 50 | < 0.001 |
| D minus uniform | F1 | +0.037 | +0.009 to +0.068 | 28 of 50 | 0.046 |
| D minus C | F1 | +0.030 | +0.005 to +0.059 | 31 of 50 | 0.053 |
| A minus C | F1 | +0.026 | +0.005 to +0.049 | 30 of 50 | 0.035 |
| C minus uniform | F1 | +0.007 | −0.011 to +0.027 | 23 of 50 | 0.87 |
| Blend minus uniform | F1 | +0.004 | −0.015 to +0.025 | 24 of 50 | 0.95 |

Ablations of C (τ): without visual 0.065, without novelty 0.010, without diversity −0.009, without
speech −0.016, without audio −0.021, without motion −0.047. Removing the CLIP visual feature raises
C from −0.009 to 0.065; removing motion lowers it the most.

## By category

Mean Kendall's τ over the five videos in each category:

| Category | A: activity | B: semantic | C: multimodal | D: classical | Blend |
| --- | --- | --- | --- | --- | --- |
| Attempting bike tricks | 0.166 | −0.085 | 0.030 | 0.203 | 0.048 |
| Bee keeping | 0.164 | −0.051 | 0.010 | 0.155 | 0.078 |
| Changing vehicle tire | 0.110 | −0.101 | −0.046 | 0.068 | −0.050 |
| Dog show | 0.033 | −0.014 | −0.011 | 0.034 | 0.004 |
| Flash mob gathering | 0.049 | −0.048 | −0.024 | −0.006 | 0.053 |
| Getting vehicle unstuck | 0.080 | 0.013 | 0.045 | 0.097 | 0.076 |
| Grooming an animal | 0.124 | −0.146 | −0.034 | 0.054 | −0.048 |
| Making sandwich | 0.079 | 0.119 | 0.121 | 0.121 | 0.091 |
| Parade | 0.142 | −0.171 | −0.082 | 0.128 | 0.044 |
| Parkour | 0.021 | −0.087 | −0.095 | 0.023 | −0.014 |

With five videos per category these are descriptive only.

## Against published results

Kendall's τ / Spearman's ρ on TVSum (sources in [docs/DATASETS.md](../docs/DATASETS.md#published-results)):

| Method | Training | τ | ρ |
| --- | --- | --- | --- |
| Random | None | 0.000 | 0.000 |
| DR-DSN (Otani et al. 2019) | Trained, no labels | 0.020 | 0.026 |
| dppLSTM (Otani et al. 2019) | Supervised | 0.042 | 0.055 |
| D: classical CV, this run | None | 0.088 | 0.114 |
| A: activity, this run | None | 0.097 | 0.127 |
| Zero-shot contrastive scoring (Pang et al. 2024) | None | 0.135 | 0.178 |
| VASNet (as listed by Son et al. 2024) | Supervised | 0.160 | 0.170 |
| CSTA (Son et al. 2024) | Supervised | 0.194 | 0.255 |
| Human (Otani et al. 2019) | | 0.177 | 0.204 |

The pairwise human τ measured here (0.178) matches the published human figure (0.177). The published
ρ for humans is computed differently, so it is not compared.

## Reading

- The classical method D, which uses no pretrained model, ranks moments the way annotators do about
  as well as the activity baseline A, and clearly better than the methods built on CLIP features. It
  is also the only method distinguishably above evenly spaced clips on F1 (first half of H4 supported).
- The multimodal method C is at chance on rank correlation, and below both A and D (H1 not supported;
  the reverse holds). C is above the visual-only baseline B (H2 supported), but only because B is
  below chance.
- The visual feature is the cause. Without a text request it scores how unlike the rest of the video
  a clip looks, and on TVSum that is negatively related to what annotators rate as important. C
  without it reaches 0.065.
- The blend fitted on SumMe is better than C but worse than D or A alone (second half of H4 not
  supported). SumMe gave weight to the visual and novelty features that do not carry over to TVSum.
- The diversity term lowers redundancy (0.559 with it, 0.655 without) and changes neither τ nor F1
  (H3: redundancy down, coverage unchanged).
- Against published work, D and A are above the two trained methods measured by Otani et al., below
  the best training-free method, and well below supervised methods and human agreement.
- A and D each cost about 1.2 to 1.3 seconds of analysis per minute of video, under a third of C.

## Follow-up: tuning the multimodal weights

Done after the held-out run above, so these are diagnostics and not held-out results. The question
was whether different weights for the same five features would make method C work on TVSum.

| Weights for C | Chosen on | TVSum τ | TVSum F1 |
| --- | --- | --- | --- |
| Untuned defaults (visual 0.30, motion 0.20, audio 0.15, speech 0.20, novelty 0.15) | Nothing | −0.009 | 0.150 |
| Grid search for best F1 (visual 0.3, motion 0.2, audio 0.3, speech 0.1, novelty 0.1) | SumMe | 0.014 | 0.143 |
| Least-squares fit (visual 0.21, motion 0.21, audio 0.34, speech 0, novelty 0.24) | SumMe | 0.013 | 0.146 |
| Least-squares fit, each video scored with weights from the other 49 | TVSum, cross-validated | 0.086 | 0.178 |

For comparison, D with its untuned weights is at 0.088 and 0.180.

- Weights chosen on SumMe move C on TVSum by about +0.02 τ (interval +0.009 to +0.037 for the grid
  search) and leave it far below D (−0.074, interval −0.110 to −0.039).
- Weights fitted within TVSum bring C level with D. They do it by putting two thirds of the weight on
  motion (motion 0.67, visual 0.12, audio 0.11, speech 0.10, novelty 0), which turns C into a
  motion-led method. The same cross-validated fit gives 0.082 for the blend and 0.090 for D.
- So the weights were part of the problem, but the fix is to lean on the classical motion cue, and no
  weighting of these features gets past about 0.09 on TVSum. The gap to the published training-free
  result (0.135) would need better features, not better weights.
- The weights that suit SumMe and the weights that suit TVSum differ. SumMe favours audio, novelty,
  and visual distinctiveness; TVSum favours motion and frame change.

## Limits of this run

- The result for C and B is specific to request-free scoring. With a text request the visual feature
  measures similarity to the request instead; TVSum has no requests, so that mode is not tested.
- F1 on TVSum depends on how reference summaries are built from tied scores and is known to separate
  methods poorly; rank correlation is the measure to rely on.
- 50 videos, 5 per category. Pooled differences between D or A and the CLIP-based methods are large
  relative to their intervals; category-level differences are not established.
- TVSum has now been seen. Any change made in response to these numbers needs a different test set.
