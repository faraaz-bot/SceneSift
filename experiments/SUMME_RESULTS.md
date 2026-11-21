# First benchmark run: SumMe

Run on 2026-11-21 with the default configuration (`configs/default.yaml`), with no setting tuned on
this data. Raw output is in `experiments/results/summe/` (git-ignored); reproduce it with the commands
in [README.md](README.md#summe-benchmark).

**Setup.** All 25 SumMe videos (67 minutes, 15 to 18 annotators each), summary budget 15% of each
video's length, no text request. Models: OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), Whisper `base`,
`all-MiniLM-L6-v2`, on CPU. Reels were within half a second of each other in mean length across methods.

## Result

Mean F1 against each annotator, averaged over the 25 videos (± sample standard deviation):

| Method | F1 | Redundancy | Analysis seconds per source minute |
| --- | --- | --- | --- |
| Uniform reference (evenly spaced 5 s clips) | 0.172 ± 0.079 | n/a | 0 |
| A: activity (motion + audio) | 0.195 ± 0.110 | 0.841 | 1.8 |
| B: semantic (visual only) | 0.165 ± 0.108 | 0.763 | 2.7 |
| C: multimodal (proposed) | **0.206 ± 0.116** | 0.776 | 4.8 |
| Agreement between human annotators | 0.311 | | |

Paired differences in F1 across videos (10,000-sample bootstrap interval; Wilcoxon signed-rank test,
not corrected for multiple comparisons):

| Comparison | Mean difference | 95% interval | Videos where first is better | p |
| --- | --- | --- | --- | --- |
| C minus B | +0.041 | +0.010 to +0.078 | 17 of 25 | 0.034 |
| C minus A | +0.011 | −0.018 to +0.052 | 12 of 25 | 0.98 |
| C minus uniform | +0.033 | −0.014 to +0.086 | 15 of 25 | 0.33 |
| A minus uniform | +0.023 | −0.015 to +0.062 | 14 of 25 | 0.40 |
| B minus uniform | −0.008 | −0.049 to +0.033 | 14 of 25 | 0.85 |
| C minus C without diversity | +0.002 | −0.012 to +0.016 | 7 of 25 | 0.87 |

Ablations of C (F1): without audio 0.188, without speech 0.197, without visual 0.201, without motion
0.202, without novelty 0.205, without diversity 0.204 (redundancy rises from 0.776 to 0.810).

## Reading

- The proposed method has the highest mean, but the only difference the data can distinguish is that it
  beats the visual-only baseline (H2 direction; marginal, and one of several tests).
- It is **not** distinguishable from the activity baseline (H1 not supported here) or from evenly
  spaced clips. One video (`Jumps`, 0.50 vs 0.09) accounts for most of its lead over the activity baseline.
- The diversity term lowers redundancy slightly and does not change F1 (H3: redundancy down, coverage
  unchanged).
- Removing audio hurts the most; the other single-modality removals are within noise.
- All methods are well below human agreement (0.31).

## Against published results

Gygli et al. (ECCV 2014) report results on SumMe with the same protocol (F-measure averaged over
annotators, 15% summary length):

| Method | F-measure |
| --- | --- |
| Uniform sampling (Gygli et al.) | 0.143 |
| Colour-histogram clustering (Gygli et al.) | 0.163 |
| Visual attention (Gygli et al.) | 0.167 |
| Hand-built interestingness features (Gygli et al.) | 0.234 |
| C: multimodal, this run | 0.206 |
| Agreement between annotators (Gygli et al.; reproduced here) | 0.311 |

The multimodal method is above the simple published baselines and below the SumMe paper's own
hand-built method. The uniform reference here (0.172) scores higher than theirs (0.143); the two are
built differently, so differences of a few hundredths between the two sets of numbers should not be
over-read. Sources are listed in [docs/DATASETS.md](../docs/DATASETS.md#published-results).

## Limits of this run

- SumMe is short, unedited user video without categories or requests. It says nothing about
  requests.
- The fusion weights are untuned guesses. Because these results have now been seen, SumMe can no
  longer serve as an untouched test set. It is the set the weights are fitted on, and
  TVSum is the test set.
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
