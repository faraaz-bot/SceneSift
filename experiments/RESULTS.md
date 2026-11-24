# Results

Run on 2026-11-23. Raw output is in `experiments/results/` (git-ignored); the commands that reproduce it
are in [README.md](README.md). How the methods work is in [docs/METHODS.md](../docs/METHODS.md).

## Question and hypotheses

When no model is trained for the task, do classical computer-vision cues, features from pretrained
models, or a blend of the two select highlights that agree best with human annotators?

- **H1:** the multimodal method C scores above the activity baseline A.
- **H2:** C scores above the visual-semantic baseline B.
- **H3:** the diversity term reduces repeated content without reducing coverage of important events.
- **H4:** the classical method D scores above evenly spaced clips, and the blend scores above both D and C.

## Protocol

- **SumMe** (25 videos, 67 minutes, 15–18 annotators each) is the development set. Every design choice
  was made on it: the classical method's five cues, and the blend's weights. The blend's SumMe scores
  are therefore given by leave-one-video-out, each video scored with weights fitted on the other 24.
- **TVSum** (50 videos, 210 minutes, 20 annotators each, 10 categories of 5) is the test set. It is run
  with everything fixed beforehand.
- **Disclosure.** TVSum was first run with an earlier version of the classical method that measured
  motion with optical flow. That method was then rebuilt from course techniques only, using SumMe
  alone, and TVSum was run once more with the new design frozen. Methods A, B, and C were not changed
  between the two runs. The earlier classical method scored τ 0.088 and F1 0.180 on TVSum, so the
  revision did not change its TVSum result beyond noise.
- Every summary is 15% of the video's length, with no text request. Models: OpenCLIP ViT-B-32
  (`laion2b_s34b_b79k`), Whisper `base`, `all-MiniLM-L6-v2`, on a 16-core CPU.
- F1 is overlap with each annotator's summary; Kendall's τ and Spearman's ρ are rank correlations with
  each annotator's scores. All are averaged over annotators, then over videos. Paired differences use a
  10,000-sample bootstrap interval over videos and a Wilcoxon signed-rank test, uncorrected for
  multiple comparisons (`compare_methods.py`).

## Main results

| Method | SumMe F1 | SumMe τ | SumMe ρ | TVSum F1 | TVSum τ | TVSum ρ |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform reference | 0.172 | n/a | n/a | 0.143 | n/a | n/a |
| A: activity (optical flow + audio) | 0.195 | 0.066 | 0.082 | 0.176 | **0.097** | **0.127** |
| B: semantic (CLIP visual) | 0.165 | 0.067 | 0.084 | 0.144 | −0.057 | −0.076 |
| C: multimodal | **0.207** | **0.102** | **0.127** | 0.149 | −0.008 | −0.011 |
| D: classical CV (course techniques only) | 0.189 | 0.078 | 0.097 | **0.185** | 0.084 | 0.109 |
| Blend, weights fitted on SumMe | 0.202 | 0.102 | 0.126 | 0.156 | 0.060 | 0.077 |
| Agreement between annotators | 0.311 | 0.229 | | 0.250 | 0.178 | |

The blend's SumMe figures are leave-one-video-out; fitted and scored on all 25 videos it reaches F1
0.222 and τ 0.128. Its fitted weights are edges 0.23, audio 0.20, novelty 0.19, moving area 0.15,
camera motion 0.14, CLIP visual 0.09, speech 0.01, and zero for optical-flow motion, scene motion, and
colour.

### Paired differences

| Set | Comparison | Measure | Mean difference | 95% interval | Videos where first is better | p |
| --- | --- | --- | --- | --- | --- | --- |
| TVSum | D minus B | τ | +0.141 | +0.092 to +0.194 | 39 of 50 | < 0.001 |
| TVSum | D minus C | τ | +0.093 | +0.053 to +0.134 | 40 of 50 | < 0.001 |
| TVSum | D minus A | τ | −0.013 | −0.038 to +0.011 | 21 of 50 | 0.47 |
| TVSum | A minus C | τ | +0.105 | +0.067 to +0.144 | 39 of 50 | < 0.001 |
| TVSum | C minus B | τ | +0.049 | +0.027 to +0.072 | 32 of 50 | < 0.001 |
| TVSum | Blend minus C | τ | +0.068 | +0.035 to +0.103 | 39 of 50 | < 0.001 |
| TVSum | Blend minus D | τ | −0.025 | −0.052 to +0.005 | 20 of 50 | 0.053 |
| TVSum | Blend minus A | τ | −0.037 | −0.065 to −0.009 | 19 of 50 | 0.018 |
| TVSum | D minus uniform | F1 | +0.042 | +0.016 to +0.070 | 27 of 50 | 0.005 |
| TVSum | D minus C | F1 | +0.036 | +0.015 to +0.058 | 35 of 50 | 0.003 |
| TVSum | D minus A | F1 | +0.009 | −0.008 to +0.026 | 30 of 50 | 0.33 |
| TVSum | Blend minus D | F1 | −0.030 | −0.049 to −0.011 | 18 of 50 | 0.011 |
| TVSum | C minus uniform | F1 | +0.006 | −0.012 to +0.025 | 22 of 50 | 0.72 |
| SumMe | C minus B | τ | +0.035 | +0.012 to +0.059 | 19 of 25 | 0.009 |
| SumMe | C minus D | τ | +0.024 | −0.023 to +0.071 | 13 of 25 | 0.35 |
| SumMe | C minus A | τ | +0.036 | −0.003 to +0.075 | 17 of 25 | 0.12 |
| SumMe | D minus A | τ | +0.012 | −0.018 to +0.044 | 13 of 25 | 0.53 |
| SumMe | C minus uniform | F1 | +0.034 | −0.013 to +0.087 | 15 of 25 | 0.30 |
| SumMe | D minus uniform | F1 | +0.017 | −0.033 to +0.068 | 14 of 25 | 0.62 |

### Other measures

| Set | Method | Precision | Recall | Event coverage | Redundancy | Analysis s per video minute |
| --- | --- | --- | --- | --- | --- | --- |
| SumMe | A | 0.185 | 0.211 | 0.204 | 0.841 | 1.9 |
| SumMe | B | 0.160 | 0.174 | 0.181 | 0.763 | 2.4 |
| SumMe | C | 0.198 | 0.221 | 0.214 | 0.773 | 4.5 |
| SumMe | D | 0.183 | 0.202 | 0.207 | 0.840 | 1.7 |
| SumMe | Blend (in-sample) | 0.213 | 0.238 | 0.238 | 0.787 | 5.3 |
| TVSum | A | 0.173 | 0.180 | 0.201 | 0.779 | 1.4 |
| TVSum | B | 0.142 | 0.146 | 0.160 | 0.540 | 2.0 |
| TVSum | C | 0.147 | 0.151 | 0.166 | 0.556 | 4.5 |
| TVSum | D | 0.182 | 0.189 | 0.202 | 0.775 | 1.3 |
| TVSum | Blend | 0.154 | 0.158 | 0.165 | 0.617 | 5.2 |

Redundancy is the mean pairwise cosine similarity between the CLIP embeddings of the selected clips.

## Ablations of the multimodal method

| Variant | SumMe F1 | SumMe τ | SumMe redundancy | TVSum F1 | TVSum τ | TVSum redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| C, full | 0.207 | 0.102 | 0.773 | 0.149 | −0.008 | 0.556 |
| without visual | 0.205 | 0.085 | 0.810 | 0.153 | 0.064 | 0.625 |
| without motion | 0.204 | 0.097 | 0.741 | 0.145 | −0.047 | 0.536 |
| without audio | 0.185 | 0.091 | 0.770 | 0.151 | −0.021 | 0.549 |
| without speech | 0.198 | 0.104 | 0.763 | 0.149 | −0.016 | 0.566 |
| without novelty | 0.205 | 0.097 | 0.777 | 0.152 | 0.010 | 0.580 |
| without diversity | 0.204 | 0.102 | 0.810 | 0.150 | −0.008 | 0.656 |

## TVSum by category

Mean Kendall's τ over the five videos in each category; descriptive only.

| Category | A: activity | B: semantic | C: multimodal | D: classical | Blend |
| --- | --- | --- | --- | --- | --- |
| Attempting bike tricks | 0.166 | −0.085 | 0.030 | 0.203 | 0.102 |
| Bee keeping | 0.164 | −0.051 | 0.010 | 0.142 | 0.110 |
| Changing vehicle tire | 0.110 | −0.101 | −0.046 | 0.110 | 0.027 |
| Dog show | 0.033 | −0.014 | −0.009 | 0.012 | 0.001 |
| Flash mob gathering | 0.049 | −0.048 | −0.025 | −0.036 | 0.064 |
| Getting vehicle unstuck | 0.080 | 0.013 | 0.051 | 0.083 | 0.103 |
| Grooming an animal | 0.124 | −0.146 | −0.033 | 0.057 | −0.014 |
| Making sandwich | 0.079 | 0.119 | 0.121 | 0.108 | 0.062 |
| Parade | 0.142 | −0.171 | −0.086 | 0.158 | 0.123 |
| Parkour | 0.021 | −0.087 | −0.099 | 0.007 | 0.019 |

## Developing the classical method

All of this used SumMe only.

| Classical method variant | SumMe F1 | SumMe τ |
| --- | --- | --- |
| Optical-flow motion, colour, Sobel edges, frame difference (first version; uses a technique outside the course) | 0.186 | 0.046 |
| Scene motion (SIFT + RANSAC homography), colour, Sobel edges, frame difference | 0.179 | 0.058 |
| Scene motion, moving area, camera motion, Canny edges, colour (**adopted**) | 0.189 | 0.078 |
| Adopted cues with least-squares weights, leave-one-video-out | 0.184 | 0.060 |

Scored alone on SumMe, the strongest single cues were scene motion and moving area (τ about 0.06),
camera motion (0.05), and Canny edge density (0.04); Sobel gradient magnitude (0.02), colour (0.02),
and Harris corner counts (0.01) were weaker. Canny edge density replaced Sobel magnitude, and camera
motion and moving area were added. Least-squares weights did worse on held-out videos than equal
weights, so equal weights are used.

## Tuning the multimodal weights

Diagnostics: the TVSum columns were computed after TVSum had been run.

| Weights for C | Chosen on | SumMe F1, all videos | SumMe F1, held out | TVSum τ | TVSum F1 |
| --- | --- | --- | --- | --- | --- |
| Untuned (visual 0.30, motion 0.20, audio 0.15, speech 0.20, novelty 0.15) | nothing | 0.207 | not fitted | −0.008 | 0.149 |
| Least squares (visual 0.20, motion 0.21, audio 0.34, speech 0, novelty 0.25) | SumMe | 0.197 | 0.191 | 0.013 | 0.146 |
| Grid search for best F1 (visual 0.3, motion 0.2, audio 0.3, speech 0.1, novelty 0.1) | SumMe | 0.226 | 0.190 | 0.014 | 0.143 |
| Least squares, each video scored with weights from the other 49 | TVSum | | | 0.086 | 0.172 |

The same cross-validated fit within TVSum gives τ 0.077 for D and 0.074 for the blend. The weights it
finds for C put 0.69 on optical-flow motion and none on novelty.

## Against published results

| Method | Training | SumMe τ | TVSum τ |
| --- | --- | --- | --- |
| Random | none | 0.000 | 0.000 |
| DR-DSN (Otani et al. 2019) | trained, no labels | | 0.020 |
| dppLSTM (Otani et al. 2019) | supervised | | 0.042 |
| D: classical CV, this work | none | 0.078 | 0.084 |
| A: activity, this work | none | 0.066 | 0.097 |
| C: multimodal, this work | none | 0.102 | −0.008 |
| Zero-shot contrastive scoring (Pang et al. 2024) | none | 0.096 | 0.135 |
| VASNet (as listed by Son et al. 2024) | supervised | 0.160 | 0.160 |
| CSTA (Son et al. 2024) | supervised | 0.246 | 0.194 |
| Human (Otani et al. 2019; Son et al. 2024) | | 0.205 | 0.177 |

The pairwise human τ measured here on TVSum (0.178) matches the published 0.177. On SumMe F1 under the
same protocol, Gygli et al. (2014) report 0.143 for uniform sampling, 0.163 for colour-histogram
clustering, and 0.234 for hand-built interestingness features; no method here reaches 0.234. Sources
are in [docs/DATASETS.md](../docs/DATASETS.md#published-results).

## Reading

- **The classical method holds up on the test set.** Built only from course techniques and with no
  pretrained model, D ranks moments as annotators do about as well as the optical-flow activity
  baseline, and clearly better than the methods built on CLIP (H4, first half: supported on TVSum,
  where D is distinguishably above evenly spaced clips; not distinguishable on SumMe).
- **The multimodal method does not generalize.** C leads on SumMe within noise and is at chance on
  TVSum, below both A and D (H1 not supported; the reverse holds on TVSum). C is above B on both sets
  (H2), but on TVSum only because B is below chance.
- **The CLIP visual feature is the cause.** Without a text request it scores how unlike the rest of the
  video a clip looks, which on TVSum's edited videos is negatively related to importance. Removing it
  raises C from −0.008 to 0.064; removing motion lowers C the most.
- **The blend helps on SumMe and not on TVSum.** On held-out SumMe videos it is level with C. On TVSum
  it is clearly better than C, because the fit gave weight to three classical cues, but it does not
  beat D alone (H4, second half: not supported).
- **Tuning C's weights does not rescue it.** Weights chosen on SumMe do worse than the untuned ones on
  held-out SumMe videos and barely move TVSum. Weights fitted within TVSum reach D's level only by
  putting most of the weight on motion.
- **Diversity** lowers redundancy on both sets and changes neither F1 nor τ (H3 supported).
- **Cost.** D needs 1.3 seconds of analysis per minute of video on TVSum; C needs 4.5.
- **All methods are far from people** and below the best published training-free and supervised
  methods.

## Limits

- 25 and 50 videos, five per TVSum category. Category-level differences are not established.
- TVSum F1 depends on how tied ratings become reference summaries and separates methods poorly; rank
  correlation is the measure to rely on.
- The classical method was revised after a first TVSum run (see Protocol), so its TVSum result is not
  from a strictly untouched test set.
- The fixed weights of A, B, and C were set by judgement.
- Neither benchmark has text requests, so selection guided by a request is not evaluated.
