# SceneSift — How the methods work

Every video goes through the same four stages. The methods differ only in the evidence they score with.

```mermaid
flowchart LR
    V[Video] --> C[1. Candidate clips]
    C --> E[2. Evidence per clip]
    E --> N[3. Normalize and fuse]
    N --> S[4. Select under the length budget]
    S --> R[Highlight reel, timestamps, evidence]
```

| Method | Evidence | Pretrained models | Weights | Diversity term |
| --- | --- | --- | --- | --- |
| A `activity` | optical-flow motion, audio | none | fixed | no |
| B `semantic` | CLIP visual | CLIP | fixed | no |
| C `multimodal` | CLIP visual, optical-flow motion, audio, speech, novelty | CLIP, Whisper, sentence embeddings | fixed | yes |
| D `classical` | scene motion, moving area, camera motion, edges, colour | none | fixed, equal | no |
| `blend` | everything above | all | fitted on SumMe | yes |

The two approaches the study is about are **D, the classical computer-vision method**, and **C, the multimodal method**. A and B are baselines, and the blend combines all evidence.

## 1. Shared stages

**Candidate clips** (`scenesift/segmentation.py`). Shot cuts are detected from frame-to-frame colour change. Shots of 2–12 s become candidates, and overlapping windows of 4 s and 8 s (50% overlap) cover the whole video, with edges snapped to a cut within 0.75 s. Near-duplicates are dropped, and so are candidates in which more than half the frames are blank (grey-level standard deviation below 6).

**Normalization** (`scenesift/scoring.py`). What counts as "a lot" depends on the video, so each feature is mapped to (0, 1) within the video by a logistic function of its median/MAD z-score. A value of 0.5 is that video's median.

**Fusion.** A method's score for a clip is the weighted sum of its normalized features, with the weights renormalized over the features it uses. Evidence a video lacks (no audio, no speech, no measurable camera motion) is left out and the remaining weights are renormalized; a clip with no value for an available feature receives the neutral 0.5.

**Selection** (`scenesift/selection.py`). Clips are taken in order of value until the budget is filled. A clip that overlaps a chosen one by more than 25% is skipped, one second of context is added on each side and counted against the budget, and the last clip is trimmed so the reel never exceeds the budget. With the diversity term, a clip's value is `0.7 × score − 0.3 × (highest visual similarity to a clip already chosen)`.

## 2. D: the classical computer-vision method

The classical method uses **only techniques taught in the course** and loads no pretrained model. Frames are sampled at 4 per second and reduced to 160-pixel-wide grey images; colour uses frames at 1 per second. The code is in `scenesift/features/classical.py` and `HistogramEncoder` in `scenesift/features/visual.py`.

| Cue | What it measures | Course techniques |
| --- | --- | --- |
| `residual` (scene motion) | How much moves in the scene once the camera's own motion is removed | SIFT keypoints and descriptors, descriptor matching with the ratio test, RANSAC, homography estimation, image warping, image subtraction |
| `moving` (moving area) | What share of the frame is in motion | Gaussian smoothing, image subtraction, thresholding |
| `camera` (camera motion) | How far the camera pans, tilts, or zooms | Homography applied to the frame corners |
| `edges` (edge detail) | How much structure the frame holds | Canny detector: Gaussian smoothing, Sobel gradients, non-maximum suppression, hysteresis thresholding |
| `colour` (colour change) | How unlike the rest of the video, and the preceding minute, the clip's colours are | HSV colour model, colour histograms, histogram comparison |

### Step by step

1. **Detect and describe keypoints.** SIFT finds scale-invariant keypoints in each frame and describes each with a 128-dimensional gradient histogram.
2. **Match.** Each descriptor in the earlier frame is matched to its nearest neighbour in the later frame. A match is kept only if it is clearly better than the second-nearest (distance ratio below 0.75).
3. **Fit a homography with RANSAC.** Four matches determine a homography. RANSAC repeatedly fits one to a random sample of four and keeps the homography with the most inliers (reprojection error under 3 pixels), so matches on moving objects are rejected as outliers and the homography describes the camera. It is trusted only with at least 8 inliers.
4. **Camera motion.** The homography is applied to the four frame corners; the mean distance they move is the `camera` cue.
5. **Align and subtract.** The earlier frame is warped by the homography onto the later one and the two are subtracted. The mean absolute difference over the pixels the warp covers is the `residual` cue: movement of things in the scene.
6. **Moving area.** Both frames are smoothed with a 5×5 Gaussian to suppress noise, aligned, and subtracted; the share of pixels whose difference exceeds 20 grey levels is the `moving` cue.
7. **Edges.** The Canny detector (thresholds 50 and 150) marks edge pixels; their share of the frame is the `edges` cue. Blurred, dark, or empty frames score low.
8. **Colour.** Each frame is described by an HSV histogram (8×4×4 bins) joined to a 6×6 grey thumbnail. A clip's descriptor is compared, by cosine similarity, with the whole video's and with the preceding 60 seconds'; the two dissimilarities are averaged.

When too few matches support a homography (a blank or blurred frame), or the warp covers less than half the frame, the frames are subtracted unaligned and the camera motion is treated as unknown. Samples that straddle a shot cut compare unrelated images and are discarded. Each cue is averaged over a clip's samples, normalized, and the five are summed with equal weights.

The five cues and their equal weights were chosen on SumMe, the development set, from a larger pool of course techniques (Harris corner counts, SIFT keypoint counts, Sobel gradient magnitude, plain frame differencing, brightness, saturation, contrast). Least-squares fitting of the weights was tried and did worse on held-out videos than equal weights, so it is not used.

## 3. C: the multimodal method

The multimodal method combines five kinds of evidence, three of them from pretrained models. Nothing is trained for this task.

| Feature | Weight | What it measures | How |
| --- | --- | --- | --- |
| `visual` | 0.30 | How visually distinctive the clip is | Frames at 1 per second are embedded with CLIP ViT-B/32. The feature is one minus the cosine similarity between the clip's mean embedding and the whole video's |
| `motion` | 0.20 | How much moves | Dense Farnebäck optical flow between consecutive frames; the median flow vector is subtracted to discount camera pans, and the remaining magnitude is averaged |
| `audio` | 0.15 | How loud and eventful the sound is | Short-time Fourier transform; the mean of normalized energy in dB and the 90th percentile of spectral flux |
| `speech` | 0.20 | How central what is said is | Whisper transcribes the audio into timestamped utterances, each embedded with a sentence-embedding model. The feature is the highest similarity between an utterance in the clip and the centroid of the transcript |
| `novelty` | 0.15 | How new the content is | The mean of visual novelty (CLIP dissimilarity to the preceding 60 s) and topic novelty (dissimilarity of an utterance to the preceding 60 s of speech) |

The weights were set by judgement and are not tuned. The method applies the diversity term during selection. Given a text request (command line only), `visual` becomes the mean of the clip's three highest CLIP text–image similarities and `speech` the highest similarity between an utterance and the request; the benchmarks have no requests, so that mode is not evaluated.

The code is in `scenesift/features/visual.py`, `motion.py`, `audio.py`, and `speech.py`.

## 4. The other methods

- **A, activity:** optical-flow `motion` and `audio` only. It answers how far "something is happening" goes without any model.
- **B, semantic:** CLIP `visual` only.
- **Blend:** all ten features with non-negative least-squares weights fitted on SumMe (`python -m scenesift fit`). Each clip's target is the share of it that annotators selected; features and targets are centred within each video; a feature that does not help gets zero. The fitted weights are in `configs/fitted_summe.yaml`.
- **Uniform reference:** evenly spaced 5-second clips, ignoring the content.

## 5. Evaluation measures

- **Temporal F1:** seconds of overlap between a method's summary and one annotator's, divided by each summary's length and combined as `2PR / (P + R)`; averaged over annotators.
- **Rank correlation:** a method's clip scores are turned into a curve over time (each time bin gets the mean score of the clips containing it), and Kendall's τ and Spearman's ρ are computed against each annotator's scores, then averaged. TVSum supplies a rating per 2-second shot; on SumMe the reference is the share of each second an annotator kept.
- **Event coverage, redundancy, analysis time:** see [experiments/README.md](../experiments/README.md).

## 6. Code map

```text
app.py                      Report page with a section to test a video (Gradio)
configs/default.yaml        Every setting with its default
configs/fitted_summe.yaml   Blend weights fitted on SumMe
scenesift/
  segmentation.py           Shot cuts and candidate clips
  features/classical.py     Classical cues: frame alignment, scene motion, moving area, camera motion, edges
  features/visual.py        CLIP embeddings; HSV colour-histogram descriptor
  features/motion.py        Farnebäck optical flow
  features/audio.py         Loudness and spectral change
  features/speech.py        Transcription, sentence embeddings, topic novelty
  scoring.py                Per-video normalization and weighted fusion
  selection.py              Budgeted selection, overlap removal, diversity
  pipeline.py               Orchestration, caching, per-clip evidence
  evaluation.py             Metrics and the method-comparison runner
  fitting.py                Least-squares and grid-search weight fitting
  export.py, media.py, frames.py, ffmpeg.py   Rendering, metadata, frame sampling
  cli.py                    python -m scenesift summarize | evaluate | fit
experiments/                Dataset converters, comparison and F1-demo scripts, results
research/                   Report (LaTeX), figure script, figures
tests/                      Unit tests and a synthetic-video end-to-end test
```

## 7. Known limitations

- A homography describes a rotating camera or a flat scene. With strong parallax the aligned difference still contains some camera motion.
- The classical cues respond to any motion; they cannot tell an important action from an unimportant one.
- CLIP distinctiveness and novelty reward anything unusual for the video, important or not, such as title cards and transitions.
- Per-video normalization makes scores relative: a dull video still yields "high" scores.
- Timestamps assume a constant frame rate, and the whole audio track is held in memory.
