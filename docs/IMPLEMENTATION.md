# SceneSift — Implementation

> The design below is implemented in the `scenesift` package. Section 5 maps each stage to its module and section 6 gives the details and the design choices. Model choices remain candidates to revisit.

## 1. Technology stack

| Function | Tool/model | Notes |
| --- | --- | --- |
| Language | Python | Research prototype |
| Decode and export | OpenCV (decode), FFmpeg (audio, export) | Timestamps are source frame index / frame rate |
| Shot/scene changes | PySceneDetect | Pair with overlapping windows |
| Visual semantics | CLIP / OpenCLIP | Sampled frames and text-image similarity |
| Motion | OpenCV optical flow | Compare after cut/camera-motion filtering |
| Audio features | NumPy/SciPy STFT | Energy and spectral change, not “excitement” ground truth |
| Speech | Whisper via faster-whisper | Timestamped transcripts where speech exists |
| Text relevance | Sentence Transformers | Prompt similarity, topic novelty |
| Classical cues | OpenCV (Sobel, frame differences, colour histograms) | No pretrained model |
| Clip selection | NumPy + custom ranking | Greedy, or maximal marginal relevance for diversity |
| Weight fitting | SciPy non-negative least squares | Fitted on development videos only |
| User interface | Gradio | Report page; upload, summary length, reels for every method |
| Evaluation | Python metrics + annotation files | Reproducible splits and per-domain reports |

## 2. Inputs and outputs

**Inputs:** Authorized video file; optional natural-language query; requested summary length.

**Outputs:** Highlight-reel MP4; source-relative start/end timestamps; per-segment scores; short summary or evidence-based descriptions; evaluation artifacts.

Every candidate must use **original-video timestamps in seconds**. Metadata should record video ID, source/license, duration, frame rate, audio availability, and any language/transcript settings.

## 3. Pipeline stages

### A. Video ingestion and candidate generation

Decode frames and audio with aligned timebases. Identify shot boundaries, then add short overlapping temporal windows; major events do not always coincide with an editing cut. Preserve contextual lead-in/out and avoid truncating a continuous action at a shot boundary.

### B. Visual semantic features

Sample representative frames per candidate and compute image embeddings. With a user prompt, compare its text embedding to visual embeddings; aggregate across sampled frames (for example top-k mean). Without a prompt, visual diversity and novelty are proxies, not guarantees of importance.

### C. Temporal motion features

Measure optical-flow magnitude and temporal change at denser sampling. Guard against false highs from scene cuts, camera pans, and shake. Normalize scores within a video.

### D. Audio and speech features

Compute aligned audio intensity/change features and speech transcripts with timestamps. Score transcript segments for prompt relevance and topic novelty. Audio energy is not a reliable proxy for semantic importance by itself; quiet clips may matter greatly.

### E. Normalization and fusion

Normalize feature distributions per video using robust normalization. A conceptual fixed-weight starting point is:

```text
score = w_visual * visual_semantics
      + w_motion * motion
      + w_audio  * audio_events
      + w_speech * transcript_relevance
      + w_novelty * novelty
```

Compare fixed weights with weights fitted on the development set. **Never tune using held-out test annotations.** Category labels may be used for post-hoc evaluation but should not be hardcoded as inference-time routing rules. When speech or audio is missing, renormalize weights across available signals.

### F. Selection and assembly

Rank candidates under the requested summary-duration budget. Remove overlapping and near-duplicate candidates, or compare to maximal marginal relevance (MMR). Restore short context padding, order selected clips chronologically, render MP4, and retain source timestamps.

### G. Interface and observability

Provide video upload, summary-length selection, progress information, a reel and a timestamp list for every method, and a download of the results. Display feature contributions and limitations rather than claiming the system reasons perfectly about events. A text request is accepted by the command-line tool.

## 4. Evaluation implementation

Use the same fixed held-out videos for activity-only, visual-only, and multimodal methods. Annotate reference highlight intervals; store video IDs and evaluation splits separately. Evaluate temporal overlap F1, rank correlation with annotator scores, event coverage, redundancy, and runtime. Cache embeddings/transcripts to keep baseline comparisons tractable; report category-level breakdowns and ablations.

## 5. Repository structure

```text
SceneSift/
├── app.py                  Report page with a demonstration section (Gradio)
├── configs/default.yaml    Every setting with its default
├── scenesift/
│   ├── media.py            Metadata inspection (ffprobe, with an OpenCV/FFmpeg fallback)
│   ├── frames.py           Timestamped frame sampling
│   ├── segmentation.py     Shot cuts and candidate segments            (stage A)
│   ├── features/
│   │   ├── visual.py       CLIP embeddings; colour-histogram fallback  (stage B)
│   │   ├── motion.py       Optical flow with camera-motion removal     (stage C)
│   │   ├── classical.py    Edge strength and frame-to-frame change     (stage C)
│   │   ├── audio.py        Loudness and spectral change                (stage D)
│   │   └── speech.py       Transcription, sentence embeddings, topic novelty (stage D)
│   ├── scoring.py          Robust per-video normalization and fusion   (stage E)
│   ├── selection.py        Budgeted selection, overlap removal, MMR    (stage F)
│   ├── export.py           FFmpeg rendering                            (stage F)
│   ├── pipeline.py         Orchestration, caching, per-clip evidence
│   ├── evaluation.py       Metrics and the method-comparison runner    (section 4)
│   ├── fitting.py          Fits the blend's weights on development videos
│   └── cli.py              `python -m scenesift summarize|evaluate|fit`
├── experiments/            Dataset converters, comparison and F1-demo scripts, results; see its README
├── tests/                  Unit tests and a synthetic-video end-to-end test
├── docs/
└── research/               Paper draft
```

## 6. How the implementation realises the plan

**Candidates.** Shots of 2–12 s become candidates. Overlapping windows of 4 s and 8 s (50% overlap) cover
the whole video, with edges snapped to a shot cut within 0.75 s. Near-duplicate candidates are dropped,
and so are candidates in which more than half the sampled frames are blank (black frames, fades, solid
colour), measured by the grey-level spread of each frame.

**Evidence per candidate.** All signals are measured once per video and then aggregated per candidate.

| Feature | Raw measurement |
| --- | --- |
| `visual` | With a request: mean of the top-3 CLIP text–image similarities among the candidate's frames (1 fps). Without one: one minus the cosine similarity between the candidate's mean embedding and the whole video's |
| `motion` | Mean optical-flow magnitude (4 fps, 160 px wide) after subtracting the median flow vector, with samples that straddle a cut removed |
| `audio` | Mean of two normalized signals: average energy in dB and the 90th percentile of spectral flux |
| `speech` | With a request: highest sentence-embedding similarity between the request and an utterance in the candidate. Without one: highest similarity to the transcript centroid |
| `novelty` | Mean of visual novelty (dissimilarity to the preceding 60 s of frames) and topic novelty (dissimilarity of an utterance to the preceding 60 s of speech) |
| `colour` | Classical. Mean of two signals from colour-histogram descriptors (HSV histogram plus a coarse grey layout, 1 fps): dissimilarity to the whole video, and dissimilarity to the preceding 60 s |
| `edges` | Classical. Mean Sobel gradient magnitude of the 160 px grey frames (4 fps) |
| `change` | Classical. Mean absolute grey-level difference between consecutive samples (4 fps), with samples that straddle a cut removed |

**Normalization and fusion.** Each raw feature is mapped to (0, 1) within the video by a logistic of its
median/MAD z-score. The score is the weighted sum in section 3E with fixed weights
(visual 0.30, motion 0.20, audio 0.15, speech 0.20, novelty 0.15; the three classical-only features
0.20 each). These are untuned starting values. When
a modality is unavailable for the video its weight is redistributed over the rest; a candidate with no
value for an available modality (for example no speech in that window) receives the neutral 0.5.

**Compared methods.** A (`activity`) uses `motion` and `audio`; B (`semantic`) uses `visual`; C
(`multimodal`) uses the first five features; D (`classical`) uses `motion`, `colour`, `edges`, and
`change`, none of which needs a pretrained model; the `blend` uses all eight. All share the same
candidates, normalization, context padding, and budget handling. Only C and the blend apply the
diversity constraint by default; the evaluation runner can also run C without it, and with each
modality removed.

**Fitted weights.** `python -m scenesift fit` fits the blend's weights on annotated development videos.
Each candidate's target is the share of it that annotators selected, averaged over annotators.
Features and targets are centred within each video and weights are found by non-negative least
squares, then scaled to sum to one, so a feature that does not help gets zero. The command also
reports leave-one-video-out scores, where each video is scored with weights fitted on the others.
The weights are written to a YAML file under `method_weights`, keyed by method; a method without an
entry keeps the fixed weights. `--method` fits another method's weights the same way, and
`--how search` replaces the regression with a search over a grid of weight combinations (steps of
0.1) for the best mean F1.

**Selection.** Candidates are taken in order of value. With diversity on, value is
`0.7 × score − 0.3 × (highest visual similarity to a clip already chosen)`. A candidate that overlaps a
chosen clip by more than 25% of the shorter one is skipped. One second of context is added on each side
and counts toward the budget, which is measured on the merged timeline so overlapping context is
counted once. The last clip is trimmed to the budget that remains, so the reel never exceeds the
requested length and normally fills it to within two seconds. Clips are merged where they touch and
rendered in source order.

**Caching and timing.** Prompt-independent analysis is cached per file and settings. Stage timings
(decode, motion, visual, classical, shots, audio, speech) are recorded without model-loading time, and the
evaluation runner charges each method only the stages it needs.

**Rank correlation.** A method's candidate scores are turned into a curve over time: each time bin gets
the mean score of the candidates that contain its centre. Kendall's τ and Spearman's ρ are computed
between that curve and each annotator's scores, then averaged over annotators. TVSum supplies a score
per 2-second shot. SumMe supplies selections, so each annotator's reference there is the share of
every one-second bin they kept. The uniform reference has no scores and gets no rank correlation.

### Design choices

- Audio features use SciPy rather than librosa, to avoid a heavy dependency for two simple measurements.
- Speech uses faster-whisper (the same Whisper models on CTranslate2) for CPU speed, with its
  voice-activity filter on. The filter avoids invented text in non-speech audio but can drop speech
  under loud music; `whisper_vad: false` turns it off.
- Fitted weights are linear and non-negative. Feature-conditioned fusion is not implemented.
- When CLIP, Whisper, or the sentence model cannot be loaded, the pipeline continues with what it has
  and reports the gap. The colour-histogram fallback is not semantic and cannot use the request.
- When OpenCV cannot decode a file (Theora `.ogv`, for example), FFmpeg makes a temporary 360p H.264
  copy for analysis. The timeline is unchanged and the reel is still cut from the original.

### Known limitations

- Timestamps assume a constant frame rate.
- Visual distinctiveness and novelty reward anything unusual for the video, important or not. In tests
  on real footage this selected title and logo cards, the one colour clip in a black-and-white reel, and
  a clip with a burned-in watermark. Blank frames are filtered; these other cases are not.
- On silent sports footage the multimodal method tended to pick cutaways (bench, crowd, hoop close-ups)
  where the activity baseline picked live play.
- Per-video normalization makes scores relative: a dull video still yields "high" scores.
- The diversity term uses visual similarity only.
- The whole audio track is held in memory, which limits very long videos.
