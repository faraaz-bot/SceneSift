# SceneSift — Proposed Implementation Plan

> Technical design proposal, not implemented code. Choices are candidates to verify during development.

## 1. Technology stack

| Function | Proposed tool/model | Notes |
| --- | --- | --- |
| Language | Python | Research prototype |
| Decode and export | FFmpeg, optionally OpenCV | Preserve original timestamps |
| Shot/scene changes | PySceneDetect | Pair with overlapping windows |
| Visual semantics | CLIP / OpenCLIP | Sampled frames and text-image similarity |
| Motion | OpenCV optical flow | Compare after cut/camera-motion filtering |
| Audio features | librosa | Energy and changes, not “excitement” ground truth |
| Speech | Whisper | Timestamped transcripts where speech exists |
| Text relevance | Sentence Transformers | Prompt similarity, topic novelty |
| Clip selection | NumPy/scikit-learn + custom ranking | Test greedy vs diversity-aware ranking |
| User interface | Gradio | Upload, prompt, duration, previews |
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

Compare fixed weights with experimentally fitted or feature-conditioned fusion. **Never tune using held-out test annotations.** Category labels may be used for post-hoc evaluation but should not be hardcoded as inference-time routing rules. When speech or audio is missing, renormalize weights across available signals.

### F. Selection and assembly

Rank candidates under the requested summary-duration budget. Remove overlapping and near-duplicate candidates, or compare to maximal marginal relevance (MMR). Restore short context padding, order selected clips chronologically, render MP4, and retain source timestamps.

### G. Interface and observability

Provide video upload, optional text prompt, duration selection, progress information, preview, MP4 download, and timestamp list. Display feature contributions and limitations rather than claiming the system reasons perfectly about events.

## 4. Evaluation implementation

Use the same fixed held-out videos for activity-only, visual-only, and multimodal methods. Annotate reference highlight intervals; store video IDs and evaluation splits separately. Evaluate temporal overlap F1, event coverage, redundancy, human relevance/context ratings, and runtime. Cache embeddings/transcripts to keep baseline comparisons tractable; report category-level breakdowns and ablations.

## 5. Suggested repository structure (future implementation)

```text
SceneSift/
├── README.md
├── docs/
│   ├── EXPERIMENT.md
│   ├── IMPLEMENTATION.md
│   └── ARCHITECTURE.md
├── src/
│   ├── ingest/
│   ├── segmentation/
│   ├── features/
│   ├── scoring/
│   ├── selection/
│   └── export/
├── app/
├── experiments/
├── tests/
└── configs/
```

The folders beyond `docs/` are illustrative and are **not** created by the documentation PR.
