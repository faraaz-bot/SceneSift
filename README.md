# SceneSift

**Classical cues versus learned features for training-free video highlight extraction**

SceneSift finds the important moments of a video and assembles them into a short highlight reel. It is a course project for CPS 843 / CP 8307 (Computer Vision) built around one question:

**When no model is trained for the task, do classical computer-vision cues, features from pretrained models, or a blend of the two select highlights that agree best with human annotators?**

Five selection methods share one pipeline and are compared on two public benchmarks with human annotations, SumMe and TVSum.

## Result in brief

On the held-out TVSum benchmark, the **classical method**, built only from techniques taught in the course and using no pretrained model, agrees with annotators about as well as an optical-flow baseline and clearly better than the methods built on CLIP. The **multimodal method** leads on the development set but is at chance on the test set. A blend of all evidence does not beat the classical method alone.

| Method | SumMe F1 | SumMe τ | TVSum F1 | TVSum τ |
| --- | --- | --- | --- | --- |
| Uniform reference | 0.172 | n/a | 0.143 | n/a |
| A: activity (optical flow + audio) | 0.195 | 0.066 | 0.176 | 0.097 |
| B: semantic (CLIP visual) | 0.165 | 0.067 | 0.144 | −0.057 |
| C: multimodal | 0.207 | 0.102 | 0.149 | −0.008 |
| D: classical CV | 0.189 | 0.078 | 0.185 | 0.084 |
| Blend, weights fitted on SumMe | 0.202 | 0.102 | 0.156 | 0.060 |
| Agreement between annotators | 0.311 | 0.229 | 0.250 | 0.178 |

Mean over videos. F1 is overlap with each annotator's summary at a 15% length budget; τ is Kendall's rank correlation with each annotator's scores. SumMe (25 videos) is the development set, and the blend's SumMe figures are leave-one-video-out. TVSum (50 videos) is the test set. Paired tests, ablations, per-category results, weight tuning, and the comparison with published work are in [experiments/RESULTS.md](experiments/RESULTS.md).

## The two approaches

All methods cut the video into candidate clips, measure evidence for each clip, normalize it within the video, score each clip by a weighted sum, and select the best clips under the length budget. They differ in the evidence. Full detail is in [docs/METHODS.md](docs/METHODS.md).

### D: the classical computer-vision method

Uses only techniques from the course lectures, with no pretrained model:

| Cue | What it measures | Course techniques |
| --- | --- | --- |
| Scene motion | Movement in the scene with the camera's own motion removed | SIFT keypoints, descriptor matching with the ratio test, RANSAC, homography, image warping, image subtraction |
| Moving area | The share of the frame in motion | Gaussian smoothing, image subtraction, thresholding |
| Camera motion | How far the camera pans, tilts, or zooms | Homography applied to the frame corners |
| Edge detail | How much structure the frame holds | Canny edge detector (Gaussian smoothing, Sobel gradients, non-maximum suppression, hysteresis) |
| Colour change | How unlike the rest of the video the clip's colours are | HSV colour model, colour histograms, histogram comparison |

Consecutive frames are aligned by matching SIFT keypoints and fitting a homography with RANSAC. The homography gives the camera motion; subtracting the aligned frames gives the motion in the scene. The five cues are combined with equal weights.

### C: the multimodal method

Combines five kinds of evidence, three of them from pretrained models (nothing is trained for this task):

| Feature | Weight | How it is measured |
| --- | --- | --- |
| Visual | 0.30 | CLIP image embeddings; how distinctive the clip is within the video |
| Motion | 0.20 | Farnebäck optical flow with the median flow removed |
| Audio | 0.15 | Loudness and spectral change from a short-time Fourier transform |
| Speech | 0.20 | Whisper transcript and sentence embeddings; how central what is said is |
| Novelty | 0.15 | How unlike the preceding minute the clip's visuals and speech are |

It also penalizes clips that look like one already selected.

### Baselines and blend

**A (activity)** uses optical-flow motion and audio. **B (semantic)** uses the CLIP visual feature alone. The **blend** uses all ten features with least-squares weights fitted on SumMe. A **uniform** reference of evenly spaced clips shows what a score looks like with no skill.

## Quick start

Requires Python 3.10+. FFmpeg is used from `PATH` when installed; otherwise the binary bundled with
`imageio-ffmpeg` is used.

```bash
python -m venv .venv && source .venv/bin/activate
# Without an NVIDIA GPU, install the CPU builds first (much smaller download):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The first run downloads the CLIP, Whisper, and sentence-embedding weights (about 1 GB in total). The classical method needs none of them.

```bash
# Report page with a section to test your own video (http://127.0.0.1:7860)
python app.py

# One video from the command line: writes output/highlights.mp4 and output/highlights.json
python -m scenesift summarize video.mp4 --method classical --duration 60

# Compare the methods on annotated videos (see experiments/README.md)
python -m scenesift evaluate experiments/manifest.csv --budgets 15% --ablations

# Tests (no model downloads needed)
python -m unittest discover -s tests -v
```

## Report page

`python app.py` serves one page laid out as a short paper: abstract, method, evaluation, results with tables and figures, discussion, and a demonstration section. In the demonstration, upload a video, choose a summary length, and click **Analyse video**. All five methods are applied to the same measurements, and the page shows a timeline of what each method selected, a summary table, each method's rendered reel with its clips and evidence, and a JSON download.

| Option | Effect |
| --- | --- |
| `--port 8000` | Serve on another port |
| `--host 0.0.0.0` | Accept connections from other machines on the network |
| `--share` | Also create a temporary public `gradio.live` link; uploaded videos pass through it |

When the app runs on a remote machine, forward the port over SSH and open the same address locally: `ssh -L 7860:127.0.0.1:7860 user@remote`. Analysis is synchronous and takes roughly 5 seconds per minute of video on a 16-core CPU, plus rendering. An uploaded video has no annotations, so the demonstration shows where the methods agree and differ, not which is correct.

## Evaluation data

| | SumMe | TVSum |
| --- | --- | --- |
| Videos | 25 personal videos, 67 minutes | 50 YouTube videos, 210 minutes, 5 in each of 10 categories |
| Human labels | 15–18 people each selected the frames they would keep in a 15% summary | 20 people each scored every 2-second shot from 1 to 5 |
| Role | Development set; all design choices and the blend's weights | Test set; per-category results |

Neither dataset is redistributed here. What each contains, and the results other papers report on them, are in [docs/DATASETS.md](docs/DATASETS.md).

## Limitations

- All methods are well below human agreement and below the best published training-free (τ 0.135 on TVSum) and supervised (0.194) methods.
- The classical cues respond to any motion and cannot tell an important action from an unimportant one.
- The samples are small: 25 and 50 videos, five per TVSum category.
- The classical method was revised after a first TVSum run, using SumMe only; see the protocol in [experiments/RESULTS.md](experiments/RESULTS.md).
- The command-line tool accepts a text request (`--prompt`), which changes the CLIP and speech features. Neither benchmark has requests, so that mode is not evaluated.

## Repository

| Path | Content |
| --- | --- |
| `scenesift/` | The pipeline: candidates, features, scoring, selection, evaluation, fitting, CLI |
| `app.py` | Report page with the demonstration section |
| `configs/` | Default settings and the blend weights fitted on SumMe |
| `docs/METHODS.md` | How each method works, step by step, and the code map |
| `docs/DATASETS.md` | The two benchmarks and published results |
| `experiments/` | Dataset converters, comparison and F1-demonstration scripts, how to run, and [RESULTS.md](experiments/RESULTS.md) |
| `research/` | The report (LaTeX), its figure script, and figures |
| `tests/` | Unit tests and an end-to-end test on a synthetic video |

Third-party software: OpenCV, PySceneDetect, SciPy, OpenCLIP, faster-whisper, Sentence Transformers, FFmpeg, and Gradio.
