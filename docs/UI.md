# SceneSift — Report page and demonstration

`app.py` serves one page laid out as a short paper: abstract, method, evaluation, results with tables
and figures, discussion, and a demonstration section where the five selection methods are applied to a
video you upload. It is built with Gradio and uses the pipeline described in
[IMPLEMENTATION.md](IMPLEMENTATION.md).

## Run locally

1. Install Python 3.10+.
2. Create and activate a virtual environment.
3. Without an NVIDIA GPU, run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` first.
4. Run `pip install -r requirements.txt`.
5. Run `python app.py` and open <http://127.0.0.1:7860>.

| Option | Effect |
| --- | --- |
| `--port 8000` | Serve on another port |
| `--host 0.0.0.0` | Accept connections from other machines on the network |
| `--share` | Also create a temporary public `gradio.live` link; uploaded videos pass through it |

When the app runs on a remote machine, forward the port over SSH and open the same address locally:
`ssh -L 7860:127.0.0.1:7860 user@remote`.

FFmpeg does not need to be installed separately: the app uses `ffmpeg`/`ffprobe` from `PATH` when present
and otherwise the binary shipped with `imageio-ffmpeg`. Formats OpenCV cannot read are converted to a
temporary copy for analysis, and the page says when that happened. The first analysis downloads model
weights (about 1 GB) and is slow; later runs reuse them.

## The report sections

Sections 1 to 4 are static text, tables, and the figures in `research/figures/` (drawn by
`research/make_figures.py`). A figure that has not been generated is left out. The numbers are the ones
in [SUMME_RESULTS.md](../experiments/SUMME_RESULTS.md) and [TVSUM_RESULTS.md](../experiments/TVSUM_RESULTS.md);
update the text in `app.py` when a run is repeated.

## The demonstration section

Upload a video, choose a summary length (15–180 seconds), and click **Analyse video**. The video is
analysed once and all five methods are applied to the same measurements, without a text request, as in
the benchmarks. The page then shows:

- a chart with one row per method marking the clips it selected along the video's timeline;
- a table with each method's number of clips, reel length, repetition (mean visual similarity between
  its clips), the evidence it used with the weights applied, and its analysis time per minute of video;
- one tab per method with its rendered highlight reel and a table of the selected clips: source
  timestamps, score, the evidence that was above this video's norm, and each feature's weighted
  contribution;
- a JSON file with the timestamps, scores, notes, and timings of all methods.

A method that cannot be applied to a video (for example the activity method on a still, silent video) is
left out and named in the notes. The blend uses the weights in `configs/fitted_summe.yaml` when that file
exists. **Clear** empties the section.

Analysis of a video is cached under `~/.cache/scenesift`, so analysing the same file again with a
different length is fast.

An uploaded video has no human annotations, so the demonstration shows where the methods agree and
differ, not which is correct. Scores are relative to the one video and reflect measured signals, not an
understanding of events. Selection guided by a text request is available from the command line
(`python -m scenesift summarize video.mp4 --prompt "..."`), not from this page.

## Tests

Run `python -m unittest discover -s tests -v`. The tests generate a small synthetic video and use the
dependency-light backends, so they need no downloaded models or videos. The page itself has no automated
browser tests; it was checked by calling its analysis endpoint with a video and by viewing the rendered
page.

## Limitations

- Processing is synchronous: one long video blocks the app until it finishes. Analysis takes roughly 4
  seconds per minute of video on a 16-core CPU, plus rendering five reels.
- Gradio may convert uploaded media for browser playback; outputs are written to a temporary directory
  on the machine running the app.
- A summary length at least as long as the source returns the whole video.
- Do not upload sensitive or copyrighted clips to a public link; the app listens on localhost by default.
