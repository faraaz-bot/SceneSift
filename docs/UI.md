# SceneSift — Interface

A local Gradio app for inspecting a video and generating a highlight reel with the pipeline described in
[IMPLEMENTATION.md](IMPLEMENTATION.md).

## Run locally

1. Install Python 3.10+.
2. Create and activate a virtual environment.
3. Without an NVIDIA GPU, run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` first.
4. Run `pip install -r requirements.txt`.
5. Run `python app.py` and open the local URL printed in the terminal.

FFmpeg does not need to be installed separately: the app uses `ffmpeg`/`ffprobe` from `PATH` when present
and otherwise the binary shipped with `imageio-ffmpeg`. Formats OpenCV cannot read are converted to a
temporary copy for analysis, and the status panel says when that happened. The first generation downloads model weights
(about 1 GB) and is slow; later runs reuse them.

## What it does

- Upload and preview a local video.
- Enter a natural-language request, or pick a suggested focus (general, sports, gaming, podcasts, travel)
  that fills in a request. The focus only changes the text; it does not switch detectors.
- Set a target highlight duration (15–180 seconds).
- **Inspect video** reads duration, dimensions, frame rate, and audio availability without analysing.
- **Generate highlights** runs the pipeline and shows:
  - the rendered highlight reel;
  - a table of selected clips with source timestamps, score, the evidence that was above this video's
    norm, any transcript text, and each modality's weighted contribution;
  - which evidence was used and which was unavailable (for example no speech);
  - a downloadable JSON file with timestamps, scores, notes, timings, and the settings used.
- **Selection method** switches between the proposed multimodal method and the two baselines, so the
  same video can be compared by eye.
- **Reset** clears the workspace.

Analysis of a video is cached under `~/.cache/scenesift`, so changing the request, duration, or method
on the same file is fast.

## Honest labelling

The status panel states that scores are relative to the one video and reflect measured signals, not
understanding of events. When a modality is missing or a model could not be loaded, the app says so
rather than silently producing a weaker result.

## Design rationale

Gradio keeps the course prototype to one locally runnable process, with no separate frontend or backend.
Metadata inspection stays in `scenesift/media.py`, and the app imports the heavy pipeline only when a
generation is requested, so the page loads quickly.

## Tests

Run `python -m unittest discover -s tests -v`. The tests generate a small synthetic video and use the
dependency-light backends, so they need no downloaded models or videos. The interface itself has been
exercised through its API endpoints (inspect, generate, reset); it has no automated browser tests.

## Limitations

- Processing is synchronous: one long video blocks the app until it finishes. As a reference point, a
  52-second 480p clip took about 7 seconds to analyse on a 32-core CPU, plus rendering.
- Gradio may convert uploaded media for browser playback; outputs are written to a temporary directory
  on the machine running the app.
- The duration slider tops out at 180 seconds. A request at least as long as the source returns the
  whole video.
- Do not upload sensitive or copyrighted clips to a public demo; the prototype defaults to localhost.
