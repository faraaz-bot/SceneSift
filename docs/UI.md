# SceneSift — First UI Iteration

This PR adds a **working media inspection interface** and deliberately **does not pretend to perform highlight extraction**.

## Run locally

1. Install Python 3.10+ and FFmpeg (the `ffprobe` executable must be on PATH).
2. Create and activate a virtual environment.
3. Run `pip install -r requirements.txt`.
4. Run `python app.py`.
5. Open the local Gradio URL printed in the terminal.

## What works now

- Upload and preview a local video in the browser.
- Enter a natural-language highlight instruction.
- Select a suggested focus (general, sports, gaming, podcasts, travel).
- Set a target highlight duration (15–180 seconds).
- Click **Inspect video** to read real duration, dimensions, frame rate and audio availability.
- Display clear warnings for clips shorter than the requested summary, or missing audio.
- Reset the workspace.

## What is explicitly not implemented

No AI scoring, scene segmentation, timeline markers, highlight export, inference results or analytics. The UI labels this honestly. The next planned change can attach real segment extraction and scoring while preserving the interface.

## Design rationale

Use simple, locally runnable Gradio rather than separate frontend and backend infrastructure for the course prototype. Keep metadata inspection isolated in `scenesift/media.py` so it can be tested independently. User-selected preset changes the text prompt; it does not route into category-specific detectors.

## Tests

Run `python -m unittest discover -s tests -v`. Tests mock `ffprobe` and do not require downloaded videos.

## Limitations

Gradio may convert uploaded media for browser playback; output paths remain local. Video processing is restricted to metadata in this PR. Do not upload sensitive or copyrighted clips to a public demo; the prototype defaults to localhost.
