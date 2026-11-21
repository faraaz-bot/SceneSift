"""SceneSift interface: inspect a video, then generate a highlight reel with the multimodal pipeline."""
from __future__ import annotations

import inspect
import os
import tempfile

import gradio as gr

from scenesift.media import format_duration, inspect_video

PRESETS = {
    "General highlights": "Find the most meaningful and visually distinct moments.",
    "Sports": "Identify important plays, reactions, and turning points.",
    "Gaming": "Identify meaningful gameplay events, successes, and turning points.",
    "Podcasts": "Find the most informative ideas and key statements.",
    "Travel": "Find distinctive places, activities, and memorable moments.",
}

METHODS = {
    "Multimodal (proposed)": "multimodal",
    "Activity baseline (motion + audio)": "activity",
    "Semantic baseline (visual only)": "semantic",
}

CLIP_HEADERS = ["#", "Start", "End", "Length", "Score", "Why it was selected", "Evidence (weighted)"]
IDLE_STATUS = ("Upload a video, then click **Inspect video** to check it or **Generate highlights** "
               "to build a reel.")
IDLE_DETAILS = [["Video", "Not selected"], ["Analysis", "Not started"]]

CSS = """
:root { color-scheme: dark; }
.gradio-container { max-width: 1200px !important; margin: auto !important; }
#hero { background: linear-gradient(115deg,#10182b 4%,#192b4b 58%,#102a31 100%);
        border: 1px solid #2f486c; border-radius: 22px; padding: 28px; margin: 14px 0 24px; }
#hero h1 {font-size: 2.5rem; font-weight: 800; margin-bottom: 8px; color: white;}
#hero p {color:#c9dbed; font-size: 1.04rem; line-height:1.6;}
#hero .eyebrow {color:#8fe5cc; text-transform:uppercase; letter-spacing:.16em; font-weight:700;}
#hero .pill {background:#294e60; border:1px solid #467589; padding:6px 12px; border-radius:99px; color:#d5f9f1;}
#app-panel {border:1px solid var(--border-color-primary); border-radius:14px; padding:16px;}
"""

# Gradio 6 moved styling from Blocks() to launch(); support both.
_STYLE = {"css": CSS, "theme": gr.themes.Soft()}
_STYLE_ON_BLOCKS = "css" in inspect.signature(gr.Blocks.__init__).parameters


def apply_preset(preset: str) -> str:
    return PRESETS.get(preset, PRESETS["General highlights"])


def _timestamp(seconds: float) -> str:
    minutes, rest = divmod(float(seconds), 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


def _metadata_rows(video_path: str, details: dict, preset: str) -> list[list[str]]:
    return [
        ["Source", os.path.basename(video_path)],
        ["Preset", preset],
        ["Duration", format_duration(details["duration_seconds"])],
        ["Resolution", f'{details["width"]} × {details["height"]}'],
        ["FPS", f'{details["fps"]:.2f}'],
        ["Audio stream", "Available" if details["has_audio"] else "Not found"],
    ]


def prepare_session(video_path: str | None, prompt: str, seconds: int, preset: str):
    """Read the video's metadata without running any analysis."""
    if not video_path:
        return ("### Upload a video first\nSelect an MP4, MOV, MKV, or WebM file and try again.",
                IDLE_DETAILS, None)
    try:
        details = inspect_video(video_path)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        return (f"### Could not inspect video\n{exc}",
                [["Video", "Inspection failed"], ["Analysis", "Not started"]], None)
    request = prompt.strip() or PRESETS["General highlights"]
    warnings = []
    if seconds >= details["duration_seconds"]:
        warnings.append("The requested highlight is at least as long as the source video.")
    if not details["has_audio"]:
        warnings.append("No audio stream found. Audio and speech evidence will be skipped.")
    warning_text = "\n".join(f"- {w}" for w in warnings)
    overview = (
        "### Video ready\n"
        "Metadata only so far. Click **Generate highlights** to analyse it.\n\n"
        f"**Instruction:** {request}\n\n"
        f"**Requested length:** {format_duration(seconds)}\n\n"
        + (f"**Notes:**\n{warning_text}\n" if warning_text else "")
    )
    return overview, _metadata_rows(video_path, details, preset) + [["Analysis", "Not started"]], video_path


def generate(video_path: str | None, prompt: str, seconds: int, preset: str, method_label: str,
             progress=gr.Progress()):
    """Run the pipeline and return the reel, the selected clips, and the evidence behind them."""
    if not video_path:
        return ("### Upload a video first\nSelect an MP4, MOV, MKV, or WebM file and try again.",
                IDLE_DETAILS, None, [], None)
    from scenesift.pipeline import summarize  # heavy imports only when analysis is requested

    method = METHODS.get(method_label, "multimodal")
    request = prompt.strip() or PRESETS["General highlights"]
    out_dir = tempfile.mkdtemp(prefix="scenesift-")
    try:
        details = inspect_video(video_path)
        summary = summarize(video_path, prompt=request, target_seconds=float(seconds), method=method,
                            out_dir=out_dir, progress=lambda fraction, message: progress(fraction, desc=message))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        return (f"### Highlight generation failed\n{exc}",
                [["Video", os.path.basename(video_path)], ["Analysis", "Failed"]], None, [], None)

    rows = []
    for number, clip in enumerate(summary.clips, start=1):
        evidence = " · ".join(f"{name} {value:.2f}" for name, value in
                              sorted(clip["evidence"].items(), key=lambda item: -item[1]))
        why = clip["rationale"] + (f' — “{clip["transcript"]}”' if clip["transcript"] else "")
        rows.append([number, _timestamp(clip["start"]), _timestamp(clip["end"]),
                     f'{clip["end"] - clip["start"]:.1f}s', f'{clip["score"]:.2f}', why, evidence])
    notes = "\n".join(f"- {note}" for note in summary.notes)
    status = (
        "### Highlights ready\n"
        f"{summary.describe()}\n\n"
        f"**Instruction:** {request}\n\n"
        + (f"**Notes:**\n{notes}\n\n" if notes else "")
        + "_Scores are relative to this video and reflect measured signals (visual similarity, motion, "
          "audio, transcript). They are not a claim that the system understood the events._"
    )
    seconds_spent = sum(summary.timings.values())
    metadata = _metadata_rows(video_path, details, preset) + [
        ["Method", method_label],
        ["Clips selected", str(len(summary.clips))],
        ["Highlight length", format_duration(summary.total_seconds)],
        ["Processing time", f"{seconds_spent:.1f}s"],
    ]
    return status, metadata, summary.output_path, rows, summary.json_path


def reset():
    return (None, PRESETS["General highlights"], 60, "General highlights", next(iter(METHODS)),
            IDLE_STATUS, IDLE_DETAILS, None, None, [], None)


with gr.Blocks(title="SceneSift | Intelligent Video Highlights", **(_STYLE if _STYLE_ON_BLOCKS else {})) as demo:
    gr.HTML("""<section id="hero">
       <div class="eyebrow">SCENESIFT / VIDEO INTELLIGENCE</div>
       <h1>Find the moments that matter.</h1>
       <p>Upload a video, describe the moments you care about, and get a highlight reel with the
       source timestamps and the evidence behind each clip.</p>
       <span class="pill">Research prototype · visual, motion, audio and speech evidence</span>
    </section>""")
    with gr.Row(equal_height=False):
        with gr.Column(scale=6, elem_id="app-panel"):
            gr.Markdown("### 01 / Source video")
            video = gr.Video(label="Upload video", sources=["upload"], format=None, height=340)
            gr.Markdown("### 02 / Highlight preferences")
            with gr.Row():
                preset = gr.Dropdown(list(PRESETS), value="General highlights", label="Suggested focus",
                                     interactive=True)
                duration = gr.Slider(minimum=15, maximum=180, step=15, value=60,
                                     label="Target duration (seconds)")
            prompt = gr.Textbox(label="What should SceneSift look for?", value=PRESETS["General highlights"],
                                lines=3, placeholder="E.g. Find the most exciting plays and celebrations")
            method = gr.Dropdown(list(METHODS), value=next(iter(METHODS)), label="Selection method",
                                 info="The baselines exist for comparison with the proposed method.",
                                 interactive=True)
            with gr.Row():
                generate_button = gr.Button("Generate highlights", variant="primary")
                inspect_button = gr.Button("Inspect video", variant="secondary")
                clear_button = gr.Button("Reset", variant="secondary")
        with gr.Column(scale=5, elem_id="app-panel"):
            gr.Markdown("### 03 / Project workspace")
            status = gr.Markdown(IDLE_STATUS)
            details = gr.Dataframe(headers=["Property", "Value"], datatype=["str", "str"], interactive=False,
                                   value=IDLE_DETAILS, label="Video metadata")
            preview = gr.Video(label="Source preview", interactive=False, height=250)
    gr.Markdown("### 04 / Highlights")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            reel = gr.Video(label="Highlight reel", interactive=False, height=340)
            report = gr.File(label="Timestamps, scores and settings (JSON)", interactive=False)
        with gr.Column(scale=7):
            clips = gr.Dataframe(headers=CLIP_HEADERS, interactive=False, wrap=True,
                                 label="Selected clips (source timestamps)")
    preset.change(apply_preset, inputs=preset, outputs=prompt)
    inspect_button.click(prepare_session, inputs=[video, prompt, duration, preset],
                         outputs=[status, details, preview])
    generate_button.click(generate, inputs=[video, prompt, duration, preset, method],
                          outputs=[status, details, reel, clips, report])
    clear_button.click(reset, outputs=[video, prompt, duration, preset, method, status, details, preview,
                                       reel, clips, report])

if __name__ == "__main__":
    demo.launch(**({} if _STYLE_ON_BLOCKS else _STYLE))
