"""SceneSift UI foundation. Video inspection works; highlight generation is future work."""
from __future__ import annotations

import os
import gradio as gr
from scenesift.media import inspect_video, format_duration

PRESETS = {
    "General highlights": "Find the most meaningful and visually distinct moments.",
    "Sports": "Identify important plays, reactions, and turning points.",
    "Gaming": "Identify meaningful gameplay events, successes, and turning points.",
    "Podcasts": "Find the most informative ideas and key statements.",
    "Travel": "Find distinctive places, activities, and memorable moments.",
}

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

def apply_preset(preset: str) -> str:
    return PRESETS.get(preset, PRESETS["General highlights"])

def prepare_session(video_path: str | None, prompt: str, seconds: int, preset: str):
    """Inspect uploaded media; do not imply that highlight inference occurred."""
    if not video_path:
        return (
            "### Upload a video first\nSelect an MP4, MOV, MKV, or WebM file and try again.",
            [["Video", "Not selected"], ["Analysis", "Not started"]],
            None,
        )
    try:
        details = inspect_video(video_path)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        return (
            f"### Could not inspect video\n{str(exc)}",
            [["Video", "Inspection failed"], ["Analysis", "Not started"]],
            None,
        )
    request = prompt.strip() or PRESETS["General highlights"]
    warnings = []
    if seconds > details["duration_seconds"]:
        warnings.append("Requested highlight is longer than the source video.")
    if not details["has_audio"]:
        warnings.append("No audio stream found. Audio/speech scoring would be unavailable.")
    warning_text = "\n".join(f"- {w}" for w in warnings)
    overview = (
        "### Video ready for analysis\n"
        "Media inspection is complete. **AI highlight detection and export are not yet implemented.**\n\n"
        f"**Instruction:** {request}\n\n"
        f"**Requested length:** {format_duration(seconds)}\n\n"
        + (f"**Notes:**\n{warning_text}\n" if warning_text else "")
    )
    rows = [
        ["Source", os.path.basename(video_path)],
        ["Preset", preset],
        ["Duration", format_duration(details["duration_seconds"])],
        ["Resolution", f'{details["width"]} × {details["height"]}'],
        ["FPS", f'{details["fps"]:.2f}'],
        ["Audio stream", "Available" if details["has_audio"] else "Not found"],
        ["Highlight generation", "Not implemented"],
    ]
    return overview, rows, video_path

with gr.Blocks(title="SceneSift | Intelligent Video Highlights", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""<section id="hero">
       <div class="eyebrow">SCENESIFT / VIDEO INTELLIGENCE</div>
       <h1>Find the moments that matter.</h1>
       <p>Upload a video, describe the moments you care about, and configure a highlight reel.
       This first version previews and inspects videos; AI extraction will be added in a later PR.</p>
       <span class="pill">UI foundation · analysis not yet connected</span>
    </section>""")
    with gr.Row(equal_height=False):
        with gr.Column(scale=6, elem_id="app-panel"):
            gr.Markdown("### 01 / Source video")
            video = gr.Video(label="Upload video", sources=["upload"], format=None, height=340)
            gr.Markdown("### 02 / Highlight preferences")
            with gr.Row():
                preset = gr.Dropdown(list(PRESETS), value="General highlights", label="Suggested focus", interactive=True)
                duration = gr.Slider(minimum=15, maximum=180, step=15, value=60, label="Target duration (seconds)")
            prompt = gr.Textbox(label="What should SceneSift look for?", value=PRESETS["General highlights"], lines=3,
                                placeholder="E.g. Find the most exciting plays and celebrations")
            with gr.Row():
                inspect_button = gr.Button("Inspect video", variant="primary")
                clear_button = gr.Button("Reset", variant="secondary")
        with gr.Column(scale=5, elem_id="app-panel"):
            gr.Markdown("### 03 / Project workspace")
            status = gr.Markdown("Upload a video and click **Inspect video** to see its metadata. No highlights are generated yet.")
            details = gr.Dataframe(headers=["Property", "Value"], datatype=["str","str"], interactive=False,
                                   value=[["Video","Not selected"],["Analysis","Not started"]], label="Video metadata")
            preview = gr.Video(label="Source preview", interactive=False, height=250)
            gr.Markdown("**Coming next:** scene timeline, scored candidates, editable clip selection, rendered preview, and MP4 export.")
    preset.change(apply_preset, inputs=preset, outputs=prompt)
    inspect_button.click(prepare_session, inputs=[video,prompt,duration,preset], outputs=[status,details,preview])
    clear_button.click(
        lambda: (None, PRESETS["General highlights"], 60, "General highlights",
                 "Upload a video and click **Inspect video** to see its metadata. No highlights are generated yet.",
                 [["Video","Not selected"],["Analysis","Not started"]], None),
        outputs=[video,prompt,duration,preset,status,details,preview],
    )

if __name__ == "__main__":
    demo.launch()
