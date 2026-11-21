"""SceneSift report page: the method and results as a short paper, with a section to test a video.

    python app.py                      # http://127.0.0.1:7860
    python app.py --host 0.0.0.0       # reachable from other machines on the network
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from scenesift.media import format_duration, inspect_video

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "research" / "figures"
FITTED_CONFIG = ROOT / "configs" / "fitted_summe.yaml"

# Display name, method key, and the colour used for the method in the report's figures.
METHODS = [
    ("A: Activity", "activity", "#2a78d6"),
    ("B: Semantic", "semantic", "#eb6834"),
    ("C: Multimodal", "multimodal", "#1baf7a"),
    ("D: Classical", "classical", "#eda100"),
    ("Blend", "blend", "#e87ba4"),
]
CLIP_HEADERS = ["#", "Start", "End", "Length", "Score", "Why it was selected", "Evidence (weighted)"]
COMPARE_HEADERS = ["Method", "Clips", "Reel length", "Repetition", "Evidence used", "Analysis time"]
IDLE_STATUS = "No video has been analysed yet."

CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; background: #ffffff !important;
                    font-family: Georgia, 'Times New Roman', Times, serif !important; color: #111111; }
.gradio-container .prose, .gradio-container .prose p, .gradio-container .prose li,
.gradio-container .prose td, .gradio-container .prose th {
    font-family: Georgia, 'Times New Roman', Times, serif !important; font-size: 16px; line-height: 1.55;
    color: #111111; }
.gradio-container .prose h1 { font-size: 27px; font-weight: 400; text-align: center; line-height: 1.3;
                              margin: 28px 0 6px; }
.gradio-container .prose h2 { font-size: 20px; font-weight: 700; margin: 30px 0 8px; }
.gradio-container .prose h3 { font-size: 17px; font-weight: 700; font-style: italic; margin: 18px 0 6px; }
.gradio-container .prose p { text-align: justify; }
#byline p { text-align: center; color: #333333; margin: 0; }
#abstract { border-top: 1px solid #111111; border-bottom: 1px solid #111111; padding: 4px 0; margin: 14px 0; }
.caption p { font-size: 14px !important; text-align: left !important; color: #333333; margin-top: -4px; }
.gradio-container .prose table { border-collapse: collapse; margin: 10px auto; border-top: 2px solid #111111;
                                 border-bottom: 2px solid #111111; }
.gradio-container .prose th { border-bottom: 1px solid #111111; font-weight: 700; }
.gradio-container .prose th, .gradio-container .prose td { border-left: none; border-right: none;
                                                           padding: 4px 12px; font-size: 15px; }
.gradio-container .prose td { border-top: none; border-bottom: none; }
.figure .icon-button-wrapper { display: none !important; }
.figure.narrow { max-width: 560px; margin-left: auto !important; margin-right: auto !important; }
footer { display: none !important; }
"""
# The page is written for a light background; switch to it when the browser asks for a dark theme.
FORCE_LIGHT = """() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url);
    }
}"""
# Gradio 6 moved styling from Blocks() to launch(); support both.
_STYLE = {"css": CSS, "theme": gr.themes.Base(), "js": FORCE_LIGHT}
_STYLE_ON_BLOCKS = "css" in inspect.signature(gr.Blocks.__init__).parameters

ABSTRACT = """
**Abstract.** We study whether hand-built computer-vision cues, features from pretrained models, or a
blend of the two best select the important moments of a video when no model is trained for the task.
SceneSift cuts a video into candidate clips, measures eight kinds of evidence for each clip, and picks
the highest-scoring clips that fit a length budget. Five selection methods are compared on two public
benchmarks with human annotations: SumMe (25 videos), used for development, and TVSum (50 videos in 10
categories), held out and run once. On TVSum, a classical method that uses only optical-flow motion,
colour histograms, edge strength, and frame differences reaches a Kendall's τ of 0.088 with annotator
importance scores, close to a motion-and-audio baseline (0.097) and well above CLIP-based semantic
(−0.057) and multimodal (−0.009) ranking. A blend with weights fitted on SumMe (0.028) does not improve
on the classical method. All methods remain below human agreement (0.178) and below the best published
training-free (0.135) and supervised (0.194) results.
"""

METHOD = """
## 1. Method

Every video goes through the same steps (Figure 1); no category label is used. The video is cut into
candidate clips: shots between 2 and 12 seconds long, plus overlapping windows of 4 and 8 seconds. Eight
kinds of evidence are measured for each clip, each is normalized within the video to the range 0 to 1 by
a logistic function of its median/MAD z-score, and a weighted sum gives the clip's score. Clips are then
chosen in order of score until 15% of the video's length is filled, skipping clips that overlap one
already chosen.
"""

METHOD_TABLE = """
The evidence is of three kinds. *Classical cues* are measured directly from pixels with no pretrained
model: motion (dense optical flow with the median flow vector removed, so camera pans are discounted),
colour (how unlike the rest of the video, and the preceding minute, a clip's colour histogram is), edges
(mean Sobel gradient magnitude), and frame change (mean absolute grey-level difference between
consecutive frames). The *signal cue* is audio loudness and spectral change. *Learned features* come
from pretrained models: CLIP image embeddings, Whisper transcripts with sentence embeddings, and novelty
relative to the preceding minute. The five compared methods differ only in the evidence they use
(Table 1).

**Table 1.** Compared methods.

| Method | Evidence | Weights |
| --- | --- | --- |
| A: Activity | motion, audio | fixed |
| B: Semantic | CLIP visual | fixed |
| C: Multimodal | CLIP visual, motion, audio, speech, novelty | fixed |
| D: Classical | motion, colour, edges, frame change | fixed |
| Blend | all eight | fitted on SumMe by non-negative least squares |
"""

EVALUATION = """
## 2. Evaluation

**Data.** SumMe has 25 user videos (67 minutes), each summarized by 15 to 18 people. TVSum has 50 web
videos (210 minutes) in 10 categories, each with importance ratings from 20 people for every 2-second
shot. SumMe was the development set, on which the blend's weights were fitted. TVSum was held out: no
method was run on it until all methods and weights were fixed, and it was then run once.

**Measures.** Temporal F1 is the overlap, in seconds, between a method's 15% summary and one annotator's
summary, divided by each summary's length and combined as 2 · precision · recall / (precision + recall)
(Figure 2). Kendall's τ is the rank correlation between a method's score over time and one annotator's
ratings; it is 1 for identical orderings and 0 on average for random scores. Both are computed against
each annotator separately and averaged. A uniform reference of evenly spaced clips shows what a score
looks like with no skill, and agreement between annotators gives a ceiling.
"""

RESULTS = """
## 3. Results

**Table 2.** Mean over videos. On SumMe the blend is scored by leave-one-video-out.

| Method | SumMe F1 | SumMe τ | TVSum F1 | TVSum τ |
| --- | --- | --- | --- | --- |
| Uniform reference | 0.172 | n/a | 0.143 | n/a |
| A: Activity | 0.195 | 0.066 | 0.176 | 0.097 |
| B: Semantic | 0.165 | 0.067 | 0.144 | −0.057 |
| C: Multimodal | 0.207 | 0.102 | 0.150 | −0.009 |
| D: Classical | 0.186 | 0.046 | 0.180 | 0.088 |
| Blend | 0.193 | 0.080 | 0.147 | 0.028 |
| Agreement between annotators | 0.311 | 0.229 | 0.250 | 0.178 |

On SumMe the multimodal method C has the highest scores, but with 25 videos the only differences that
can be distinguished are that C exceeds B on both measures and exceeds D on rank correlation. No method
is distinguishable from evenly spaced clips on F1.

On the held-out TVSum the ordering reverses (Figure 3). The two methods that use no CLIP features, A and
D, have the highest rank correlation and F1. D exceeds C by 0.096 in τ (95% bootstrap interval 0.059 to
0.134) and is better on 40 of the 50 videos; it exceeds B by 0.145. D is the only method distinguishably
above evenly spaced clips on F1 (+0.037, interval 0.009 to 0.068). The blend fitted on SumMe is above C
but below D by 0.059 (interval 0.027 to 0.093). Analysis costs about 1.3 seconds per minute of video for
D and 4.3 seconds for C on a CPU.
"""

RESULTS_DETAIL = """
Removing one feature at a time from C (Figure 4) shows the cause: without the CLIP visual feature, C's τ
on TVSum rises from −0.009 to 0.065, and without motion it falls to −0.047. By category (Figure 5), A
and D are positive almost everywhere and strongest where the important moments involve visible action;
the CLIP-based methods match them only for sandwich making, where the important steps look different
from the rest of the video.

Tuning C's weights does not change the conclusion (Table 3). Weights chosen on SumMe score below the
untuned ones on held-out SumMe videos and barely move TVSum. Weights fitted within TVSum by
leave-one-video-out bring C level with D, but only by placing two thirds of the weight on motion.

**Table 3.** Tuning the multimodal weights. TVSum values were computed after the held-out run.

| Weights for C | SumMe F1, held out | TVSum τ |
| --- | --- | --- |
| Untuned | 0.207 | −0.009 |
| Least squares on SumMe | 0.192 | 0.013 |
| Grid search on SumMe | 0.180 | 0.014 |
| Least squares within TVSum, cross-validated | not run | 0.086 |
| D: Classical, untuned | 0.186 | 0.088 |

Against published work on TVSum, D (0.088) and A (0.097) are above the trained dppLSTM (0.042) and
DR-DSN (0.020) as measured by Otani et al., below the training-free method of Pang et al. (0.135), and
below supervised methods such as CSTA (0.194). On SumMe no method reaches the 0.234 F1 that Gygli et al.
report for hand-built features.
"""

DISCUSSION = """
## 4. Discussion

Without a text request, the CLIP visual feature scores how unlike the rest of the video a clip looks. On
SumMe's unedited recordings that is mildly useful. On TVSum's edited web videos it is negatively related
to what annotators rated as important, plausibly because the segments that look least like the rest are
title cards, credits, cutaways, and transitions. Motion and frame change mark moments where something is
happening, and that carried over from one dataset to the other.

Fitting the blend on 25 SumMe videos did not help: the fit favoured audio, novelty, and visual
distinctiveness, which do not predict importance on TVSum. No linear weighting of these features
exceeded a τ of about 0.09 on TVSum, so better features, not better weights, are what would close the
gap to published methods.

The samples are small (25 and 50 videos, five per TVSum category), TVSum F1 is known to separate methods
poorly, and selection guided by a text request was not evaluated because neither benchmark has requests.
All methods are well below human agreement. Figure 6 shows a typical video in detail.
"""

DEMONSTRATION = """
## 5. Demonstration on a new video

The five methods can be applied to any video below. Each method selects clips from the same measurements
of the video and its reel is rendered. No human annotations exist for an uploaded video, so the output
shows where the methods agree and differ, not which is correct. Scores are relative to the one video and
reflect measured signals, not an understanding of events.
"""

REFERENCES = """
## References

1. M. Gygli, H. Grabner, H. Riemenschneider, L. Van Gool. Creating Summaries from User Videos. ECCV, 2014.
2. Y. Song, J. Vallmitjana, A. Stent, A. Jaimes. TVSum: Summarizing Web Videos Using Titles. CVPR, 2015.
3. M. Otani, Y. Nakashima, E. Rahtu, J. Heikkilä. Rethinking the Evaluation of Video Summaries. CVPR, 2019.
4. Z. Pang, Y. Nakashima, M. Otani, H. Nagahara. Unleashing the Power of Contrastive Learning for
   Zero-Shot Video Summarization. Journal of Imaging 10(9), 2024.
5. J. Son, J. Park, K. Kim. CSTA: CNN-based Spatiotemporal Attention for Video Summarization.
   arXiv:2405.11905, 2024.
6. A. Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.
7. A. Radford et al. Robust Speech Recognition via Large-Scale Weak Supervision. ICML, 2023.

The full report, with all tables, is `research/main.tex` in the project repository.
"""


def _timestamp(seconds: float) -> str:
    minutes, rest = divmod(float(seconds), 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


def _config():
    """Default settings plus the fitted blend weights when the repository provides them."""
    from scenesift.config import Config

    return Config.from_file(FITTED_CONFIG) if FITTED_CONFIG.is_file() else Config()


def _figure(name: str, caption: str, narrow: bool = False) -> None:
    """A report figure with its caption; skipped when the image has not been generated."""
    path = FIGURES / f"{name}.png"
    if path.is_file():
        gr.Image(value=str(path), show_label=False, interactive=False, container=False,
                 elem_classes=["figure", "narrow"] if narrow else ["figure"])
        gr.Markdown(caption, elem_classes="caption")


def _comparison_chart(summaries: dict, duration: float, out_file: str) -> str:
    """One row per method showing which parts of the video it picked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {key: name for name, key, _ in METHODS}
    colours = {key: colour for _, key, colour in METHODS}
    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(10, 0.5 * len(summaries) + 1.0))
    for row, (key, summary) in enumerate(summaries.items()):
        axis.barh(row, duration, left=0, height=0.62, color="#eeede9", linewidth=0)
        for start, end in summary.intervals():
            axis.barh(row, end - start, left=start, height=0.62, color=colours[key], linewidth=0)
    axis.set_yticks(range(len(summaries)), [names[key] for key in summaries], fontsize=11)
    axis.invert_yaxis()
    axis.set_xlim(0, duration)
    step = next((s for s in (5, 10, 15, 30, 60, 120, 300, 600) if duration / s <= 12), 1200)
    ticks = list(range(0, int(duration) + 1, step))
    axis.set_xticks(ticks, [f"{t // 60}:{t % 60:02d}" for t in ticks], fontsize=10, color="#333333")
    axis.set_xlabel("Time in the video; bars are the clips each method selected", fontsize=10, color="#333333")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.savefig(out_file, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_file


def _clip_rows(summary) -> list[list]:
    rows = []
    for number, clip in enumerate(summary.clips, start=1):
        evidence = " · ".join(f"{name} {value:.2f}" for name, value in
                              sorted(clip["evidence"].items(), key=lambda item: -item[1]))
        why = clip["rationale"] + (f' — “{clip["transcript"]}”' if clip["transcript"] else "")
        rows.append([number, _timestamp(clip["start"]), _timestamp(clip["end"]),
                     f'{clip["end"] - clip["start"]:.1f}s', f'{clip["score"]:.2f}', why, evidence])
    return rows


def _empty(status: str):
    return (status, [], None, [], *([None, []] * len(METHODS)), None)


def analyse(video_path: str | None, seconds: int, progress=gr.Progress()):
    """Apply every method to the uploaded video: a comparison, then each method's reel and clips."""
    if not video_path:
        return _empty("**Upload a video first.** MP4, MOV, MKV, and WebM files are accepted.")
    from scenesift.evaluation import METHOD_STAGES  # heavy imports only when analysis is requested
    from scenesift.pipeline import analyze, summarize

    cfg = _config()
    out_dir = Path(tempfile.mkdtemp(prefix="scenesift-"))
    try:
        details = inspect_video(video_path)
        analysis = analyze(video_path, cfg, lambda fraction, message: progress(0.7 * fraction, desc=message))
        summaries, skipped = {}, []
        for index, (name, key, _) in enumerate(METHODS):
            progress(0.7 + 0.3 * index / len(METHODS), desc=f"Selecting and rendering: {name}")
            try:
                summaries[key] = summarize(video_path, target_seconds=float(seconds), method=key, cfg=cfg,
                                           analysis=analysis, out_dir=out_dir / key)
            except ValueError as exc:  # for example the activity method on a still, silent video
                skipped.append(f"{name} could not be applied: {exc}")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        return _empty(f"**The video could not be analysed.** {exc}")
    if not summaries:
        return _empty("**No method could be applied to this video.**")

    minutes = max(analysis.duration / 60.0, 1e-6)
    names = {key: name for name, key, _ in METHODS}
    comparison = []
    for key, summary in summaries.items():
        cost = sum(analysis.timings.get(stage, 0.0) for stage in METHOD_STAGES[key])
        comparison.append([names[key], len(summary.clips), format_duration(summary.total_seconds),
                           "n/a" if summary.redundancy is None else f"{summary.redundancy:.2f}",
                           ", ".join(f"{name} {weight:.0%}" for name, weight in summary.weights.items()),
                           f"{cost / minutes:.1f} s per minute of video"])
    chart = _comparison_chart(summaries, analysis.duration, str(out_dir / "comparison.png"))
    report = out_dir / "scenesift_results.json"
    report.write_text(json.dumps({key: summary.to_dict() for key, summary in summaries.items()}, indent=2),
                      encoding="utf-8")
    notes = list(dict.fromkeys([*analysis.warnings, *skipped]))
    status = (
        f"**Analysed {os.path.basename(video_path)}.** Each method selected about "
        f"{format_duration(min(float(seconds), analysis.duration))} from a "
        f"{format_duration(analysis.duration)} video. Repetition is the mean visual similarity between a "
        "method's clips; lower means more varied."
        + ("\n\n" + "\n".join(f"- {note}" for note in notes) if notes else "")
    )
    metadata = [
        ["Duration", format_duration(details["duration_seconds"])],
        ["Resolution", f'{details["width"]} × {details["height"]}'],
        ["Frame rate", f'{details["fps"]:.2f}'],
        ["Audio", "Available" if details["has_audio"] else "Not found"],
    ]
    per_method = []
    for _, key, _ in METHODS:
        summary = summaries.get(key)
        per_method += [summary.output_path, _clip_rows(summary)] if summary else [None, []]
    return (status, metadata, chart, comparison, *per_method, str(report))


def clear():
    return (None, 60, *_empty(IDLE_STATUS))


with gr.Blocks(title="SceneSift: Classical Cues versus Learned Features for Video Highlight Extraction",
               **(_STYLE if _STYLE_ON_BLOCKS else {})) as demo:
    gr.Markdown("# SceneSift: Classical Cues versus Learned Features for Training-Free "
                "Video Highlight Extraction")
    gr.Markdown("SceneSift Project Team\n\nCPS 843 / CP 8307: Computer Vision", elem_id="byline")
    gr.Markdown(ABSTRACT, elem_id="abstract")
    gr.Markdown(METHOD)
    _figure("pipeline", "**Figure 1.** The SceneSift pipeline.")
    gr.Markdown(METHOD_TABLE)
    gr.Markdown(EVALUATION)
    _figure("f1", "**Figure 2.** Temporal F1 against one annotator: the two summaries are aligned in time and "
            "the overlap is divided by each summary's length. Video: “Pure Fix TV: How to Wheelie” "
            "(TVSum, CC BY).", narrow=True)
    gr.Markdown(RESULTS)
    _figure("results", "**Figure 3.** Mean F1 and Kendall's τ per method with 95% bootstrap intervals over "
            "videos. The dashed line is the agreement between annotators.")
    gr.Markdown(RESULTS_DETAIL)
    _figure("ablation", "**Figure 4.** Change in the multimodal method's τ when one feature is removed.", narrow=True)
    _figure("categories", "**Figure 5.** Mean Kendall's τ on TVSum by category (five videos each).", narrow=True)
    gr.Markdown(DISCUSSION)
    _figure("example", "**Figure 6.** The TVSum video with the median difference between D and C. Top: frames "
            "from the four highest-scoring clips of D and of C with the annotators' mean rating. Middle: mean "
            "rating of every 2-second shot. Bottom: each method's summary. Video: “#453 girl gets van stuck "
            "in the back fourty [Davidsfarm]” (TVSum, CC BY).")
    gr.Markdown(DEMONSTRATION)
    video = gr.Video(label="Video", sources=["upload"], format=None, height=320)
    duration = gr.Slider(minimum=15, maximum=180, step=15, value=60, label="Summary length (seconds)")
    with gr.Row():
        analyse_button = gr.Button("Analyse video", variant="primary")
        clear_button = gr.Button("Clear", variant="secondary")
    status = gr.Markdown(IDLE_STATUS)
    details = gr.Dataframe(headers=["Property", "Value"], datatype=["str", "str"], interactive=False,
                           label="Video")
    gr.Markdown("### 5.1 Selections by method")
    comparison_chart = gr.Image(label="Clips selected by each method", interactive=False, type="filepath")
    comparison_table = gr.Dataframe(headers=COMPARE_HEADERS, interactive=False, wrap=True, label="Summary")
    gr.Markdown("### 5.2 Highlight reels")
    method_outputs = []
    with gr.Tabs():
        for name, _, _ in METHODS:
            with gr.Tab(name):
                reel = gr.Video(label=f"Highlight reel, {name}", interactive=False, height=320)
                clips = gr.Dataframe(headers=CLIP_HEADERS, interactive=False, wrap=True,
                                     label="Selected clips (source timestamps)")
                method_outputs += [reel, clips]
    report = gr.File(label="Timestamps, scores, and settings for all methods (JSON)", interactive=False)
    gr.Markdown(REFERENCES)

    results = [status, details, comparison_chart, comparison_table, *method_outputs, report]
    analyse_button.click(analyse, inputs=[video, duration], outputs=results)
    clear_button.click(clear, outputs=[video, duration, *results])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SceneSift report page.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Address to listen on; 0.0.0.0 makes it reachable from other machines.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Also create a temporary public gradio.live link. Uploaded videos pass through it.")
    args = parser.parse_args()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                **({} if _STYLE_ON_BLOCKS else _STYLE))


if __name__ == "__main__":
    main()
