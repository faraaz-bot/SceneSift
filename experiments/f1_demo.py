"""Draw a labelled picture and video showing how temporal F1 is computed for one video.

The picture lines up one annotator's summary, a method's summary, and their overlap on a timeline,
works through precision, recall, and F1, and then shows the same comparison for every annotator.
The video plays the source with a label saying, moment by moment, who picked what.

    python experiments/f1_demo.py experiments/tvsum/manifest.csv iVt07TCkFM0 --method classical

Outputs go to `output/f1_demo/` (git-ignored). They contain frames of the source video, so the
source's licence applies to them.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scenesift.config import METHOD_FEATURES, Config  # noqa: E402
from scenesift.evaluation import (load_annotation, merge, overlap_length, parse_budget, read_manifest,  # noqa: E402
                                  temporal_prf, total_length)
from scenesift.ffmpeg import ffmpeg_exe  # noqa: E402
from scenesift.pipeline import summarize  # noqa: E402

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
# Roles: who picked a moment. Light values for the picture, dark values for the video panel.
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "rule": "#dddcd6", "track": "#eeede9",
         "annotator": "#eb6834", "method": "#2a78d6", "both": "#1baf7a"}
DARK = {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "rule": "#3a3a37", "track": "#2a2a28",
        "annotator": "#d95926", "method": "#3987e5", "both": "#199e70"}
STATE_TEXT = {
    "both": "BOTH picked this moment: it counts toward F1",
    "method": "ONLY THE METHOD picked this: it lowers precision",
    "annotator": "ONLY THE ANNOTATOR picked this: it lowers recall",
    "neither": "NEITHER picked this: no effect on F1",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size)


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def intersection(a: list, b: list) -> list[tuple[float, float]]:
    out = []
    for s1, e1 in merge(a):
        for s2, e2 in merge(b):
            if min(e1, e2) > max(s1, s2):
                out.append((max(s1, s2), min(e1, e2)))
    return out


def state_at(moment: float, method: list, annotator: list) -> str:
    in_method = any(s <= moment < e for s, e in method)
    in_annotator = any(s <= moment < e for s, e in annotator)
    return "both" if in_method and in_annotator else "method" if in_method else \
        "annotator" if in_annotator else "neither"


def strip(draw: ImageDraw.ImageDraw, intervals: list, colour: str, left: int, right: int, top: int,
          height: int, duration: float, track: str | None) -> None:
    """One timeline row: an optional pale track with rounded bars for the picked intervals."""
    if track:
        draw.rounded_rectangle([left, top, right, top + height], radius=4, fill=track)
    for start, end in merge(intervals):
        x0 = left + (right - left) * start / duration
        x1 = left + (right - left) * end / duration
        draw.rounded_rectangle([x0 + 1, top, max(x1 - 1, x0 + 3), top + height], radius=min(4, height // 2),
                               fill=colour)


def frame_times(method: list, annotator: list, duration: float, count: int) -> list[float]:
    """Moments to show as frames: one from the longest stretch of each kind, the rest evenly spaced."""
    both = intersection(method, annotator)
    edges = sorted({0.0, duration, *(t for pair in (*merge(method), *merge(annotator)) for t in pair)})
    longest: dict[str, tuple[float, float]] = {}
    for start, end in zip(edges, edges[1:]):
        state = state_at((start + end) / 2.0, method, annotator)
        if state != "neither" and end - start > longest.get(state, (0.0, 0.0))[0]:
            longest[state] = (end - start, (start + end) / 2.0)
    times = [middle for _, middle in longest.values()]
    for index in range(count):
        moment = (index + 0.5) * duration / count
        if len(times) < count and all(abs(moment - other) > duration / (2 * count) for other in times):
            times.append(moment)
    return sorted(times)


def grab_frame(path: str, moment: float) -> np.ndarray | None:
    capture = cv2.VideoCapture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def draw_picture(out: Path, video: str, title: str, credit: str, method_name: str, method: list,
                 annotators: dict[str, list], chosen: str, duration: float) -> None:
    c = LIGHT
    width, left, right = 1600, 330, 1540
    rows = len(annotators)
    height = 1010 + rows * 16
    image = Image.new("RGB", (width, height), c["surface"])
    draw = ImageDraw.Draw(image)
    annotator = annotators[chosen]
    both = intersection(method, annotator)
    kept, picked, shared = total_length(annotator), total_length(method), total_length(both)
    precision, recall, f1 = temporal_prf(method, annotator)

    draw.text((60, 40), "How the F1 score is worked out", font=font(38, True), fill=c["text"])
    draw.text((60, 92), f"Video: {title}  ({clock(duration)})   Method: {method_name}", font=font(20),
              fill=c["muted"])

    # Frames across the video, each tagged with who picked that moment.
    count, gap = 8, 12
    thumb_w = (width - 120 - gap * (count - 1)) // count
    thumb_h = thumb_w * 9 // 16
    for index, moment in enumerate(frame_times(method, annotator, duration, count)):
        frame = grab_frame(video, moment)
        x = 60 + index * (thumb_w + gap)
        if frame is not None:
            image.paste(Image.fromarray(frame).resize((thumb_w, thumb_h)), (x, 140))
        state = state_at(moment, method, annotator)
        if state != "neither":
            draw.rectangle([x, 140 + thumb_h, x + thumb_w, 140 + thumb_h + 8], fill=c[state])
        label = {"both": "both", "method": "method only", "annotator": "annotator only",
                 "neither": "neither"}[state]
        draw.text((x, 140 + thumb_h + 14), f"{clock(moment)}  {label}", font=font(15), fill=c["muted"])

    # The three rows the score is built from, on one time axis.
    top = 140 + thumb_h + 70
    draw.text((60, top), "1. Line the two summaries up in time", font=font(24, True), fill=c["text"])
    top += 50
    for label, seconds, intervals, role in (
            (f"Annotator kept ({chosen.replace('_', ' ')})", kept, annotator, "annotator"),
            (f"Method picked ({method_name})", picked, method, "method"),
            ("Both: the overlap", shared, both, "both")):
        draw.rectangle([60, top + 6, 76, top + 22], fill=c[role])
        draw.text((86, top - 2), label, font=font(17), fill=c["text"])
        draw.text((86, top + 20), f"{seconds:.1f} seconds", font=font(15), fill=c["muted"])
        strip(draw, intervals, c[role], left, right, top, 30, duration, c["track"])
        top += 58
    tick = 15 if duration <= 240 else 30
    for moment in np.arange(0, duration + 1e-6, tick):
        x = left + (right - left) * moment / duration
        draw.line([x, top - 6, x, top], fill=c["muted"], width=1)
        draw.text((x - 14, top + 4), clock(moment), font=font(14), fill=c["muted"])

    # The arithmetic.
    top += 56
    draw.text((60, top), "2. Turn the overlap into a score", font=font(24, True), fill=c["text"])
    lines = [
        ("Precision", f"overlap ÷ what the method picked  =  {shared:.1f} ÷ {picked:.1f}  =  {precision:.2f}",
         "how much of the method's summary the annotator also wanted"),
        ("Recall", f"overlap ÷ what the annotator kept  =  {shared:.1f} ÷ {kept:.1f}  =  {recall:.2f}",
         "how much of the annotator's summary the method found"),
        ("F1", f"2 × precision × recall ÷ (precision + recall)  =  {f1:.2f}",
         "one number that is high only when both are high; 1.00 is a perfect match, 0 is no overlap"),
    ]
    top += 46
    for name, formula, meaning in lines:
        draw.text((60, top), name, font=font(20, True), fill=c["text"])
        draw.text((200, top), formula, font=font(20), fill=c["text"])
        draw.text((200, top + 28), meaning, font=font(15), fill=c["muted"])
        top += 66

    # Every annotator, because people disagree and the reported score is the average.
    top += 10
    draw.text((60, top), f"3. Repeat for all {rows} annotators and average", font=font(24, True),
              fill=c["text"])
    draw.text((60, top + 34), "Each row is one person's summary. Blue bands mark what the method picked. "
              "The F1 against each person is on the right.", font=font(15), fill=c["muted"])
    top += 70
    for start, end in merge(method):
        x0 = left + (right - left) * start / duration
        x1 = left + (right - left) * end / duration
        draw.rectangle([x0, top - 4, x1, top + rows * 16], fill="#d6e6f8")
    scores = []
    draw.text((right + 8, top - 22), "F1", font=font(13, True), fill=c["text"])
    for name, intervals in annotators.items():
        score = temporal_prf(method, intervals)[2]
        scores.append(score)
        strip(draw, intervals, c["annotator"], left, right, top, 10, duration, None)
        marker = "  ◄ shown above" if name == chosen else ""
        draw.text((60, top - 3), name.replace("_", " ") + marker, font=font(13, name == chosen),
                  fill=c["text"] if name == chosen else c["muted"])
        draw.text((right + 8, top - 3), f"{score:.2f}", font=font(13, name == chosen), fill=c["text"])
        top += 16
    top += 14
    draw.text((60, top), f"Mean F1 over the {rows} annotators: {statistics.fmean(scores):.2f}.  "
              "This mean is the video's score; a method's result is the mean over all videos.",
              font=font(18, True), fill=c["text"])
    draw.text((60, height - 34), credit, font=font(13), fill=c["muted"])
    image.save(out)


def render_video(out: Path, video: str, method_name: str, method: list, annotator: list, chosen: str,
                 duration: float, has_audio: bool) -> None:
    c = DARK
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, video_h, panel_h = 960, 540, 190
    left, right = 250, 930
    with tempfile.TemporaryDirectory(prefix="scenesift-f1demo-") as scratch:
        silent = Path(scratch) / "silent.mp4"
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, video_h + panel_h))
        both = intersection(method, annotator)
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                moment = index / fps
                index += 1
                canvas = Image.new("RGB", (width, video_h + panel_h), c["surface"])
                scale = min(width / frame.shape[1], video_h / frame.shape[0])
                size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
                picture = Image.fromarray(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB))
                canvas.paste(picture, ((width - size[0]) // 2, (video_h - size[1]) // 2))
                draw = ImageDraw.Draw(canvas)

                state = state_at(moment, method, annotator)
                draw.rectangle([0, 0, width, 54], fill="#000000")
                if state != "neither":
                    draw.rectangle([18, 15, 42, 39], fill=c[state])
                draw.text((56, 14), STATE_TEXT[state], font=font(22, True), fill=c["text"])

                y = video_h + 18
                for label, intervals, role in ((f"Annotator ({chosen.replace('_', ' ')})", annotator, "annotator"),
                                               (f"Method ({method_name})", method, "method"),
                                               ("Both: overlap", both, "both")):
                    draw.rectangle([18, y + 3, 30, y + 15], fill=c[role])
                    draw.text((38, y), label, font=font(15), fill=c["text"])
                    strip(draw, intervals, c[role], left, right, y, 18, duration, c["track"])
                    y += 30
                x = left + (right - left) * min(moment, duration) / duration
                draw.line([x, video_h + 12, x, y - 6], fill=c["text"], width=2)

                def so_far(intervals: list) -> float:
                    return overlap_length(intervals, [(0.0, moment)])

                shared, picked, kept = so_far(both), so_far(method), so_far(annotator)
                precision = shared / picked if picked else 0.0
                recall = shared / kept if kept else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                draw.text((18, y + 8), f"{clock(moment)} / {clock(duration)}", font=font(15), fill=c["muted"])
                draw.text((left, y + 8),
                          f"So far:  overlap {shared:.0f} s   precision {precision:.2f}   "
                          f"recall {recall:.2f}   F1 {f1:.2f}", font=font(17, True), fill=c["text"])
                draw.text((left, y + 34), "precision = overlap ÷ method's seconds     "
                          "recall = overlap ÷ annotator's seconds", font=font(13), fill=c["muted"])
                writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        finally:
            capture.release()
            writer.release()
        command = [ffmpeg_exe(), "-y", "-v", "error", "-i", str(silent)]
        if has_audio:
            command += ["-i", video, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(out)]
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_id")
    parser.add_argument("--method", choices=sorted(METHOD_FEATURES), default="classical")
    parser.add_argument("--annotator", help="Annotator to show in detail; default is the one with the median F1.")
    parser.add_argument("--budget", default="15%", help="Summary length: seconds or a share such as 15%%.")
    parser.add_argument("--config", help="YAML file overriding defaults.")
    parser.add_argument("--title", help="Title to print; defaults to the source id.")
    parser.add_argument("--credit", default="", help="Source and licence line printed on the picture.")
    parser.add_argument("--out", type=Path, default=Path("output/f1_demo"))
    parser.add_argument("--no-video", action="store_true", help="Only draw the picture.")
    args = parser.parse_args()
    try:
        row = next((r for r in read_manifest(args.manifest) if r["source_id"] == args.source_id), None)
        if row is None:
            raise ValueError(f"{args.source_id!r} is not in {args.manifest}")
        cfg = Config.from_file(args.config) if args.config else Config()
        annotation = load_annotation(row["annotation"])
        from scenesift.pipeline import analyze

        analysis = analyze(row["path"], cfg)
        _, seconds = parse_budget(args.budget, analysis.duration)
        summary = summarize(row["path"], target_seconds=seconds, method=args.method, cfg=cfg, analysis=analysis)
        method = summary.intervals()
        chosen = args.annotator
        if chosen is None:
            ranked = sorted(annotation.annotators, key=lambda n: temporal_prf(method, annotation.annotators[n])[2])
            chosen = ranked[len(ranked) // 2]
        elif chosen not in annotation.annotators:
            raise ValueError(f"No annotator {chosen!r}; choose from {sorted(annotation.annotators)}")
        args.out.mkdir(parents=True, exist_ok=True)
        picture = args.out / f"{args.source_id}_{args.method}_f1.png"
        draw_picture(picture, row["path"], args.title or args.source_id, args.credit, args.method, method,
                     annotation.annotators, chosen, analysis.duration)
        print(f"picture: {picture}")
        if not args.no_video:
            clip = args.out / f"{args.source_id}_{args.method}_f1.mp4"
            render_video(clip, row["path"], args.method, method, annotation.annotators[chosen], chosen,
                         analysis.duration, bool(analysis.info["has_audio"]))
            print(f"video: {clip}")
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
