"""Draw the report's figures from the evaluation outputs.

    python research/make_figures.py --summe experiments/results/summe --tvsum experiments/results/tvsum \
        --tvsum-manifest experiments/tvsum/manifest.csv --config configs/fitted_summe.yaml

Writes PDF and PNG files to `research/figures/`. The charts read the `results.csv` files written by
`python -m scenesift evaluate`; the two figures that show a video need the TVSum videos.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scenesift.config import Config  # noqa: E402
from scenesift.evaluation import load_annotation, merge, parse_budget, read_manifest, temporal_prf  # noqa: E402
from scenesift.evaluation import total_length  # noqa: E402
from scenesift.pipeline import analyze, summarize  # noqa: E402

INK, MUTED, RULE, TRACK, SURFACE = "#0b0b0b", "#52514e", "#dddcd6", "#eeede9", "#ffffff"
# One fixed colour per method, used in every figure. The uniform reference is neutral.
METHODS = [("activity", "A: activity", "#2a78d6"), ("semantic", "B: semantic", "#eb6834"),
           ("multimodal", "C: multimodal", "#1baf7a"), ("classical", "D: classical", "#eda100"),
           ("blend", "Blend", "#e87ba4")]
COLOUR = {key: colour for key, _, colour in METHODS} | {"uniform": "#9a9993"}
LABEL = {key: label for key, label, _ in METHODS} | {"uniform": "Uniform"}
ANNOTATOR, OVERLAP = "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list("blue-gray-red", ["#c2362f", "#f0efec", "#1c5cab"])
FULL, COLUMN = 7.16, 3.5  # IEEE text and column widths in inches

plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": RULE,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.0, "pdf.fonttype": 42})


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def load(results_dir: Path) -> list[dict]:
    with open(results_dir / "results.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict], method: str, metric: str, variant: str = "full") -> dict[str, float]:
    wanted = "reference" if method == "uniform" else variant
    return {row["source_id"]: float(row[metric]) for row in rows
            if row["method"] == method and row["variant"] == wanted and row[metric] not in ("", "None")}


def mean_interval(numbers: list[float]) -> tuple[float, float, float]:
    """Mean with a 95% bootstrap interval over videos."""
    data = np.asarray(numbers)
    rng = np.random.default_rng(0)
    means = rng.choice(data, size=(10000, len(data)), replace=True).mean(axis=1)
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def save(figure: plt.Figure, out_dir: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(out_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out_dir / name}.pdf")


def figure_pipeline(out_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(FULL, 2.55))
    axis.set_xlim(0, 110)
    axis.set_ylim(0, 40)
    axis.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str = "", fill: str = "#f6f5f2") -> None:
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                      facecolor=fill, edgecolor=RULE, linewidth=0.8))
        axis.text(x + w / 2, y + h - 2.4 if body else y + h / 2, title, ha="center",
                  va="center", fontsize=7, fontweight="bold", color=INK)
        if body:
            axis.text(x + w / 2, y + (h - 4.4) / 2, body, ha="center", va="center", fontsize=5.8, color=MUTED,
                      linespacing=1.35)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                       color=MUTED, linewidth=0.8))

    box(1, 14, 12, 12, "Input video", "frames\nand audio")
    box(17, 14, 15, 12, "Candidate clips", "shots (2–12 s)\n+ 4 s and 8 s\nwindows")
    box(36, 27, 29, 12, "Classical cues", "motion (optical flow) · colour\nhistograms · edges · frame change",
        fill="#fdf3d9")
    box(36, 14, 29, 12, "Signal cue", "audio loudness and\nspectral change", fill="#e3eefb")
    box(36, 1, 29, 12, "Learned features", "CLIP visual · Whisper transcript +\nsentence embeddings · novelty",
        fill="#dff3ea")
    box(69, 14, 12, 12, "Normalize", "per video,\nmedian/MAD\nto (0, 1)")
    box(85, 14, 10, 12, "Fuse", "weighted\nsum per\nmethod")
    box(99, 14, 10, 12, "Select", "budget,\noverlap,\ndiversity")
    arrow(13.4, 20, 16.6, 20)
    for y in (33, 20, 7):
        arrow(32.4, 20, 35.6, y)
        arrow(65.4, y, 68.6, 20)
    arrow(81.4, 20, 84.6, 20)
    arrow(95.4, 20, 98.6, 20)
    axis.text(104, 11, "highlight reel\n+ timestamps", ha="center", va="top", fontsize=6, color=MUTED)
    save(figure, out_dir, "pipeline")


def strip(axis: plt.Axes, intervals: list, row: float, colour: str, duration: float, height: float = 0.62) -> None:
    axis.barh(row, duration, left=0, height=height, color=TRACK, linewidth=0)
    for start, end in merge(intervals):
        axis.barh(row, end - start, left=start, height=height, color=colour, linewidth=0)


def figure_f1(out_dir: Path, method: list, annotator: list, duration: float, name: str) -> None:
    both = [(max(a, c), min(b, d)) for a, b in merge(method) for c, d in merge(annotator) if min(b, d) > max(a, c)]
    kept, picked, shared = total_length(annotator), total_length(method), total_length(both)
    precision, recall, f1 = temporal_prf(method, annotator)
    figure, axis = plt.subplots(figsize=(COLUMN, 1.75))
    for row, (label, intervals, colour, seconds) in enumerate((
            ("Annotator kept", annotator, ANNOTATOR, kept), ("Method picked", method, COLOUR["classical"], picked),
            ("Overlap", both, OVERLAP, shared))):
        strip(axis, intervals, -row, colour, duration)
        axis.text(-0.02 * duration, -row, f"{label}\n{seconds:.1f} s", ha="right", va="center", fontsize=7,
                  color=INK, linespacing=1.2)
    axis.set_yticks([])
    axis.set_ylim(-2.6, 0.6)
    axis.set_xlim(0, duration)
    ticks = np.arange(0, duration + 1, 30)
    axis.set_xticks(ticks, [clock(t) for t in ticks])
    axis.spines["left"].set_visible(False)
    axis.set_xlabel(f"Time in the video ({name})")
    text = (f"precision = {shared:.1f} / {picked:.1f} = {precision:.2f}      "
            f"recall = {shared:.1f} / {kept:.1f} = {recall:.2f}      F1 = {f1:.2f}")
    figure.text(0.5, -0.2, text, ha="center", fontsize=7, color=INK)
    save(figure, out_dir, "f1")


def figure_results(out_dir: Path, summe: list[dict], tvsum: list[dict], held_out: Path | None) -> None:
    blend_summe = {}
    if held_out and held_out.is_file():
        with open(held_out, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                blend_summe[row["source_id"]] = row
    panels = [("SumMe: F1", summe, "f1", "annotator_agreement_f1"),
              ("SumMe: Kendall's τ", summe, "kendall_tau", "annotator_agreement_tau"),
              ("TVSum: F1", tvsum, "f1", "annotator_agreement_f1"),
              ("TVSum: Kendall's τ", tvsum, "kendall_tau", "annotator_agreement_tau")]
    order = ["uniform", "activity", "semantic", "multimodal", "classical", "blend"]
    figure, axes = plt.subplots(1, 4, figsize=(FULL, 2.05), sharey=True)
    for axis, (title, rows, metric, human_key) in zip(axes, panels):
        human = statistics.fmean(float(r[human_key]) for r in rows if r["method"] == "uniform")
        for position, method in enumerate(order):
            data = values(rows, method, metric)
            if method == "blend" and rows is summe and blend_summe:  # the blend was fitted on SumMe
                data = {key: float(row[metric]) for key, row in blend_summe.items()}
            if not data:
                continue
            mean, low, high = mean_interval(list(data.values()))
            axis.barh(position, mean, height=0.62, color=COLOUR[method], linewidth=0)
            axis.plot([low, high], [position, position], color=INK, linewidth=0.8)
            axis.text(max(high, 0) + 0.012, position, f"{mean:.3f}", va="center", fontsize=6.3, color=INK,
                      bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})
        axis.axvline(human, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
        axis.text(human, -0.72, f"human {human:.2f}", ha="center", va="bottom", fontsize=6, color=MUTED)
        axis.axvline(0, color=RULE, linewidth=0.6)
        axis.set_title(title, fontsize=7.5, color=INK, loc="left", pad=11)
        axis.set_xlim(-0.1 if "τ" in title else 0, max(human * 1.12, 0.26) + 0.07)
        axis.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(order)), [LABEL[m] for m in order])
    axes[0].invert_yaxis()
    save(figure, out_dir, "results")


def figure_categories(out_dir: Path, tvsum: list[dict]) -> None:
    by_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in tvsum:
        if row["variant"] == "full" and row["kendall_tau"] not in ("", "None"):
            by_category[row["category"]][row["method"]].append(float(row["kendall_tau"]))
    categories = sorted(by_category)
    keys = [key for key, _, _ in METHODS]
    grid = np.array([[statistics.fmean(by_category[c][m]) for m in keys] for c in categories])
    figure, axis = plt.subplots(figsize=(COLUMN, 2.5))
    axis.imshow(grid, cmap=DIVERGING, vmin=-0.2, vmax=0.2, aspect="auto")
    for (i, j), value in np.ndenumerate(grid):
        axis.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6.5,
                  color="#ffffff" if abs(value) > 0.13 else INK)
    axis.set_xticks(range(len(keys)), [LABEL[k].replace(": ", ":\n") for k in keys], fontsize=6.5)
    axis.set_yticks(range(len(categories)), [c.capitalize() for c in categories], fontsize=6.5)
    axis.tick_params(length=0)
    axis.xaxis.tick_top()
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks(np.arange(-0.5, len(keys)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(categories)), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=1.5)
    axis.tick_params(which="minor", length=0)
    save(figure, out_dir, "categories")


def figure_ablation(out_dir: Path, summe: list[dict], tvsum: list[dict]) -> None:
    features = ["visual", "motion", "audio", "speech", "novelty"]
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.75), sharey=True)
    for axis, (title, rows) in zip(axes, (("SumMe", summe), ("TVSum", tvsum))):
        full = values(rows, "multimodal", "kendall_tau")
        for position, feature in enumerate(features):
            without = values(rows, "multimodal", "kendall_tau", f"without-{feature}")
            change = [without[k] - full[k] for k in full if k in without]
            mean, low, high = mean_interval(change)
            axis.barh(position, mean, height=0.6, color="#1c5cab" if mean > 0 else "#c2362f", linewidth=0)
            axis.plot([low, high], [position, position], color=INK, linewidth=0.8)
            axis.text(high + 0.006 if mean > 0 else low - 0.006, position, f"{mean:+.3f}", va="center",
                      ha="left" if mean > 0 else "right", fontsize=6.3, color=INK)
        axis.axvline(0, color=MUTED, linewidth=0.6)
        axis.set_title(title, fontsize=7.5, color=INK, loc="left")
        axis.set_xlim(-0.1, 0.14)
        axis.set_xlabel("Change in τ")
        axis.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(features)), [f"− {f}" for f in features])
    axes[0].invert_yaxis()
    save(figure, out_dir, "ablation")


def grab(path: str, moment: float) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Could not read a frame at {moment:.1f}s from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def figure_example(out_dir: Path, row: dict, cfg: Config, title: str) -> None:
    annotation = load_annotation(row["annotation"])
    analysis = analyze(row["path"], cfg)
    duration = analysis.duration
    _, budget = parse_budget("15%", duration)
    summaries = {key: summarize(row["path"], target_seconds=budget, method=key, cfg=cfg, analysis=analysis)
                 for key, _, _ in METHODS}
    importance = np.mean([scores for scores in annotation.importance.values()], axis=0)
    step = annotation.importance_step

    figure = plt.figure(figsize=(FULL, 3.4))
    grid = figure.add_gridspec(3, 8, height_ratios=[1.15, 1.0, 1.25], hspace=0.42, wspace=0.06)
    for offset, key in ((0, "classical"), (4, "multimodal")):
        clips = sorted(summaries[key].clips, key=lambda clip: -clip["score"])[:4]
        for index, clip in enumerate(sorted(clips, key=lambda clip: clip["start"])):
            axis = figure.add_subplot(grid[0, offset + index])
            moment = (clip["start"] + clip["end"]) / 2.0
            axis.imshow(grab(row["path"], moment))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(COLOUR[key])
                spine.set_linewidth(1.6)
            rating = importance[min(int(moment // step), len(importance) - 1)]
            axis.set_xlabel(f"{clock(moment)} (rated {rating:.1f})", fontsize=5.8, color=MUTED, labelpad=2)
            if index == 0:
                axis.set_title(f"Picked by {LABEL[key]}", fontsize=7.5, color=INK, loc="left")

    curve = figure.add_subplot(grid[1, :])
    centres = (np.arange(len(importance)) + 0.5) * step
    curve.fill_between(centres, 1, importance, color="#f6d2c3", linewidth=0)
    curve.plot(centres, importance, color=ANNOTATOR, linewidth=1.2)
    curve.set_xlim(0, duration)
    curve.set_ylim(1, max(4.0, float(importance.max()) + 0.2))
    curve.set_xticks([])
    curve.set_ylabel("Mean rating\n(1 to 5)", color=MUTED)
    curve.tick_params(axis="y", colors=MUTED)
    curve.set_title(f"Annotators' importance over time: {title}", fontsize=7.5, color=INK, loc="left")

    strips = figure.add_subplot(grid[2, :])
    for position, (key, label, colour) in enumerate(METHODS):
        strip(strips, summaries[key].intervals(), position, colour, duration)
    strips.set_yticks(range(len(METHODS)), [label for _, label, _ in METHODS])
    strips.invert_yaxis()
    strips.set_xlim(0, duration)
    ticks = np.arange(0, duration + 1, 30)
    strips.set_xticks(ticks, [clock(t) for t in ticks])
    strips.spines["left"].set_visible(False)
    strips.set_xlabel("Time in the video; coloured bars are each method's 15% summary")
    save(figure, out_dir, "example")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summe", type=Path, required=True, help="Directory with SumMe results.csv.")
    parser.add_argument("--tvsum", type=Path, required=True, help="Directory with TVSum results.csv.")
    parser.add_argument("--tvsum-manifest", type=Path, help="TVSum manifest; needed for the video figures.")
    parser.add_argument("--config", help="YAML config used for the evaluation runs.")
    parser.add_argument("--f1-video", default="iVt07TCkFM0")
    parser.add_argument("--f1-annotator", default="user_02")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        summe, tvsum = load(args.summe), load(args.tvsum)
        figure_pipeline(args.out)
        figure_results(args.out, summe, tvsum, args.summe / "blend_held_out.csv")
        figure_categories(args.out, tvsum)
        figure_ablation(args.out, summe, tvsum)
        if args.tvsum_manifest:
            cfg = Config.from_file(args.config) if args.config else Config()
            rows = {row["source_id"]: row for row in read_manifest(args.tvsum_manifest)}
            row = rows[args.f1_video]
            analysis = analyze(row["path"], cfg)
            _, budget = parse_budget("15%", analysis.duration)
            picked = summarize(row["path"], target_seconds=budget, method="classical", cfg=cfg, analysis=analysis)
            figure_f1(args.out, picked.intervals(), load_annotation(row["annotation"]).annotators[args.f1_annotator],
                      analysis.duration, args.f1_video)
            # The worked example is the video with the median gap between D and C, not a hand-picked one.
            classical, multimodal = values(tvsum, "classical", "kendall_tau"), values(tvsum, "multimodal", "kendall_tau")
            ranked = sorted(classical, key=lambda k: classical[k] - multimodal[k])
            median = ranked[len(ranked) // 2]
            print(f"example video: {median} (D {classical[median]:+.3f}, C {multimodal[median]:+.3f})")
            figure_example(args.out, rows[median], cfg, median)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
