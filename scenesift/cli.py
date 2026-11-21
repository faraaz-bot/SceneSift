"""Command-line entry points: `python -m scenesift summarize|evaluate|fit`."""
from __future__ import annotations

import argparse
import sys

from .config import METHOD_FEATURES, Config


def _config(path: str | None) -> Config:
    return Config.from_file(path) if path else Config()


def _summarize(args: argparse.Namespace) -> int:
    from .pipeline import summarize

    def progress(fraction: float, message: str) -> None:
        print(f"\r{fraction:5.0%}  {message:<50}", end="", file=sys.stderr, flush=True)

    summary = summarize(args.video, prompt=args.prompt, target_seconds=args.duration, method=args.method,
                        cfg=_config(args.config), out_dir=args.out, render=not args.no_render,
                        progress=progress)
    print(file=sys.stderr)
    print(summary.describe())
    for clip in summary.clips:
        print(f"  {clip['start']:8.2f} – {clip['end']:8.2f}  score {clip['score']:.3f}  {clip['rationale']}")
    for note in summary.notes:
        print(f"note: {note}")
    if summary.output_path:
        print(f"video: {summary.output_path}")
    print(f"details: {summary.json_path}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation import run_comparison

    run_comparison(args.manifest, args.out, cfg=_config(args.config), methods=tuple(args.methods),
                   budgets=tuple(args.budgets), split=args.split, ablations=args.ablations)
    return 0


def _fit(args: argparse.Namespace) -> int:
    from .fitting import run_fit

    run_fit(args.manifest, args.out, cfg=_config(args.config), split=args.split, budget=args.budget,
            method=args.method, how=args.how, report_file=args.report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scenesift", description="Multimodal video highlight extraction.")
    commands = parser.add_subparsers(dest="command", required=True)

    summarize = commands.add_parser("summarize", help="Create a highlight reel from one video.")
    summarize.add_argument("video")
    summarize.add_argument("--prompt", help="What to look for, in plain language.")
    summarize.add_argument("--duration", type=float, default=60.0, help="Target length in seconds.")
    summarize.add_argument("--method", choices=sorted(METHOD_FEATURES), default="multimodal")
    summarize.add_argument("--out", default="output", help="Directory for highlights.mp4 and highlights.json.")
    summarize.add_argument("--config", help="YAML file overriding defaults.")
    summarize.add_argument("--no-render", action="store_true", help="Write timestamps only.")
    summarize.set_defaults(run=_summarize)

    evaluate = commands.add_parser("evaluate", help="Compare methods on annotated videos.")
    evaluate.add_argument("manifest", help="CSV with source_id, category, path, annotation, split.")
    evaluate.add_argument("--out", default="experiments/results")
    evaluate.add_argument("--methods", nargs="+", choices=sorted(METHOD_FEATURES),
                          default=["activity", "semantic", "multimodal", "classical", "blend"])
    evaluate.add_argument("--budgets", nargs="+", default=["30", "60", "120"],
                          help="Seconds (30) or a share of each video's length (15%%).")
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--ablations", action="store_true",
                          help="Also run the multimodal method with each modality removed and without diversity.")
    evaluate.add_argument("--config", help="YAML file overriding defaults.")
    evaluate.set_defaults(run=_evaluate)

    fit = commands.add_parser("fit", help="Fit a method's weights on annotated development videos.")
    fit.add_argument("manifest", help="CSV with source_id, category, path, annotation, split.")
    fit.add_argument("--out", default="configs/fitted.yaml", help="YAML file to write the weights to.")
    fit.add_argument("--split", default="train", help="Manifest split to fit on.")
    fit.add_argument("--method", choices=sorted(METHOD_FEATURES), default="blend")
    fit.add_argument("--how", choices=["regression", "search"], default="regression",
                     help="Least-squares fit to annotator selections, or a grid search for the best F1.")
    fit.add_argument("--budget", default="15%", help="Summary budget for the leave-one-out report.")
    fit.add_argument("--report", help="CSV file for the per-video leave-one-out scores.")
    fit.add_argument("--config", help="YAML file overriding defaults.")
    fit.set_defaults(run=_fit)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
