"""Fit a method's fusion weights to human annotations on development videos.

Two ways of fitting are offered. `regression` (the default) sets each candidate segment's target to
the share of it that annotators selected and solves non-negative least squares on features and
targets centred within each video. `search` tries every weight combination on a coarse grid and
keeps the one with the best mean F1.

Both report leave-one-video-out scores: each video is scored with weights fitted on the others.
"""
from __future__ import annotations

import itertools
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from .config import METHOD_FEATURES, Config
from .evaluation import (Annotation, evaluate_intervals, load_annotation, overlap_length, parse_budget,
                         rank_correlation, read_manifest, reference_scores)
from .pipeline import Scored, analyze, importance_curve, score, summarize
from .scoring import fuse


def design(scored: Scored, annotation: Annotation, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Centred feature matrix and target for one video, scaled so every video counts equally."""
    columns = []
    for name in names:
        values = scored.features.get(name)
        # Missing evidence is neutral (0.5), matching how fusion treats it.
        columns.append(np.zeros(len(scored.segments)) if values is None
                       else np.nan_to_num(values, nan=0.5) - 0.5)
    target = np.array([
        statistics.fmean(overlap_length([(segment.start, segment.end)], intervals) / segment.length
                         for intervals in annotation.annotators.values())
        for segment in scored.segments])
    scale = 1.0 / np.sqrt(max(len(scored.segments), 1))
    return np.column_stack(columns) * scale, (target - target.mean()) * scale


def fit_weights(designs: list[tuple[np.ndarray, np.ndarray]], names: tuple[str, ...]) -> dict[str, float]:
    """Non-negative weights summing to one. Features that do not help get zero."""
    matrix = np.vstack([x for x, _ in designs])
    target = np.concatenate([y for _, y in designs])
    solution, _ = nnls(matrix, target)
    if solution.sum() <= 0:
        raise ValueError("No feature is positively related to the annotations; weights cannot be fitted.")
    return {name: round(float(value / solution.sum()), 4) for name, value in zip(names, solution)}


def weight_grid(names: tuple[str, ...], steps: int) -> list[dict[str, float]]:
    """Every way of sharing `steps` equal parts of the total weight among the features."""
    grid = []
    for parts in itertools.product(range(steps + 1), repeat=len(names) - 1):
        if sum(parts) <= steps:
            shares = (*parts, steps - sum(parts))
            grid.append({name: round(share / steps, 4) for name, share in zip(names, shares)})
    return grid


def _evaluate(video: dict, method: str, weights: dict[str, float], cfg: Config, budget: float | str) -> dict:
    """F1 and rank correlation of one method on one video with the given weights."""
    scored = video["scored"]
    use = [name for name in METHOD_FEATURES[method] if weights.get(name, 0.0) > 0.0]
    fused, contributions, applied = fuse(scored.features, weights, use)
    scored = replace(scored, score=fused, contributions=contributions, weights=applied)
    analysis = video["analysis"]
    _, seconds = parse_budget(budget, analysis.duration)
    summary = summarize(analysis.path, target_seconds=seconds, method=method,
                        cfg=replace(cfg, method_weights={**cfg.method_weights, method: weights}),
                        analysis=analysis, scored=scored)
    step, references = video["references"]
    bins = max(len(values) for values in references.values())
    return {**evaluate_intervals(summary.intervals(), video["annotation"]),
            **rank_correlation(importance_curve(scored, analysis.duration, step, bins), references)}


def run_fit(manifest: str | Path, out_file: str | Path, *, cfg: Config | None = None, split: str = "train",
            budget: float | str = "15%", method: str = "blend", how: str = "regression",
            grid_steps: int = 10, report_file: str | Path | None = None, log=print) -> dict:
    """Fit `method`'s weights on every video of `split`, write them, and report held-out scores.

    The weights are merged into `out_file` under `method_weights`, so one file can hold several
    methods. The leave-one-video-out scores use weights fitted without the video being scored;
    `report_file` saves them per video as CSV.
    """
    import yaml

    cfg = cfg or Config()
    if method not in METHOD_FEATURES:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHOD_FEATURES)}.")
    if how not in ("regression", "search"):
        raise ValueError("how must be 'regression' or 'search'.")
    names = METHOD_FEATURES[method]
    rows = [row for row in read_manifest(manifest) if row["split"] == split]
    if len(rows) < 2:
        raise ValueError(f"Fitting needs at least two videos with split={split!r} in {manifest}.")
    videos = []
    for row in rows:
        annotation = load_annotation(row["annotation"])
        log(f"[{row['source_id']}] analysing {row['path']}")
        analysis = analyze(row["path"], cfg)
        scored = score(analysis, cfg, None, method)
        videos.append({"id": row["source_id"], "annotation": annotation, "analysis": analysis,
                       "scored": scored, "design": design(scored, annotation, names),
                       "references": reference_scores(annotation, analysis.duration)})
    others = [[v for j, v in enumerate(videos) if j != index] for index in range(len(videos))]

    if how == "regression":
        def choose(subset: list[dict]) -> dict[str, float]:
            return fit_weights([v["design"] for v in subset], names)
    else:
        grid = weight_grid(names, grid_steps)
        log(f"Scoring {len(grid)} weight combinations on {len(videos)} videos")
        def f1_or_nan(video: dict, weights: dict[str, float]) -> float:
            try:
                return _evaluate(video, method, weights, cfg, budget)["f1"]
            except ValueError:  # all the weight sits on evidence this video lacks
                return float("nan")

        table = np.array([[f1_or_nan(video, weights) for video in videos] for weights in grid])
        position = {video["id"]: column for column, video in enumerate(videos)}

        def choose(subset: list[dict]) -> dict[str, float]:
            # A combination must work on every video it is chosen from; NaN means rule it out.
            means = table[:, [position[v["id"]] for v in subset]].mean(axis=1)
            return grid[int(np.nanargmax(means))]

    held_out = []
    for video, rest in zip(videos, others):
        try:
            result = _evaluate(video, method, choose(rest), cfg, budget)
        except ValueError:  # the chosen weights need evidence this video lacks; use the defaults
            result = _evaluate(video, method, cfg.weights, cfg, budget)
        held_out.append({"source_id": video["id"], **result})
    weights = choose(videos)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    existing = (yaml.safe_load(out_file.read_text(encoding="utf-8")) or {}) if out_file.is_file() else {}
    merged = {**existing.get("method_weights", {}), method: weights}
    out_file.write_text(
        "# Per-method fusion weights fitted on development videos.\n"
        "# Written by `python -m scenesift fit`; do not edit by hand.\n"
        + yaml.safe_dump({**existing, "method_weights": merged}, sort_keys=False), encoding="utf-8")
    taus = [r["kendall_tau"] for r in held_out if r["kendall_tau"] is not None]
    report = {"method": method, "how": how, "weights": weights, "videos": len(videos), "held_out": held_out,
              "held_out_f1": statistics.fmean(r["f1"] for r in held_out),
              "held_out_kendall_tau": statistics.fmean(taus) if taus else None}
    if report_file is not None:
        import csv

        report_file = Path(report_file)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(held_out[0]))
            writer.writeheader()
            writer.writerows(held_out)
    log(f"{method} weights ({how}): {weights}")
    log(f"Leave-one-video-out {method}: F1 {report['held_out_f1']:.3f}"
        + (f", Kendall tau {report['held_out_kendall_tau']:.3f}" if taus else ""))
    log(f"Wrote {out_file}")
    return report
