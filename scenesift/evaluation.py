"""Evaluation against human reference intervals, and the method-comparison runner."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from .config import FEATURES, Config
from .pipeline import analyze, summarize

Interval = tuple[float, float]

# Stages each method would need if it were run on its own, for the runtime comparison.
METHOD_STAGES = {
    "activity": ("decode", "motion", "shots", "audio"),
    "semantic": ("decode", "visual", "shots"),
    "multimodal": ("decode", "motion", "visual", "shots", "audio", "speech"),
}


def merge(intervals: list[Interval]) -> list[Interval]:
    merged: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals if e > s):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def total_length(intervals: list[Interval]) -> float:
    return sum(end - start for start, end in merge(intervals))


def overlap_length(a: list[Interval], b: list[Interval]) -> float:
    a, b = merge(a), merge(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def temporal_prf(predicted: list[Interval], reference: list[Interval]) -> tuple[float, float, float]:
    """Precision, recall, and F1 measured in seconds of overlap."""
    overlap = overlap_length(predicted, reference)
    predicted_length, reference_length = total_length(predicted), total_length(reference)
    precision = overlap / predicted_length if predicted_length > 0 else 0.0
    recall = overlap / reference_length if reference_length > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def event_coverage(predicted: list[Interval], events: list[Interval], min_fraction: float = 0.5) -> float:
    """Fraction of reference events with at least `min_fraction` of their duration in the summary."""
    events = [(s, e) for s, e in events if e > s]
    if not events:
        return 0.0
    covered = sum(1 for event in events
                  if overlap_length(predicted, [event]) >= min_fraction * (event[1] - event[0]) - 1e-9)
    return covered / len(events)


def uniform_intervals(duration: float, budget: float, clip_seconds: float = 5.0) -> list[Interval]:
    """Evenly spaced clips filling the budget: a content-blind reference for interpreting scores."""
    count = max(1, round(budget / clip_seconds))
    length = budget / count
    return [((index + 0.5) * duration / count - length / 2.0, (index + 0.5) * duration / count + length / 2.0)
            for index in range(count)]


@dataclass
class Annotation:
    video_id: str
    category: str
    annotators: dict[str, list[Interval]]
    prompt: str | None = None


def load_annotation(path: str | Path) -> Annotation:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    annotators = {name: [(float(s), float(e)) for s, e in intervals]
                  for name, intervals in data["annotators"].items()}
    if not annotators:
        raise ValueError(f"{path}: no annotators.")
    return Annotation(str(data["video_id"]), str(data.get("category", "unknown")), annotators,
                      data.get("prompt") or None)


def annotator_agreement(annotation: Annotation) -> float | None:
    """Mean pairwise temporal F1 between annotators; None with a single annotator."""
    pairs = list(combinations(annotation.annotators.values(), 2))
    if not pairs:
        return None
    return statistics.fmean(temporal_prf(a, b)[2] for a, b in pairs)


def evaluate_intervals(predicted: list[Interval], annotation: Annotation) -> dict[str, float]:
    """Score one summary against every annotator independently, then average."""
    rows = [(*temporal_prf(predicted, reference), event_coverage(predicted, reference))
            for reference in annotation.annotators.values()]
    precision, recall, f1, coverage = (statistics.fmean(column) for column in zip(*rows))
    return {"precision": precision, "recall": recall, "f1": f1, "f1_best_annotator": max(r[2] for r in rows),
            "event_coverage": coverage}


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_id", "category", "path", "annotation", "split"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"{path}: manifest needs columns {sorted(required)}.")
    for row in rows:
        for key in ("path", "annotation"):
            if row[key] and not Path(row[key]).is_absolute():
                row[key] = str((path.parent / row[key]).resolve())
    return rows


def parse_budget(value: float | str, duration: float) -> tuple[str, float]:
    """A budget is seconds (`30`) or a share of the source length (`"15%"`). Returns (label, seconds)."""
    if isinstance(value, str) and value.strip().endswith("%"):
        percent = float(value.strip()[:-1])
        if not 0 < percent < 100:
            raise ValueError(f"Budget percentage must be between 0 and 100: {value!r}")
        return f"{percent:g}%", duration * percent / 100.0
    seconds = float(value)
    if seconds <= 0:
        raise ValueError(f"Budget must be positive: {value!r}")
    return f"{seconds:g}s", seconds


def run_comparison(manifest: str | Path, out_dir: str | Path, *, cfg: Config | None = None,
                   methods: tuple[str, ...] = ("activity", "semantic", "multimodal"),
                   budgets: tuple[float | str, ...] = (30.0, 60.0, 120.0), split: str = "test",
                   ablations: bool = False, log=print) -> list[dict]:
    """Run every method on the same videos and budgets; write results.csv and summary.md."""
    cfg = cfg or Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_manifest(manifest) if row["split"] == split]
    if not rows:
        raise ValueError(f"No videos with split={split!r} in {manifest}.")
    results: list[dict] = []
    for row in rows:
        annotation = load_annotation(row["annotation"])
        prompt = (row.get("prompt") or "").strip() or annotation.prompt
        log(f"[{row['source_id']}] analysing {row['path']}")
        analysis = analyze(row["path"], cfg)
        minutes = analysis.duration / 60.0
        agreement = annotator_agreement(annotation)
        variants: list[tuple[str, str, tuple[str, ...], bool | None]] = [(m, "full", (), None) for m in methods]
        if ablations and "multimodal" in methods:
            variants += [("multimodal", f"without-{name}", (name,), None) for name in FEATURES]
            variants.append(("multimodal", "without-diversity", (), False))
        for requested in budgets:
            label, budget = parse_budget(requested, analysis.duration)
            if budget >= analysis.duration:
                log(f"[{row['source_id']}] skipping {label} budget: source is {analysis.duration:.0f}s")
                continue
            results.append({
                "source_id": row["source_id"], "category": row["category"], "method": "uniform",
                "variant": "reference", "budget": label, "budget_seconds": round(budget, 3),
                "summary_seconds": round(budget, 3),
                **evaluate_intervals(uniform_intervals(analysis.duration, budget), annotation),
                "redundancy": None, "seconds_per_source_minute": 0.0,
                "annotators": len(annotation.annotators), "annotator_agreement_f1": agreement,
                "prompt": prompt or "", "evidence_used": "none",
            })
            for method, variant, drop, diversity in variants:
                try:
                    summary = summarize(row["path"], prompt=prompt, target_seconds=budget, method=method,
                                        cfg=cfg, drop=drop, diversity=diversity, analysis=analysis)
                except ValueError as exc:
                    log(f"[{row['source_id']}] {method}/{variant} @ {label} not applicable: {exc}")
                    continue
                stage_seconds = sum(analysis.timings.get(stage, 0.0) for stage in METHOD_STAGES[method])
                results.append({
                    "source_id": row["source_id"], "category": row["category"], "method": method,
                    "variant": variant, "budget": label, "budget_seconds": round(budget, 3),
                    "summary_seconds": summary.total_seconds,
                    **evaluate_intervals(summary.intervals(), annotation),
                    "redundancy": summary.redundancy,
                    "seconds_per_source_minute": stage_seconds / minutes if minutes > 0 else None,
                    "annotators": len(annotation.annotators), "annotator_agreement_f1": agreement,
                    "prompt": prompt or "", "evidence_used": "+".join(summary.weights),
                })
    if not results:
        raise ValueError("No results: every budget was at least as long as its source video.")
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out_dir / "summary.md").write_text(summarize_results(results), encoding="utf-8")
    log(f"Wrote {out_dir / 'results.csv'} and {out_dir / 'summary.md'}")
    return results


def _mean_sd(values: list[float | None]) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.fmean(values):.3f} ± {statistics.stdev(values):.3f}"


def summarize_results(results: list[dict]) -> str:
    """Per-category and pooled tables. Values are mean ± sample standard deviation over videos."""
    metrics = ("f1", "precision", "recall", "event_coverage", "redundancy", "seconds_per_source_minute")
    lines = ["# SceneSift evaluation summary", "",
             "Mean ± sample standard deviation across videos; `n` is the number of videos. "
             "With small `n` these are descriptive only.", ""]
    for scope in ("category", "pooled"):
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in results:
            key = (row["category"] if scope == "category" else "all", row["budget"],
                   row["method"], row["variant"])
            groups[key].append(row)
        lines += [f"## {'By category' if scope == 'category' else 'Pooled'}", "",
                  "| Category | Budget | Method | Variant | n | " + " | ".join(metrics) + " |",
                  "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in metrics) + " |"]
        for (category, budget, method, variant), rows in sorted(groups.items()):
            cells = " | ".join(_mean_sd([row[m] for row in rows]) for m in metrics)
            lines.append(f"| {category} | {budget} | {method} | {variant} | {len(rows)} | {cells} |")
        lines.append("")
    return "\n".join(lines)
