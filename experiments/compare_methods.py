"""Paired comparisons between methods from a results.csv written by `python -m scenesift evaluate`.

For each pair of methods it reports the mean per-video difference, a bootstrap 95% interval, the number
of videos where the first is better, and a Wilcoxon signed-rank p-value (not corrected for multiple
comparisons).

    python experiments/compare_methods.py experiments/results/tvsum/results.csv --metric kendall_tau
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PAIRS = [("blend", "multimodal"), ("blend", "classical"), ("blend", "activity"), ("blend", "semantic"),
         ("blend", "uniform"), ("classical", "uniform"), ("classical", "semantic"),
         ("classical", "activity"), ("multimodal", "classical"), ("multimodal", "activity"),
         ("multimodal", "semantic"), ("multimodal", "uniform")]


def load(path: Path, metric: str, budget: str | None) -> dict[str, dict[str, float]]:
    """{method: {source_id: value}} for full variants (and the uniform reference)."""
    table: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] not in ("full", "reference") or (budget and row["budget"] != budget):
                continue
            if row[metric] not in ("", "None"):
                table.setdefault(row["method"], {})[row["source_id"]] = float(row[metric])
    return table


def compare(first: dict[str, float], second: dict[str, float], samples: int = 10000) -> dict | None:
    shared = sorted(set(first) & set(second))
    if len(shared) < 3:
        return None
    difference = np.array([first[k] - second[k] for k in shared])
    rng = np.random.default_rng(0)
    means = rng.choice(difference, size=(samples, len(difference)), replace=True).mean(axis=1)
    p_value = stats.wilcoxon(difference).pvalue if np.any(difference != 0) else 1.0
    return {"n": len(shared), "mean": float(difference.mean()), "low": float(np.percentile(means, 2.5)),
            "high": float(np.percentile(means, 97.5)), "wins": int((difference > 0).sum()),
            "p": float(p_value)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--budget", help="Only rows with this budget label, e.g. 15%%.")
    args = parser.parse_args()
    try:
        table = load(args.results, args.metric, args.budget)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"| Comparison ({args.metric}) | Mean difference | 95% interval | First is better | p |")
    print("| --- | --- | --- | --- | --- |")
    for first, second in PAIRS:
        if first in table and second in table:
            result = compare(table[first], table[second])
            if result:
                print(f"| {first} minus {second} | {result['mean']:+.3f} | {result['low']:+.3f} to "
                      f"{result['high']:+.3f} | {result['wins']} of {result['n']} | {result['p']:.3g} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
