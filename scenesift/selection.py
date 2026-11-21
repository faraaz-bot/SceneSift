"""Choose clips under a duration budget, avoiding overlap and (optionally) repeated content."""
from __future__ import annotations

import numpy as np

from .datatypes import Clip, Segment


def _padded(segment: Segment, before: float, after: float, duration: float) -> tuple[float, float]:
    return max(0.0, segment.start - before), min(duration, segment.end + after)


def _union_length(spans: list[tuple[float, float]]) -> float:
    total, reach = 0.0, -np.inf
    for start, end in sorted(spans):
        if end > reach:
            total += end - max(start, reach)
            reach = end
    return total


def select(segments: list[Segment], scores: np.ndarray, budget: float, duration: float, *,
           embeddings: np.ndarray | None = None, diversity: bool = False, mmr_lambda: float = 0.7,
           max_overlap: float = 0.25, context_before: float = 1.0, context_after: float = 1.0,
           min_clip_seconds: float = 2.0) -> list[Clip]:
    """Greedy selection; with `diversity`, maximal marginal relevance against clips already chosen.

    Candidates are taken in order of value. Context padding counts toward the budget, and the last
    clip is trimmed to whatever budget remains, so the result never exceeds the budget.
    Clips are returned in source order, with overlapping ones merged.
    """
    if budget <= 0:
        raise ValueError("The duration budget must be positive.")
    if budget >= duration:
        return [Clip(0.0, float(duration), float(np.max(scores)), list(range(len(segments))))]

    spans = [_padded(s, context_before, context_after, duration) for s in segments]
    chosen: dict[int, tuple[float, float]] = {}
    skipped: set[int] = set()
    while True:
        # Overlapping context is only counted once, so the budget is measured on the merged timeline.
        used = _union_length(list(chosen.values()))
        remaining = float(budget) - used
        if remaining < min_clip_seconds and chosen:
            break
        best, best_value = None, -np.inf
        for index, (start, end) in enumerate(spans):
            if index in chosen or index in skipped:
                continue
            blocked = False
            for o_start, o_end in chosen.values():
                overlap = min(end, o_end) - max(start, o_start)
                if overlap > max_overlap * min(end - start, o_end - o_start):
                    blocked = True
                    break
            if blocked:
                continue
            value = float(scores[index])
            if diversity and chosen and embeddings is not None:
                similarity = max(float(embeddings[index] @ embeddings[other]) for other in chosen)
                value = mmr_lambda * value - (1.0 - mmr_lambda) * max(0.0, similarity)
            if value > best_value:
                best, best_value = index, value
        if best is None:
            break
        start, end = spans[best]
        added = _union_length([*chosen.values(), (start, end)]) - used
        if added > remaining + 1e-9:
            if added < end - start - 1e-9:
                skipped.add(best)  # touches a chosen clip and does not fit; try the next candidate
                continue
            # Spend what is left on the middle of the best remaining candidate rather than on a
            # weaker clip that happens to fit.
            centre = (segments[best].start + segments[best].end) / 2.0
            start = min(max(start, centre - remaining / 2.0), end - remaining)
            end = start + remaining
        chosen[best] = (start, end)

    clips: list[Clip] = []
    for index, (start, end) in sorted(chosen.items(), key=lambda item: item[1]):
        if clips and start <= clips[-1].end + 1e-6:
            last = clips[-1]
            last.end = max(last.end, end)
            last.score = max(last.score, float(scores[index]))
            last.members.append(index)
        else:
            clips.append(Clip(start, end, float(scores[index]), [index]))
    return clips
