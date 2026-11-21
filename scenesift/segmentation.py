"""Shot boundaries and candidate segments (shots plus overlapping windows)."""
from __future__ import annotations

import numpy as np

from .config import Config
from .datatypes import Segment


def detect_cuts_scenedetect(path: str) -> list[float]:
    from scenedetect import ContentDetector, detect

    scenes = detect(path, ContentDetector())
    # `seconds` replaced get_seconds() in PySceneDetect 0.7.
    return [float(start.seconds if hasattr(start, "seconds") else start.get_seconds()) for start, _ in scenes[1:]]


def cuts_from_differences(times: np.ndarray, differences: np.ndarray) -> list[float]:
    """Fallback detector: a cut is a frame-to-frame histogram change far above the video's norm."""
    finite = differences[np.isfinite(differences)]
    if finite.size < 3:
        return []
    median = float(np.median(finite))
    spread = float(np.median(np.abs(finite - median))) * 1.4826
    threshold = max(0.4, median + 8.0 * spread)
    return [float(t) for t, d in zip(times, differences) if np.isfinite(d) and d > threshold]


def _snap(value: float, cuts: np.ndarray, tolerance: float) -> float:
    if cuts.size == 0:
        return value
    nearest = cuts[np.argmin(np.abs(cuts - value))]
    return float(nearest) if abs(nearest - value) <= tolerance else value


def _iou(a: Segment, b: Segment) -> float:
    overlap = min(a.end, b.end) - max(a.start, b.start)
    if overlap <= 0:
        return 0.0
    return overlap / (max(a.end, b.end) - min(a.start, b.start))


def drop_blank(segments: list[Segment], times: np.ndarray, detail: np.ndarray, threshold: float,
               max_fraction: float) -> list[Segment]:
    """Remove candidates made mostly of featureless frames (black, fades, solid colour).

    `detail` is the grey-level standard deviation of each sampled frame. If every candidate would be
    removed (a uniformly dark video), nothing is.
    """
    kept = []
    for segment in segments:
        inside = detail[(times >= segment.start) & (times < segment.end)]
        if inside.size == 0 or float((inside < threshold).mean()) <= max_fraction:
            kept.append(segment)
    return kept or segments


def build_candidates(duration: float, cuts: list[float], cfg: Config) -> list[Segment]:
    """Shots of usable length plus overlapping windows, so events inside long shots are not lost."""
    if duration <= cfg.min_segment_seconds:
        return [Segment(0.0, float(duration), "window")]
    cut_array = np.array(sorted(c for c in cuts if 0 < c < duration), dtype=float)
    candidates: list[Segment] = []

    bounds = [0.0, *cut_array.tolist(), float(duration)]
    for start, end in zip(bounds, bounds[1:]):
        if cfg.min_segment_seconds <= end - start <= cfg.max_segment_seconds:
            candidates.append(Segment(start, end, "shot"))

    for window in cfg.window_seconds:
        window = min(float(window), float(duration))
        stride = window * (1.0 - cfg.window_overlap)
        start = 0.0
        while start < duration - cfg.min_segment_seconds + 1e-9:
            snapped_start = _snap(start, cut_array, cfg.snap_seconds) if start > 0 else 0.0
            end = min(start + window, duration)
            snapped_end = _snap(end, cut_array, cfg.snap_seconds) if end < duration else float(duration)
            if snapped_end - snapped_start >= cfg.min_segment_seconds:
                candidates.append(Segment(snapped_start, snapped_end, "window"))
            start += stride

    candidates.sort(key=lambda s: (s.start, s.end, s.source != "shot"))
    kept: list[Segment] = []
    for candidate in candidates:
        if not any(_iou(candidate, other) > 0.9 for other in kept[-8:]):
            kept.append(candidate)
    return kept or [Segment(0.0, float(duration), "window")]
