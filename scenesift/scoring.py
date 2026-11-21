"""Per-video normalization and weighted fusion of the available evidence."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def robust_unit(values: np.ndarray) -> np.ndarray:
    """Map values to (0, 1) relative to this video: logistic of the median/MAD z-score. NaN stays NaN."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    median = np.median(values[finite])
    spread = np.median(np.abs(values[finite] - median)) * 1.4826
    if spread < 1e-9:
        spread = values[finite].std()
    if spread < 1e-9:
        out[finite] = 0.5
        return out
    z = np.clip((values[finite] - median) / spread, -30.0, 30.0)
    out[finite] = 1.0 / (1.0 + np.exp(-z))
    return out


def combine(parts: list[np.ndarray | None]) -> np.ndarray | None:
    """Mean of the sub-signals that exist, ignoring per-segment gaps."""
    present = [p for p in parts if p is not None and np.isfinite(p).any()]
    if not present:
        return None
    stacked = np.vstack(present)
    counts = np.isfinite(stacked).sum(axis=0)
    sums = np.nansum(stacked, axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def fuse(features: dict[str, np.ndarray | None], weights: dict[str, float], use: Iterable[str]):
    """Weighted sum over the evidence that exists for this video.

    Weights of missing modalities are redistributed over the rest. A segment with no value for an
    available modality (for example no speech in that window) gets the neutral 0.5.
    Returns (score, per-feature contributions, the weights actually applied).
    """
    available = {name: features[name] for name in use
                 if features.get(name) is not None and weights.get(name, 0.0) > 0.0}
    if not available:
        raise ValueError("None of the evidence this method relies on is available for this video.")
    total = sum(weights[name] for name in available)
    applied = {name: weights[name] / total for name in available}
    contributions = {name: applied[name] * np.nan_to_num(values, nan=0.5)
                     for name, values in available.items()}
    score = np.sum(list(contributions.values()), axis=0)
    return score, contributions, applied
