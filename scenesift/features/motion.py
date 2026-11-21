"""Motion evidence from dense optical flow, with camera motion removed and cuts masked."""
from __future__ import annotations

import cv2
import numpy as np


def to_analysis_gray(frame_bgr: np.ndarray, width: int) -> np.ndarray:
    height = max(8, round(frame_bgr.shape[0] * width / max(1, frame_bgr.shape[1])))
    small = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def local_motion(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean flow magnitude after subtracting the median flow vector.

    The median approximates a global pan or shake, so what remains is movement of things in the scene.
    """
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    residual = flow - np.median(flow.reshape(-1, 2), axis=0)
    return float(np.linalg.norm(residual, axis=2).mean())


def histogram_difference(previous: np.ndarray, current: np.ndarray) -> float:
    """Bhattacharyya distance between grey-level histograms; large at hard cuts."""
    a = cv2.calcHist([previous], [0], None, [32], [0, 256])
    b = cv2.calcHist([current], [0], None, [32], [0, 256])
    cv2.normalize(a, a, 1.0, 0.0, cv2.NORM_L1)
    cv2.normalize(b, b, 1.0, 0.0, cv2.NORM_L1)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def mask_cuts(times: np.ndarray, motion: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Flow across a cut compares unrelated images; drop those samples."""
    masked = motion.astype(float).copy()
    for cut in cuts:
        index = int(np.searchsorted(times, cut - 1e-6))
        if 0 < index < len(times):
            masked[index] = np.nan
    return masked
