"""Hand-built visual cues measured directly from pixels. No pretrained model is involved."""
from __future__ import annotations

import cv2
import numpy as np


def edge_strength(gray: np.ndarray) -> float:
    """Mean Sobel gradient magnitude: how much fine structure (edges, texture) the frame holds."""
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(cv2.magnitude(horizontal, vertical).mean())


def frame_change(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean absolute grey-level difference between consecutive samples."""
    return float(cv2.absdiff(previous, current).mean())
