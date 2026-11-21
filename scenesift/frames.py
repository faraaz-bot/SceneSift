"""Sequential frame sampling with timestamps derived from the source frame index."""
from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


def iter_frames(path: str, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time_seconds, BGR frame) at roughly `fps` samples per second."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the video.")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps != source_fps or source_fps <= 0:
        source_fps = 30.0
    step = 1.0 / fps
    next_time = 0.0
    index = 0
    try:
        while capture.grab():
            time = index / source_fps
            index += 1
            if time + 1e-9 < next_time:
                continue
            ok, frame = capture.retrieve()
            if ok:
                yield time, frame
            while next_time <= time + 1e-9:
                next_time += step
    finally:
        capture.release()


def can_decode(path: str) -> bool:
    """True when OpenCV can read at least one frame."""
    capture = cv2.VideoCapture(path)
    try:
        return bool(capture.isOpened() and capture.read()[0])
    finally:
        capture.release()
