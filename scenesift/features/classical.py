"""Hand-built visual cues measured directly from pixels. No pretrained model is involved.

Every cue uses techniques from an introductory computer-vision course:

- SIFT keypoints, descriptor matching with the ratio test, and a homography fitted with RANSAC
  align two consecutive frames, which separates camera motion from motion in the scene;
- image subtraction of the aligned frames measures what moved;
- Gaussian smoothing and thresholding turn that difference into the share of the frame that moved;
- the Canny detector (Gaussian smoothing, Sobel gradients, non-maximum suppression, hysteresis
  thresholding) measures how much structure a frame holds.

The colour cue (HSV histograms) is built from `HistogramEncoder` in `visual.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RATIO = 0.75  # Lowe's ratio test: keep a match only if it is clearly better than the runner-up
MIN_INLIERS = 8  # below this the homography is not trusted
REPROJECTION_PIXELS = 3.0  # RANSAC inlier threshold
MOVING_THRESHOLD = 20  # grey levels; a smoothed difference above this counts as movement
CANNY_LOW, CANNY_HIGH = 50, 150  # hysteresis thresholds


def edge_density(gray: np.ndarray) -> float:
    """Share of pixels the Canny detector marks as edges: how much structure the frame holds."""
    return float((cv2.Canny(gray, CANNY_LOW, CANNY_HIGH) > 0).mean())


def frame_change(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean absolute grey-level difference between two frames (plain image subtraction)."""
    return float(cv2.absdiff(previous, current).mean())


@dataclass(frozen=True)
class Movement:
    """What changed between two consecutive frames. NaN where a value cannot be measured."""

    residual: float  # mean difference after alignment: movement of things in the scene
    moving: float  # share of the frame whose smoothed, aligned difference exceeds the threshold
    camera: float  # mean displacement of the frame corners under the homography, in pixels


NO_MOVEMENT = Movement(float("nan"), float("nan"), float("nan"))


class FrameAligner:
    """Separates camera motion from motion in the scene for consecutive frames.

    SIFT keypoints of the two frames are matched, a homography is fitted to the matches with
    RANSAC, and the earlier frame is warped onto the later one. The homography itself says how
    far the camera moved; subtracting the aligned frames leaves the movement of things in the
    scene, plus anything a single plane-to-plane mapping cannot explain. When too few matches
    support a homography the frames are subtracted unaligned and the camera motion is unknown.
    """

    def __init__(self) -> None:
        self._sift = cv2.SIFT_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._previous: tuple | None = None  # (frame, keypoints, descriptors) of the last call

    def _describe(self, gray: np.ndarray) -> tuple:
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        return gray, keypoints, descriptors

    def homography(self, earlier: tuple, later: tuple) -> np.ndarray | None:
        """Homography mapping the earlier frame onto the later one, or None if it cannot be trusted."""
        _, keys_a, desc_a = earlier
        _, keys_b, desc_b = later
        if desc_a is None or desc_b is None or len(keys_a) < MIN_INLIERS or len(keys_b) < 2:
            return None
        pairs = self._matcher.knnMatch(desc_a, desc_b, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) < MIN_INLIERS:
            return None
        source = np.float32([keys_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([keys_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, REPROJECTION_PIXELS)
        if matrix is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
            return None
        return matrix

    def measure(self, gray: np.ndarray) -> Movement:
        """Compare this frame with the previous one. Call once per frame, in order."""
        current = self._describe(gray)
        previous, self._previous = self._previous, current
        if previous is None:
            return NO_MOVEMENT
        earlier = previous[0]
        height, width = gray.shape
        smooth_earlier = cv2.GaussianBlur(earlier, (5, 5), 0)
        smooth_now = cv2.GaussianBlur(gray, (5, 5), 0)
        matrix = self.homography(previous, current)
        covered = None
        if matrix is not None:
            # Only compare pixels the warped frame actually covers.
            covered = cv2.warpPerspective(np.full((height, width), 255, np.uint8), matrix, (width, height)) > 0
            if covered.mean() < 0.5:
                matrix = None
        if matrix is None:
            smooth_difference = cv2.absdiff(smooth_earlier, smooth_now)
            return Movement(frame_change(earlier, gray), float((smooth_difference > MOVING_THRESHOLD).mean()),
                            float("nan"))
        difference = cv2.absdiff(cv2.warpPerspective(earlier, matrix, (width, height)), gray)
        smooth_difference = cv2.absdiff(cv2.warpPerspective(smooth_earlier, matrix, (width, height)), smooth_now)
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        shift = np.linalg.norm(cv2.perspectiveTransform(corners, matrix) - corners, axis=2).mean()
        return Movement(float(difference[covered].mean()),
                        float((smooth_difference[covered] > MOVING_THRESHOLD).mean()), float(shift))
