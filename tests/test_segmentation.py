"""Candidate generation from shots and overlapping windows."""
import unittest

import numpy as np

from scenesift.config import Config
from scenesift.datatypes import Segment
from scenesift.segmentation import build_candidates, cuts_from_differences, drop_blank


class SegmentationTests(unittest.TestCase):
    def test_candidates_cover_the_video_and_respect_length_limits(self):
        cfg = Config()
        candidates = build_candidates(60.0, [10.0, 13.0, 40.0], cfg)
        self.assertTrue(all(0.0 <= c.start < c.end <= 60.0 for c in candidates))
        self.assertTrue(all(c.length >= cfg.min_segment_seconds - 1e-9 for c in candidates))
        covered = np.zeros(600, dtype=bool)
        for c in candidates:
            covered[int(c.start * 10):int(c.end * 10)] = True
        self.assertTrue(covered.all())

    def test_shots_of_usable_length_become_candidates(self):
        candidates = build_candidates(60.0, [10.0, 13.0, 40.0], Config())
        shots = {(c.start, c.end) for c in candidates if c.source == "shot"}
        self.assertIn((0.0, 10.0), shots)
        self.assertIn((10.0, 13.0), shots)
        self.assertNotIn((13.0, 40.0), shots)  # longer than max_segment_seconds; windows cover it

    def test_window_edges_snap_to_nearby_cuts(self):
        candidates = build_candidates(60.0, [8.4], Config(window_seconds=[8.0], window_overlap=0.0))
        self.assertIn(8.4, {c.end for c in candidates})

    def test_long_shot_gets_overlapping_windows(self):
        candidates = build_candidates(40.0, [], Config(window_seconds=[4.0], window_overlap=0.5))
        starts = sorted(c.start for c in candidates)
        self.assertEqual(starts[:3], [0.0, 2.0, 4.0])

    def test_very_short_video_is_one_candidate(self):
        candidates = build_candidates(1.5, [], Config())
        self.assertEqual([(c.start, c.end) for c in candidates], [(0.0, 1.5)])

    def test_mostly_blank_candidates_are_dropped(self):
        times = np.arange(0, 20, 0.25)
        detail = np.where(times >= 16, 1.0, 40.0)  # the last four seconds are black
        segments = [Segment(0.0, 8.0), Segment(8.0, 16.0), Segment(13.0, 19.0), Segment(16.0, 20.0)]
        kept = drop_blank(segments, times, detail, threshold=6.0, max_fraction=0.5)
        self.assertEqual(kept, segments[:3])  # 13-19 s is exactly half blank, which is allowed

    def test_uniformly_dark_video_keeps_its_candidates(self):
        times = np.arange(0, 10, 0.25)
        segments = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
        self.assertEqual(drop_blank(segments, times, np.full(times.size, 2.0), 6.0, 0.5), segments)

    def test_fallback_cut_detector_finds_outliers_only(self):
        times = np.arange(0, 10, 0.25)
        differences = np.full(times.size, 0.02)
        differences[20] = 0.9
        self.assertEqual(cuts_from_differences(times, differences), [5.0])
        self.assertEqual(cuts_from_differences(times, np.full(times.size, 0.02)), [])


if __name__ == "__main__":
    unittest.main()
