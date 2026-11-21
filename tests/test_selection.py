"""Budgeted selection: budget is respected, overlap is avoided, diversity changes the choice."""
import unittest

import numpy as np

from scenesift.selection import select
from scenesift.datatypes import Segment


def total(clips):
    return sum(clip.length for clip in clips)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.segments = [Segment(float(s), float(s + 4)) for s in range(0, 60, 2)]
        self.scores = np.linspace(0.1, 0.9, len(self.segments))

    def test_never_exceeds_budget(self):
        for budget in (3, 10, 17, 30, 45):
            clips = select(self.segments, self.scores, budget, 62.0)
            self.assertLessEqual(total(clips), budget + 1e-6)
            self.assertGreater(total(clips), 0)

    def test_uses_most_of_the_budget(self):
        clips = select(self.segments, self.scores, 20, 62.0)
        self.assertGreaterEqual(total(clips), 18.0)

    def test_overlapping_context_is_not_double_counted(self):
        # Neighbouring windows share padding; the merged reel should still fill the budget.
        segments = [Segment(float(s), float(s + 8)) for s in range(0, 120, 4)]
        scores = np.concatenate([np.linspace(0.2, 0.9, 15), np.linspace(0.9, 0.2, 15)])
        for budget in (20, 30, 45):
            clips = select(segments, scores, budget, 124.0, max_overlap=0.5)
            self.assertLessEqual(total(clips), budget + 1e-6)
            self.assertGreaterEqual(total(clips), budget - 2.0)

    def test_clips_are_chronological_and_disjoint(self):
        clips = select(self.segments, self.scores, 30, 62.0)
        for earlier, later in zip(clips, clips[1:]):
            self.assertLess(earlier.end, later.start + 1e-6)

    def test_highest_scoring_segment_is_included(self):
        clips = select(self.segments, self.scores, 12, 62.0)
        best = self.segments[int(np.argmax(self.scores))]
        self.assertTrue(any(c.start <= best.start and c.end >= best.end for c in clips))

    def test_budget_shorter_than_any_candidate_trims_the_best(self):
        clips = select(self.segments, self.scores, 1.5, 62.0)
        self.assertEqual(len(clips), 1)
        self.assertAlmostEqual(clips[0].length, 1.5)
        best = self.segments[int(np.argmax(self.scores))]
        self.assertGreaterEqual(clips[0].start, best.start - 1.0)
        self.assertLessEqual(clips[0].end, best.end + 1.0)

    def test_budget_at_least_duration_returns_whole_video(self):
        clips = select(self.segments, self.scores, 100, 62.0)
        self.assertEqual([(c.start, c.end) for c in clips], [(0.0, 62.0)])

    def test_context_padding_is_clamped_to_the_video(self):
        clips = select([Segment(0.0, 4.0), Segment(20.0, 24.0)], np.array([0.9, 0.8]), 12, 24.0)
        self.assertEqual(clips[0].start, 0.0)
        self.assertEqual(clips[-1].end, 24.0)

    def test_diversity_prefers_dissimilar_content(self):
        segments = [Segment(0.0, 4.0), Segment(20.0, 24.0), Segment(40.0, 44.0)]
        scores = np.array([0.9, 0.85, 0.6])
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        plain = select(segments, scores, 12, 60.0, embeddings=embeddings, diversity=False)
        diverse = select(segments, scores, 12, 60.0, embeddings=embeddings, diversity=True, mmr_lambda=0.5)
        self.assertEqual([c.members for c in plain], [[0], [1]])
        self.assertEqual([c.members for c in diverse], [[0], [2]])

    def test_rejects_non_positive_budget(self):
        with self.assertRaises(ValueError):
            select(self.segments, self.scores, 0, 62.0)


if __name__ == "__main__":
    unittest.main()
