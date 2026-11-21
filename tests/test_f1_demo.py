"""Helpers behind the F1 demonstration picture and video."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from f1_demo import clock, frame_times, intersection, state_at  # noqa: E402

METHOD = [(10.0, 20.0), (40.0, 50.0)]
ANNOTATOR = [(15.0, 30.0), (70.0, 72.0)]


class F1DemoTests(unittest.TestCase):
    def test_intersection_is_the_time_both_picked(self):
        self.assertEqual(intersection(METHOD, ANNOTATOR), [(15.0, 20.0)])
        self.assertEqual(intersection(METHOD, [(60.0, 65.0)]), [])

    def test_state_names_who_picked_a_moment(self):
        self.assertEqual([state_at(t, METHOD, ANNOTATOR) for t in (5.0, 12.0, 17.0, 25.0, 45.0)],
                         ["neither", "method", "both", "annotator", "method"])

    def test_frames_include_one_of_each_kind(self):
        times = frame_times(METHOD, ANNOTATOR, 100.0, 8)
        self.assertEqual(len(times), 8)
        self.assertEqual(times, sorted(times))
        self.assertLessEqual({"both", "method", "annotator"}, {state_at(t, METHOD, ANNOTATOR) for t in times})

    def test_clock(self):
        self.assertEqual(clock(104.3), "1:44")


if __name__ == "__main__":
    unittest.main()
