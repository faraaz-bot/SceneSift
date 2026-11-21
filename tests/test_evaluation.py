"""Interval metrics and the manifest-driven comparison runner."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scenesift.config import Config
from scenesift.evaluation import (Annotation, annotator_agreement, evaluate_intervals, event_coverage,
                                  merge, overlap_length, parse_budget, run_comparison, temporal_prf,
                                  uniform_intervals)
from synthetic import EVENTS, make_video


class MetricTests(unittest.TestCase):
    def test_merge_and_overlap(self):
        self.assertEqual(merge([(5, 8), (0, 2), (1, 3), (8, 9)]), [(0.0, 3.0), (5.0, 9.0)])
        self.assertAlmostEqual(overlap_length([(0, 10)], [(2, 4), (8, 12)]), 4.0)
        self.assertAlmostEqual(overlap_length([(0, 1)], [(2, 3)]), 0.0)

    def test_precision_recall_f1(self):
        precision, recall, f1 = temporal_prf([(0, 10)], [(5, 15)])
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)
        self.assertEqual(temporal_prf([(0, 10)], [(0, 10)]), (1.0, 1.0, 1.0))
        self.assertEqual(temporal_prf([], [(0, 10)]), (0.0, 0.0, 0.0))

    def test_event_coverage_threshold(self):
        events = [(0, 4), (10, 14), (20, 24)]
        self.assertAlmostEqual(event_coverage([(0, 2), (10, 11), (30, 40)], events), 1 / 3)
        self.assertAlmostEqual(event_coverage([(0, 2), (10, 11)], events, min_fraction=0.25), 2 / 3)

    def test_budget_parsing(self):
        self.assertEqual(parse_budget(30, 200.0), ("30s", 30.0))
        self.assertEqual(parse_budget("30", 200.0), ("30s", 30.0))
        self.assertEqual(parse_budget("15%", 200.0), ("15%", 30.0))
        for bad in ("0", "-5", "0%", "150%"):
            with self.assertRaises(ValueError):
                parse_budget(bad, 200.0)

    def test_uniform_reference_fills_the_budget_evenly(self):
        clips = uniform_intervals(100.0, 20.0)
        self.assertEqual(len(clips), 4)
        self.assertAlmostEqual(sum(e - s for s, e in clips), 20.0)
        self.assertAlmostEqual(clips[0][0], 10.0)
        self.assertAlmostEqual(clips[-1][1], 90.0)
        self.assertEqual(uniform_intervals(100.0, 1.0), [(49.5, 50.5)])

    def test_scores_are_averaged_over_annotators(self):
        annotation = Annotation("v", "test", {"a": [(0, 10)], "b": [(20, 30)]})
        result = evaluate_intervals([(0, 10)], annotation)
        self.assertAlmostEqual(result["f1"], 0.5)
        self.assertAlmostEqual(result["f1_best_annotator"], 1.0)
        self.assertAlmostEqual(annotator_agreement(annotation), 0.0)
        self.assertIsNone(annotator_agreement(Annotation("v", "test", {"a": [(0, 1)]})))


class ComparisonRunnerTests(unittest.TestCase):
    def test_runner_writes_results_for_every_method(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = make_video(root)
            (root / "ann.json").write_text(json.dumps({
                "video_id": "synthetic", "category": "synthetic",
                "annotators": {"a1": EVENTS, "a2": [[8.5, 12.0], [18.0, 20.5]]}}))
            with open(root / "manifest.csv", "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source_id", "category", "path", "annotation", "split"])
                writer.writerow(["synthetic", "synthetic", video.name, "ann.json", "test"])
                writer.writerow(["ignored", "synthetic", "missing.mp4", "missing.json", "train"])
            cfg = Config(visual_backend="histogram", speech_backend="none", text_backend="none",
                         cache_dir=str(root / "cache"))
            results = run_comparison(root / "manifest.csv", root / "out", cfg=cfg,
                                     budgets=(8.0, 60.0, "25%"), ablations=True, log=lambda _: None)
            methods = {(r["method"], r["variant"]) for r in results}
            self.assertLessEqual({("uniform", "reference"), ("activity", "full"), ("semantic", "full"),
                                  ("multimodal", "full"),
                                  ("multimodal", "without-motion"), ("multimodal", "without-diversity")},
                                 methods)
            self.assertEqual({r["budget"] for r in results}, {"8s", "25%"})  # 60 s exceeds the source
            self.assertEqual({r["budget_seconds"] for r in results}, {8.0, 6.0})
            self.assertTrue(all(r["summary_seconds"] <= r["budget_seconds"] + 1e-6 for r in results))
            activity = next(r for r in results if r["method"] == "activity")
            self.assertGreater(activity["f1"], 0.6)
            self.assertTrue((root / "out" / "results.csv").is_file())
            report = (root / "out" / "summary.md").read_text()
            self.assertIn("| synthetic | 8s | activity | full | 1 |", report)
            self.assertIn("| synthetic | 25% | activity | full | 1 |", report)


if __name__ == "__main__":
    unittest.main()
