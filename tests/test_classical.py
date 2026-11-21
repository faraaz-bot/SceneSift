"""Classical cues, the classical and blend methods, rank correlation, and weight fitting."""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from scenesift.config import ALL_FEATURES, CLASSICAL_FEATURES, Config
from scenesift.datatypes import Segment
from scenesift.evaluation import (Annotation, load_annotation, rank_correlation, reference_scores,
                                  run_comparison, temporal_prf)
from scenesift.features.classical import edge_strength, frame_change
from scenesift.fitting import fit_weights, run_fit, weight_grid
from scenesift.pipeline import Scored, analyze, importance_curve, score, summarize
from synthetic import EVENTS, make_video


class CueTests(unittest.TestCase):
    def test_edge_strength_is_zero_for_flat_and_high_for_textured_frames(self):
        flat = np.full((60, 80), 90, dtype=np.uint8)
        stripes = np.tile(np.array([0, 255], dtype=np.uint8).repeat(4), (60, 10))
        self.assertEqual(edge_strength(flat), 0.0)
        self.assertGreater(edge_strength(stripes), 50.0)

    def test_frame_change_is_the_mean_absolute_difference(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.full((10, 10), 30, dtype=np.uint8)
        self.assertEqual(frame_change(a, a), 0.0)
        self.assertAlmostEqual(frame_change(a, b), 30.0)


class RankCorrelationTests(unittest.TestCase):
    def test_same_and_reversed_orderings(self):
        reference = {"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        self.assertAlmostEqual(rank_correlation(np.arange(5.0), reference)["kendall_tau"], 1.0)
        self.assertAlmostEqual(rank_correlation(-np.arange(5.0), reference)["spearman_rho"], -1.0)

    def test_constant_scores_are_undefined(self):
        result = rank_correlation(np.ones(5), {"a": np.arange(5.0)})
        self.assertEqual(result, {"kendall_tau": None, "spearman_rho": None})

    def test_intervals_become_per_second_selection_shares(self):
        annotation = Annotation("v", "test", {"a": [(1.0, 2.5)]})
        step, scores = reference_scores(annotation, 4.0)
        self.assertEqual(step, 1.0)
        np.testing.assert_allclose(scores["a"], [0.0, 1.0, 0.5, 0.0])

    def test_importance_scores_are_used_when_present(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "a.json"
            path.write_text(json.dumps({"video_id": "v", "annotators": {"a": [[0, 2]]},
                                        "importance": {"step_seconds": 2.0, "annotators": {"a": [5, 1, 3]}}}))
            step, scores = reference_scores(load_annotation(path), 6.0)
        self.assertEqual(step, 2.0)
        np.testing.assert_allclose(scores["a"], [5.0, 1.0, 3.0])

    def test_importance_curve_averages_covering_candidates(self):
        scored = Scored(segments=[Segment(0.0, 4.0), Segment(2.0, 6.0)], features={},
                        score=np.array([1.0, 3.0]), contributions={}, weights={},
                        embeddings=np.zeros((2, 1)), transcripts=["", ""], notes=[])
        np.testing.assert_allclose(importance_curve(scored, 8.0, 2.0), [1.0, 2.0, 3.0, 1.0])


class FitTests(unittest.TestCase):
    def test_weight_goes_to_the_feature_that_tracks_the_target(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(200, 8))
        weights = fit_weights([(features, features[:, 2] * 0.5)], ALL_FEATURES)
        self.assertAlmostEqual(weights["audio"], 1.0, places=3)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)

    def test_fitting_fails_when_nothing_is_related(self):
        features = np.eye(8)
        with self.assertRaises(ValueError):
            fit_weights([(features, -np.ones(8))], ALL_FEATURES)

    def test_grid_covers_the_simplex(self):
        grid = weight_grid(("a", "b", "c"), 2)
        self.assertEqual(len(grid), 6)
        self.assertIn({"a": 0.5, "b": 0.0, "c": 0.5}, grid)
        self.assertTrue(all(abs(sum(w.values()) - 1.0) < 1e-9 for w in grid))


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.root.name)
        cls.video = str(make_video(cls.dir))
        cls.cfg = Config(visual_backend="histogram", speech_backend="none", text_backend="none",
                         cache_dir=str(cls.dir / "cache"))
        cls.analysis = analyze(cls.video, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def test_classical_method_uses_only_pixel_cues_and_finds_the_events(self):
        summary = summarize(self.video, target_seconds=9, method="classical", cfg=self.cfg,
                            analysis=self.analysis)
        self.assertEqual(set(summary.weights), set(CLASSICAL_FEATURES))
        self.assertGreater(temporal_prf(summary.intervals(), EVENTS)[2], 0.6)

    def test_blend_uses_fitted_weights_and_skips_zero_weights(self):
        cfg = replace(self.cfg, method_weights={"blend": {"motion": 0.75, "edges": 0.25, "visual": 0.0}})
        summary = summarize(self.video, target_seconds=9, method="blend", cfg=cfg, analysis=self.analysis)
        self.assertEqual(summary.weights, {"motion": 0.75, "edges": 0.25})
        self.assertEqual(summary.evidence_missing, [])

    def test_analysis_round_trips_classical_measurements_through_the_cache(self):
        again = analyze(self.video, self.cfg)
        np.testing.assert_allclose(again.edges, self.analysis.edges)
        np.testing.assert_allclose(again.change, self.analysis.change)
        np.testing.assert_allclose(again.colour_embeddings, self.analysis.colour_embeddings)
        self.assertEqual(len(self.analysis.edges), len(self.analysis.motion_times))

    def test_scores_rank_the_events_above_the_rest(self):
        scored = score(self.analysis, self.cfg, None, "classical")
        curve = importance_curve(scored, self.analysis.duration, 1.0)
        truth = np.array([any(s <= k + 0.5 < e for s, e in EVENTS) for k in range(len(curve))], dtype=float)
        self.assertGreater(rank_correlation(curve, {"truth": truth})["kendall_tau"], 0.4)

    def test_fit_and_comparison_runner(self):
        other = self.dir / "second"
        other.mkdir()
        second = make_video(other)
        (self.dir / "ann.json").write_text(json.dumps({
            "video_id": "synthetic", "category": "synthetic", "annotators": {"a1": EVENTS, "a2": EVENTS}}))
        with open(self.dir / "manifest.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_id", "category", "path", "annotation", "split"])
            writer.writerow(["one", "synthetic", self.video, "ann.json", "train"])
            writer.writerow(["two", "synthetic", str(second), "ann.json", "train"])
        report = run_fit(self.dir / "manifest.csv", self.dir / "fitted.yaml", cfg=self.cfg, budget=8.0,
                         log=lambda _: None)
        self.assertEqual(report["videos"], 2)
        self.assertGreater(report["held_out_f1"], 0.6)
        fitted = Config.from_dict({**self.cfg.to_dict(),
                                   **Config.from_file(self.dir / "fitted.yaml").to_dict(),
                                   "visual_backend": "histogram", "speech_backend": "none",
                                   "text_backend": "none", "cache_dir": self.cfg.cache_dir})
        self.assertEqual(fitted.method_weights, {"blend": report["weights"]})
        searched = run_fit(self.dir / "manifest.csv", self.dir / "fitted.yaml", cfg=self.cfg, budget=8.0,
                           method="activity", how="search", grid_steps=4, log=lambda _: None)
        self.assertAlmostEqual(sum(searched["weights"].values()), 1.0)
        self.assertEqual(set(Config.from_file(self.dir / "fitted.yaml").method_weights), {"blend", "activity"})
        results = run_comparison(self.dir / "manifest.csv", self.dir / "out", cfg=fitted, split="train",
                                 methods=("classical", "blend"), budgets=(8.0,), log=lambda _: None)
        by_method = {r["method"]: r for r in results if r["source_id"] == "one"}
        self.assertIsNone(by_method["uniform"]["kendall_tau"])
        for method in ("classical", "blend"):
            self.assertGreater(by_method[method]["kendall_tau"], 0.3)
            self.assertIsNotNone(by_method[method]["annotator_agreement_tau"])
        self.assertIn("kendall_tau", (self.dir / "out" / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
