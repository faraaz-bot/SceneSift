"""Normalization and fusion, including missing-modality weight redistribution."""
import unittest

import numpy as np

from scenesift.scoring import combine, fuse, robust_unit

WEIGHTS = {"visual": 0.30, "motion": 0.20, "audio": 0.15, "speech": 0.20, "novelty": 0.15}


class ScoringTests(unittest.TestCase):
    def test_robust_unit_is_monotonic_and_bounded(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        unit = robust_unit(values)
        self.assertTrue(np.all(np.diff(unit) > 0))
        self.assertTrue(np.all((unit > 0) & (unit <= 1)))
        self.assertAlmostEqual(unit[2], 0.5)

    def test_robust_unit_keeps_nan_and_handles_constant_input(self):
        unit = robust_unit(np.array([2.0, np.nan, 2.0]))
        self.assertTrue(np.isnan(unit[1]))
        self.assertEqual(unit[0], 0.5)
        self.assertTrue(np.isnan(robust_unit(np.array([np.nan, np.nan]))).all())

    def test_combine_ignores_missing_parts(self):
        self.assertIsNone(combine([None, np.array([np.nan, np.nan])]))
        merged = combine([np.array([0.2, np.nan]), np.array([0.6, 0.4]), None])
        np.testing.assert_allclose(merged, [0.4, 0.4])

    def test_weights_are_renormalized_over_available_evidence(self):
        features = {"visual": np.array([1.0, 0.0]), "motion": np.array([0.0, 1.0]), "audio": None,
                    "speech": None, "novelty": None}
        score, contributions, applied = fuse(features, WEIGHTS, WEIGHTS)
        self.assertAlmostEqual(sum(applied.values()), 1.0)
        self.assertAlmostEqual(applied["visual"], 0.6)
        np.testing.assert_allclose(score, [0.6, 0.4])
        self.assertEqual(set(contributions), {"visual", "motion"})

    def test_segment_without_evidence_gets_the_neutral_value(self):
        features = {"speech": np.array([np.nan, 1.0])}
        score, _, _ = fuse(features, WEIGHTS, ["speech"])
        np.testing.assert_allclose(score, [0.5, 1.0])

    def test_method_with_no_available_evidence_is_an_error(self):
        with self.assertRaises(ValueError):
            fuse({"motion": None, "audio": None}, WEIGHTS, ["motion", "audio"])


if __name__ == "__main__":
    unittest.main()
