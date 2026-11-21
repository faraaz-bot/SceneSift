"""End-to-end run on a synthetic video using the dependency-light backends."""
import json
import tempfile
import unittest
from pathlib import Path

from scenesift.config import Config
from scenesift.evaluation import temporal_prf
from scenesift.media import inspect_video
from scenesift.pipeline import analyze, summarize
from synthetic import EVENTS, make_video


class PipelineTests(unittest.TestCase):
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

    def test_shot_cuts_are_found(self):
        for expected in (8.0, 12.0, 18.0, 21.0):
            self.assertTrue(any(abs(cut - expected) < 0.3 for cut in self.analysis.cuts), self.analysis.cuts)

    def test_activity_method_finds_the_active_scenes(self):
        summary = summarize(self.video, target_seconds=9, method="activity", cfg=self.cfg,
                            analysis=self.analysis)
        self.assertLessEqual(summary.total_seconds, 9 + 1e-6)
        self.assertGreater(temporal_prf(summary.intervals(), EVENTS)[2], 0.7)
        self.assertEqual(set(summary.weights), {"motion", "audio"})

    def test_render_and_metadata(self):
        out = self.dir / "out"
        summary = summarize(self.video, prompt="find the action", target_seconds=8, cfg=self.cfg,
                            analysis=self.analysis, out_dir=out)
        rendered = inspect_video(summary.output_path)
        self.assertAlmostEqual(rendered["duration_seconds"], summary.total_seconds, delta=0.5)
        self.assertTrue(rendered["has_audio"])
        saved = json.loads((out / "highlights.json").read_text())
        self.assertEqual(saved["clips"], summary.clips)
        self.assertEqual(saved["evidence_missing"], ["speech"])
        self.assertTrue(any("request could not guide" in note for note in summary.notes))
        for clip in summary.clips:
            self.assertEqual(set(clip["evidence"]), set(summary.weights))

    def test_silent_video_skips_audio_and_reweights(self):
        silent_dir = self.dir / "silent"
        silent_dir.mkdir()
        video = str(make_video(silent_dir, with_audio=False))
        summary = summarize(video, target_seconds=8, cfg=self.cfg, out_dir=silent_dir / "out")
        self.assertNotIn("audio", summary.weights)
        self.assertAlmostEqual(sum(summary.weights.values()), 1.0, places=3)
        self.assertFalse(inspect_video(summary.output_path)["has_audio"])
        self.assertTrue(any("audio" in note.lower() for note in summary.notes))

    def test_analysis_is_cached(self):
        again = analyze(self.video, self.cfg)
        self.assertEqual(again.cuts, self.analysis.cuts)
        self.assertEqual(again.timings, self.analysis.timings)

    def test_whole_video_when_budget_exceeds_duration(self):
        summary = summarize(self.video, target_seconds=120, cfg=self.cfg, analysis=self.analysis)
        self.assertEqual(summary.intervals(), [(0.0, 24.0)])

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.video, method="magic", cfg=self.cfg, analysis=self.analysis)


if __name__ == "__main__":
    unittest.main()
