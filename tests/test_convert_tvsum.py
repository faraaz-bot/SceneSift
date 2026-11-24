"""TVSum conversion: per-frame scores become shot scores and per-annotator reference intervals."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from convert_tvsum import convert, shot_scores, top_shot_intervals  # noqa: E402

from scenesift.evaluation import load_annotation
from synthetic import DURATION, FPS, make_video


class ConvertTvsumTests(unittest.TestCase):
    def test_frame_scores_collapse_to_one_score_per_shot(self):
        frames = np.repeat([4, 1, 5], 20)  # 10 fps, three 2-second shots
        np.testing.assert_allclose(shot_scores(frames, 10.0), [4, 1, 5])
        np.testing.assert_allclose(shot_scores(np.repeat([2, 3], [20, 5]), 10.0), [2, 3])  # short last shot

    def test_top_shots_fill_the_share_and_merge(self):
        scores = np.array([1, 5, 5, 1, 1, 4, 1, 1, 1, 1], dtype=float)
        self.assertEqual(top_shot_intervals(scores, 20.0, 0.3), [[2.0, 6.0], [10.0, 12.0]])

    def test_ties_prefer_the_stronger_neighbourhood_then_the_earlier_shot(self):
        scores = np.array([3, 1, 1, 1, 1, 1, 3, 5, 1, 1], dtype=float)
        self.assertEqual(top_shot_intervals(scores, 20.0, 0.2), [[12.0, 16.0]])
        self.assertEqual(top_shot_intervals(np.ones(10), 20.0, 0.1), [[0.0, 2.0]])

    def test_convert_writes_annotations_and_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "tvsum" / "data").mkdir(parents=True)
            (root / "tvsum" / "video").mkdir()
            make_video(root / "tvsum" / "video", with_audio=False).rename(root / "tvsum" / "video" / "abc.mp4")
            frames = int(DURATION * FPS)
            rows = [np.repeat([1, 5, 1, 2], frames // 4), np.repeat([2, 2, 4, 1], frames // 4)]
            with open(root / "tvsum" / "data" / "ydata-tvsum50-anno.tsv", "w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                for scores in rows:
                    writer.writerow(["abc", "PK", ",".join(str(v) for v in scores)])
                writer.writerow(["missing", "DS", "1,1,1"])
            self.assertEqual(convert(root / "tvsum", root / "out"), 1)
            with open(root / "out" / "manifest.csv", newline="") as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual([(r["source_id"], r["category"], r["split"]) for r in manifest],
                             [("abc", "parkour", "test")])
            annotation = load_annotation(manifest[0]["annotation"])
            self.assertEqual(annotation.importance_step, 2.0)
            self.assertEqual(annotation.importance["user_01"], [1.0] * 3 + [5.0] * 3 + [1.0] * 3 + [2.0] * 3)
            self.assertEqual(annotation.annotators["user_01"], [(6.0, 8.0)])  # 15% of 24 s is one shot
            self.assertEqual(annotation.annotators["user_02"], [(12.0, 14.0)])


if __name__ == "__main__":
    unittest.main()
