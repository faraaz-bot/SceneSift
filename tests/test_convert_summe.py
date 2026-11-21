"""SumMe conversion: frame selections become per-annotator intervals in seconds."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from convert_summe import convert, user_intervals  # noqa: E402

from scenesift.evaluation import load_annotation


class ConvertSummeTests(unittest.TestCase):
    def test_runs_of_selected_frames_become_intervals(self):
        selected = np.array([0, 3, 3, 0, 0, 7, 7, 7, 0, 1])
        self.assertEqual(user_intervals(selected, 10.0), [[0.1, 0.3], [0.5, 0.8], [0.9, 1.0]])
        self.assertEqual(user_intervals(np.zeros(5), 10.0), [])

    def test_convert_writes_annotations_and_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "summe" / "GT").mkdir(parents=True)
            (root / "summe" / "videos").mkdir()
            scores = np.zeros((100, 2), dtype=np.uint8)
            scores[10:30, 0] = 1
            scores[50:60, 1] = 2
            for name in ("Has Video", "No_Video"):
                scipy.io.savemat(root / "summe" / "GT" / f"{name}.mat",
                                 {"user_score": scores, "nFrames": 100, "video_duration": 4.0, "FPS": 25})
            (root / "summe" / "videos" / "Has Video.mp4").write_bytes(b"placeholder")
            self.assertEqual(convert(root / "summe", root / "out"), 1)
            with open(root / "out" / "manifest.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["source_id"] for r in rows], ["Has_Video"])
            annotation = load_annotation(rows[0]["annotation"])
            self.assertEqual(annotation.annotators, {"user_01": [(0.4, 1.2)], "user_02": [(2.0, 2.4)]})
            self.assertIsNone(json.loads(Path(rows[0]["annotation"]).read_text())["prompt"])


if __name__ == "__main__":
    unittest.main()
