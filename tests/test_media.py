"""Unit tests for the UI's independent video-inspection helper."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, SimpleNamespace
from scenesift.media import format_duration, inspect_video

class MediaTests(unittest.TestCase):
    def test_duration(self):
        self.assertEqual(format_duration(60), "1:00")
        self.assertEqual(format_duration(3661), "1:01:01")

    def test_missing_video(self):
        with self.assertRaises(FileNotFoundError):
            inspect_video("/unlikely/to/exist.mp4")

    def test_inspect_video_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "test.mp4"
            path.write_bytes(b"placeholder")
            payload = {"format":{"duration":"610.5"},"streams":[
                {"codec_type":"video","width":1920,"height":1080,"avg_frame_rate":"30000/1001"},
                {"codec_type":"audio"}]}
            mocked = SimpleNamespace(returncode=0, stdout=json.dumps(payload))
            with patch("scenesift.media.subprocess.run", return_value=mocked) as run:
                found = inspect_video(str(path))
            self.assertAlmostEqual(found["duration_seconds"], 610.5)
            self.assertTrue(found["has_audio"])
            self.assertAlmostEqual(found["fps"], 29.97, places=2)
            self.assertEqual(run.call_args.kwargs["timeout"], 30)

if __name__ == "__main__":
    unittest.main()
