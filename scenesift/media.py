"""Safe, read-only media metadata inspection using ffprobe."""
from __future__ import annotations
import json
import math
import os
import subprocess
from fractions import Fraction

def format_duration(seconds: float | int) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"

def inspect_video(path: str) -> dict:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("The uploaded file is no longer available.")
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
            capture_output=True, text=True, timeout=30, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe was not found. Install FFmpeg and restart the app.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Video metadata inspection timed out.") from exc
    if completed.returncode != 0:
        raise ValueError("FFmpeg could not read the video. Try a supported media file.")
    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        visual = next((s for s in streams if s.get("codec_type") == "video"), None)
        if visual is None:
            raise ValueError("No video stream was found.")
        seconds = float(payload.get("format", {}).get("duration") or visual.get("duration") or 0)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("The source video has no readable duration.")
        fps_str = visual.get("avg_frame_rate") or visual.get("r_frame_rate") or "0/1"
        fps = float(Fraction(fps_str))
        return {
            "duration_seconds": seconds,
            "width": int(visual.get("width") or 0),
            "height": int(visual.get("height") or 0),
            "fps": fps if math.isfinite(fps) else 0.0,
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        }
    except (KeyError, TypeError, ZeroDivisionError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed video metadata.") from exc
