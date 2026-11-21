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
    except FileNotFoundError:
        return _inspect_without_ffprobe(path)
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

def _inspect_without_ffprobe(path: str) -> dict:
    """Fallback when ffprobe is not installed: OpenCV for the video stream, FFmpeg's banner for audio."""
    import cv2

    from .ffmpeg import ffmpeg_exe

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError("The video could not be opened. Try a supported media file.")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frames <= 0:
        raise ValueError("The source video has no readable duration.")
    try:
        banner = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path],
                                capture_output=True, text=True, timeout=30, check=False).stderr
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Video metadata inspection timed out.") from exc
    has_audio = any("Stream #" in line and "Audio:" in line for line in banner.splitlines())
    return {"duration_seconds": frames / fps, "width": width, "height": height, "fps": fps,
            "has_audio": has_audio}
