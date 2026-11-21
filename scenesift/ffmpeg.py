"""Locate an FFmpeg executable: the system one, else the binary shipped by imageio-ffmpeg."""
from __future__ import annotations

import shutil


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg, or run `pip install imageio-ffmpeg`."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()
