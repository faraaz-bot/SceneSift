"""Render selected clips into one MP4 with FFmpeg, re-encoding for frame-accurate cuts."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .ffmpeg import ffmpeg_exe
from .datatypes import Clip


def render(video_path: str, clips: list[Clip], out_path: str | Path, *, has_audio: bool,
           crf: int = 20, preset: str = "veryfast") -> Path:
    if not clips:
        raise ValueError("No clips to render.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chains, inputs = [], []
    for index, clip in enumerate(clips):
        chains.append(f"[0:v]trim=start={clip.start:.3f}:end={clip.end:.3f},setpts=PTS-STARTPTS[v{index}]")
        inputs.append(f"[v{index}]")
        if has_audio:
            chains.append(f"[0:a]atrim=start={clip.start:.3f}:end={clip.end:.3f},asetpts=PTS-STARTPTS[a{index}]")
            inputs.append(f"[a{index}]")
    outputs = "[vcat][a]" if has_audio else "[vcat]"
    chains.append(f"{''.join(inputs)}concat=n={len(clips)}:v=1:a={1 if has_audio else 0}{outputs}")
    chains.append("[vcat]scale=trunc(iw/2)*2:trunc(ih/2)*2[v]")  # yuv420p needs even dimensions

    command = [ffmpeg_exe(), "-y", "-v", "error", "-i", video_path, "-filter_complex", ";".join(chains),
               "-map", "[v]"]
    if has_audio:
        command += ["-map", "[a]", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to render the highlight reel: {completed.stderr.strip()[-600:]}")
    return out_path


def make_proxy(video_path: str, out_path: str | Path, height: int = 360) -> Path:
    """Small video-only H.264 copy on the same timeline, for codecs OpenCV cannot decode."""
    out_path = Path(out_path)
    command = [ffmpeg_exe(), "-y", "-v", "error", "-i", video_path, "-an", "-vf", f"scale=-2:{height}",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", str(out_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"The video could not be decoded: {completed.stderr.strip()[-400:]}")
    return out_path
