"""Generate a small synthetic video with known 'events' for end-to-end tests."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from scenesift.ffmpeg import ffmpeg_exe

# (start, end, BGR background, has_motion_and_sound)
SCENES = [
    (0.0, 8.0, (120, 60, 20), False),
    (8.0, 12.0, (30, 30, 200), True),
    (12.0, 18.0, (40, 140, 40), False),
    (18.0, 21.0, (30, 200, 220), True),
    (21.0, 24.0, (110, 110, 110), False),
]
EVENTS = [(start, end) for start, end, _, active in SCENES if active]
DURATION = SCENES[-1][1]
FPS = 12
SAMPLE_RATE = 16000


def make_video(directory: Path, with_audio: bool = True) -> Path:
    """Write the synthetic clip and return its path."""
    rng = np.random.default_rng(0)
    silent = directory / "silent.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the synthetic video.")
    for index in range(int(DURATION * FPS)):
        moment = index / FPS
        start, _, colour, active = next(s for s in SCENES if s[0] <= moment < s[1])
        frame = np.full((120, 160, 3), colour, dtype=np.uint8)
        frame = np.clip(frame + rng.integers(-4, 5, frame.shape), 0, 255).astype(np.uint8)
        if active:
            phase = moment - start
            for k in range(3):
                x = int(20 + (phase * 90 + k * 45) % 120)
                y = int(60 + 40 * np.sin(phase * 6 + k * 2.1))
                cv2.circle(frame, (x, y), 12, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    if not with_audio:
        return silent

    time = np.arange(int(DURATION * SAMPLE_RATE)) / SAMPLE_RATE
    audio = 0.01 * rng.standard_normal(time.size)
    for start, end in EVENTS:
        inside = (time >= start) & (time < end)
        burst = 0.5 * np.sin(2 * np.pi * 880 * time) * (np.sin(2 * np.pi * 4 * time) > 0)
        audio[inside] += burst[inside]
    wav = directory / "audio.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    out = directory / "synthetic.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(silent), "-i", str(wav), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out
