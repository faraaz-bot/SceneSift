"""Audio evidence: loudness and spectral change. These measure activity, not importance."""
from __future__ import annotations

import subprocess

import numpy as np
from scipy import signal

from ..ffmpeg import ffmpeg_exe

SAMPLE_RATE = 16000


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Mono float32 samples, or None when the file has no decodable audio."""
    command = [ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sample_rate),
               "-f", "s16le", "-"]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or len(completed.stdout) < 2:
        return None
    usable = len(completed.stdout) - len(completed.stdout) % 2
    return np.frombuffer(completed.stdout[:usable], dtype=np.int16).astype(np.float32) / 32768.0


def audio_signals(samples: np.ndarray, sample_rate: int = SAMPLE_RATE, hop_seconds: float = 0.25):
    """Return (times, energy_db, flux) on a `hop_seconds` grid, or None for silent or too-short audio."""
    if samples.size < sample_rate // 2 or float(np.max(np.abs(samples))) < 1e-4:
        return None
    _, frame_times, spectrum = signal.stft(samples, fs=sample_rate, nperseg=1024, noverlap=512,
                                           boundary=None, padded=False)
    magnitude = np.abs(spectrum)
    power = (magnitude ** 2).mean(axis=0)
    flux = np.concatenate([[0.0], np.maximum(magnitude[:, 1:] - magnitude[:, :-1], 0.0).sum(axis=0)])

    bins = np.floor(frame_times / hop_seconds).astype(int)
    count = int(bins.max()) + 1
    weights = np.bincount(bins, minlength=count).astype(float)
    filled = weights > 0
    mean_power = np.bincount(bins, weights=power, minlength=count)[filled] / weights[filled]
    mean_flux = np.bincount(bins, weights=flux, minlength=count)[filled] / weights[filled]
    times = (np.arange(count)[filled] + 0.5) * hop_seconds
    return times, 10.0 * np.log10(mean_power + 1e-12), np.log1p(mean_flux)
