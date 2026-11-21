"""Speech transcription and sentence embeddings for transcript relevance and topic novelty."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..datatypes import Utterance


def load_transcriber(cfg: Config, device: str):
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(cfg.whisper_model, device=device, compute_type=compute_type)


def transcribe(model, samples: np.ndarray, vad: bool = True) -> list[Utterance]:
    """Timestamped utterances; an empty list means no speech was found."""
    segments, _ = model.transcribe(samples, vad_filter=vad)
    utterances = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            utterances.append(Utterance(float(segment.start), float(segment.end), text))
    return utterances


class TextEncoder:
    def __init__(self, model: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.model = SentenceTransformer(model, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)


def topic_novelty(utterances: list[Utterance], embeddings: np.ndarray, window_seconds: float) -> np.ndarray:
    """1 - similarity of each utterance to the mean of what was said in the preceding window."""
    novelty = np.full(len(utterances), np.nan)
    for index, utterance in enumerate(utterances):
        earlier = [j for j in range(index) if utterances[j].end > utterance.start - window_seconds]
        if not earlier:
            continue
        context = embeddings[earlier].mean(axis=0)
        norm = np.linalg.norm(context)
        if norm > 1e-9:
            novelty[index] = 1.0 - float(embeddings[index] @ (context / norm))
    return novelty
