"""Frame embeddings. CLIP gives language-grounded semantics; the histogram encoder is a degraded fallback."""
from __future__ import annotations

import cv2
import numpy as np

from ..config import Config


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class ClipEncoder:
    name = "clip"
    supports_text = True

    def __init__(self, model: str, pretrained: str, device: str) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model)
        self.name = f"clip:{model}/{pretrained}"

    def encode_images(self, frames: list[np.ndarray]) -> np.ndarray:
        """frames: RGB uint8 arrays."""
        from PIL import Image

        batch = self._torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
        with self._torch.no_grad():
            features = self.model.encode_image(batch)
        return _normalize(features.float().cpu().numpy())

    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        with self._torch.no_grad():
            features = self.model.encode_text(tokens)
        return _normalize(features.float().cpu().numpy())


class HistogramEncoder:
    """Colour and coarse-layout descriptor. No language grounding: prompts cannot guide it."""

    name = "histogram"
    supports_text = False

    def encode_images(self, frames: list[np.ndarray]) -> np.ndarray:
        rows = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            hist = hist / max(hist.sum(), 1.0)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            layout = cv2.resize(gray, (6, 6), interpolation=cv2.INTER_AREA).ravel() / 255.0
            rows.append(np.concatenate([np.sqrt(hist), layout / 6.0]))
        return _normalize(np.asarray(rows, dtype=np.float32))

    def encode_text(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("The histogram encoder cannot embed text.")


def resolve_device(cfg: Config) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_visual_encoder(cfg: Config, warnings: list[str]):
    if cfg.visual_backend == "histogram":
        return HistogramEncoder()
    try:
        return ClipEncoder(cfg.clip_model, cfg.clip_pretrained, resolve_device(cfg))
    except (ImportError, OSError, RuntimeError) as exc:
        if cfg.visual_backend == "clip":
            raise
        warnings.append(
            f"CLIP could not be loaded ({type(exc).__name__}: {exc}). Using colour histograms instead: "
            "visual scores are not semantic and the text request cannot guide visual selection."
        )
        return HistogramEncoder()


def prepare_frame(frame_bgr: np.ndarray, short_side: int = 224) -> np.ndarray:
    """Downscale before buffering so batches stay small; returns RGB."""
    height, width = frame_bgr.shape[:2]
    scale = short_side / max(1, min(height, width))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
