"""Pipeline configuration. Defaults are untuned starting points, not fitted values."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

FEATURES = ("visual", "motion", "audio", "speech", "novelty")

# Evidence each compared method may use (docs/EXPERIMENT.md section 7).
METHOD_FEATURES = {
    "activity": ("motion", "audio"),
    "semantic": ("visual",),
    "multimodal": FEATURES,
}
# Only the proposed method uses the diversity constraint unless overridden.
METHOD_DIVERSITY = {"activity": False, "semantic": False, "multimodal": True}


def _default_weights() -> dict[str, float]:
    return {"visual": 0.30, "motion": 0.20, "audio": 0.15, "speech": 0.20, "novelty": 0.15}


@dataclass
class Config:
    # Sampling
    motion_fps: float = 4.0
    visual_fps: float = 1.0
    analysis_width: int = 160
    # Candidate segments
    shot_backend: str = "auto"  # auto | scenedetect | histogram
    window_seconds: list[float] = field(default_factory=lambda: [4.0, 8.0])
    window_overlap: float = 0.5
    min_segment_seconds: float = 2.0
    max_segment_seconds: float = 12.0
    snap_seconds: float = 0.75
    blank_frame_std: float = 6.0  # grey-level spread below which a frame counts as blank
    max_blank_fraction: float = 0.5  # candidates with a larger share of blank frames are dropped
    # Models
    visual_backend: str = "auto"  # auto | clip | histogram
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    speech_backend: str = "auto"  # auto | whisper | none
    whisper_model: str = "base"
    whisper_vad: bool = True  # drops non-speech audio; can miss speech under loud music
    text_backend: str = "auto"  # auto | sentence-transformers | none
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"  # auto | cpu | cuda
    # Fusion
    weights: dict[str, float] = field(default_factory=_default_weights)
    prompt_top_k: int = 3
    context_window_seconds: float = 60.0
    # Selection
    mmr_lambda: float = 0.7
    max_overlap: float = 0.25
    context_before: float = 1.0
    context_after: float = 1.0
    # Export
    crf: int = 20
    preset: str = "veryfast"
    # Caching of prompt-independent analysis
    cache_dir: str = "~/.cache/scenesift"
    use_cache: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(FEATURES)
        if unknown:
            raise ValueError(f"Unknown feature weights: {sorted(unknown)}")
        if not 0 <= self.window_overlap < 1:
            raise ValueError("window_overlap must be in [0, 1).")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be in [0, 1].")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        merged = dict(data)
        if "weights" in merged:
            merged["weights"] = {**_default_weights(), **merged["weights"]}
        return cls(**merged)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        import yaml

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})
