"""Shared data types. All times are seconds on the original video's timeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    source: str = "window"  # "shot" or "window"

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str


@dataclass
class Clip:
    start: float
    end: float
    score: float
    members: list[int] = field(default_factory=list)  # indices of the candidates it covers

    @property
    def length(self) -> float:
        return self.end - self.start
