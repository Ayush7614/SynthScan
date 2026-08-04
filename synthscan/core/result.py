"""Core types for SynthScan."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Verdict(str, Enum):
    """Human-readable classification of a piece of content."""

    AI_GENERATED = "AI-generated"
    AI_ASSISTED = "AI-assisted"
    HUMAN = "Human-written"


@dataclass
class SegmentResult:
    """A single sentence (or segment) and its AI-likelihood."""

    text: str
    start: int
    end: int
    ai_probability: float  # 0.0 (human) -> 1.0 (AI)


@dataclass
class ScanResult:
    """The result of scanning one piece of text."""

    text: str
    ai_probability: float       # 0.0 (human) -> 1.0 (AI)
    verdict: Verdict
    segments: List[SegmentResult] = field(default_factory=list)
    backend: str = "unknown"

    @property
    def is_ai(self) -> bool:
        """True when the text is likely AI-generated (probability >= 0.5)."""
        return self.ai_probability >= 0.5

    def human_readable(self) -> str:
        """A short human-readable summary."""
        return (
            f"[{self.backend}] {self.verdict.value} "
            f"(AI probability: {self.ai_probability:.2%})"
        )
