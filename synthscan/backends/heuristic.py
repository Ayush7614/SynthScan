"""A lightweight, dependency-free statistical detector.

IMPORTANT: This backend is a *baseline for development, tests, and demos*. It is
NOT production-accurate (real accuracy requires deep learning - see the
``binoculars`` backend). It combines a few well-known writing statistics to give
a plausible AI-likelihood signal with zero heavy dependencies, so the project can
run everywhere and be tested offline.

Heuristics used:
  - Lexical diversity: unique tokens / total tokens. High lexical variety tends
    to mark human writing; low variety can indicate repetitive machine text.
  - Burstiness: coefficient of variation of sentence lengths. Human writing is
    "bursty" (varies a lot); LLM text is comparatively uniform.
  - AI-filler vocabulary: proportion of common LLM stock phrases/filler words.

These are *weak* signals and are only used to exercise the pipeline. Use the
``binoculars`` backend (or a community-trained model) for real decisions.
"""

import math
import re
from statistics import pstdev
from typing import List

from synthscan.core.detector import register_backend, verdict_for
from synthscan.core.result import ScanResult, SegmentResult
from synthscan.core.segmenter import chunk_long

# Common LLM "filler" phrases / words that are overrepresented in AI writing.
_AI_FILLERS = {
    "delve", "delve into", "moreover", "furthermore", "additionally",
    "in conclusion", "in summary", "it's important to note", "crucial",
    "elevate", "leverage", "seamless", "robust", "comprehensive",
    "landscape", "tapestry", "testament", "underscore", "pivotal",
    "foster", "navigate", "holistic", "paradigm", "synergy",
}
_FILLER_RE = re.compile(
    r"\b(" + "|".join(sorted(_AI_FILLERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _lexical_diversity(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _burstiness(sentence_lengths: List[int]) -> float:
    """Coefficient of variation of sentence lengths (stddev / mean)."""
    non_empty = [n for n in sentence_lengths if n > 0]
    if len(non_empty) < 2:
        return 0.0
    mean = sum(non_empty) / len(non_empty)
    if mean == 0:
        return 0.0
    return pstdev(non_empty) / mean


def _filler_ratio(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    matches = _FILLER_RE.findall(text)
    return min(1.0, len(matches) / max(1, len(tokens)) * 10.0)


def _score_text(text: str) -> float:
    """Return a raw ai_probability in [0, 1] using the statistical heuristics.

    Human writing trends: high diversity, high burstiness, low filler.
    We combine the three signals and normalize to a 0..1 probability.
    """
    sentences = chunk_long(text, min_segment_len=20)
    lengths = [len(s) for s, _, _ in sentences]
    if not lengths:
        return 0.5

    diversity = _lexical_diversity(text)
    burstiness = _burstiness(lengths)
    filler = _filler_ratio(text)

    # Human bonus from diversity & burstiness (higher => more human => lower AI).
    human_like = (0.55 * diversity) + (0.25 * min(1.0, burstiness))
    # AI bonus from filler ratio.
    ai_like = 0.5 * filler

    raw = max(0.0, min(1.0, 1.0 - human_like + ai_like))

    # Bias very short texts toward a neutral signal (insufficient evidence).
    if len(_tokens(text)) < 15:
        raw = 0.5 * (raw + 0.5)
    return raw


class HeuristicDetector:
    """Statistical baseline detector - see module docstring for caveats."""

    name = "heuristic"

    def detect_text(self, text: str, **kwargs) -> ScanResult:
        text = text.strip()
        if not text:
            return ScanResult(
                text="",
                ai_probability=0.5,
                verdict=verdict_for(0.5),
                segments=[],
                backend=self.name,
            )

        prob = _score_text(text)
        verdict = verdict_for(prob)

        segments = []
        for seg_text, start, end in chunk_long(text, min_segment_len=20):
            seg_prob = _score_text(seg_text)
            segments.append(
                SegmentResult(
                    text=seg_text,
                    start=start,
                    end=end,
                    ai_probability=seg_prob,
                )
            )

        return ScanResult(
            text=text,
            ai_probability=prob,
            verdict=verdict,
            segments=segments,
            backend=self.name,
        )


register_backend("heuristic", HeuristicDetector)
