"""Production-grade zero-shot detection via the open-source Binoculars method.

Binoculars ("Spotting LLMs With Binoculars", ICML 2024) is a zero-shot,
domain-agnostic detector that needs no training data for the target text. It is
open source (BSD-3-Clause).

This backend lazily loads Binoculars (which pulls Falcon-7B-class models) so the
core package installs and runs without any ML dependencies; the heavy pieces are
only fetched when this backend is actually used.

Add it with:

    pip install "synthscan[ml]"

or manually:  pip install torch transformers binoculars

Per Binoculars convention, a *lower* binoculars score is more AI-like, but its
``predict`` returns a direct label. We rely on the label for the verdict and map
the raw score to a monotonic ai_probability in [0,1] where higher = more AI.
"""

from typing import Any, Optional

from synthscan.core.detector import register_backend, verdict_for
from synthscan.core.result import ScanResult, SegmentResult
from synthscan.core.segmenter import chunk_long

_MIN_SEGMENT_CHARS = 40


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class BinocularsDetector:
    """Zero-shot AI-text detector powered by Binoculars.

    Requires the optional ``ml`` extra. Raises a helpful ImportError at
    instantiation (not at import time) if Binoculars is unavailable.
    """

    name = "binoculars"

    def __init__(self, device: Optional[str] = None, **kwargs: Any) -> None:
        self._bino = None
        self._device = device
        self._kwargs = kwargs

    def _load(self):
        if self._bino is not None:
            return self._bino
        try:
            from binoculars import Binoculars
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "The 'binoculars' backend requires 'synthscan[ml]'. "
                "Install with:  pip install 'synthscan[ml]'  "
                "or: pip install torch transformers binoculars"
            ) from exc
        init_kwargs = dict(self._kwargs)
        if self._device:
            init_kwargs["device"] = self._device
        self._bino = Binoculars(**init_kwargs)
        return self._bino

    def _score(self, text: str):
        """Return (ai_probability, is_ai_label) for a chunk of text."""
        bino = self._load()
        raw_score = bino.compute_score(text)
        label = bino.predict(text)
        is_ai = "ai" in label.lower()
        # Map the raw score to a monotonic probability: higher score => more AI.
        probability = _clamp(raw_score)
        # When the label disagrees, let the label dominate the decision.
        if is_ai and probability < 0.5:
            probability = 0.5 + probability * 0.5
        if not is_ai and probability > 0.5:
            probability = probability * 0.5
        return probability, is_ai

    def detect_text(self, text: str, **kwargs) -> ScanResult:
        text = text.strip()
        if not text:
            return ScanResult(
                text="", ai_probability=0.5, verdict=verdict_for(0.5),
                segments=[], backend=self.name,
            )

        prob, is_ai = self._score(text)
        segments = []
        for seg_text, start, end in chunk_long(text, min_segment_len=_MIN_SEGMENT_CHARS):
            seg_prob, _ = self._score(seg_text)
            segments.append(
                SegmentResult(
                    text=seg_text, start=start, end=end, ai_probability=seg_prob
                )
            )

        return ScanResult(
            text=text,
            ai_probability=prob,
            verdict=verdict_for(prob),
            segments=segments,
            backend=self.name,
        )


register_backend("binoculars", BinocularsDetector)
