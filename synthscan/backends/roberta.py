"""A lightweight, CPU-friendly open-source text detector.

Uses a small trained classifier (``roberta-base-openai-detector``, ~500 MB) via
the HuggingFace transformers pipeline. It runs comfortably on CPU with a few GB
of RAM - no GPU or 7B-parameter models required - which makes it the sensible
**default production** backend for most users.

Accuracy caveats (be honest):
  - This model was originally trained to detect GPT-2-era outputs, so it is
    *strong*, but not as current as the heavy Falcon-based Binoculars detector.
  - Use Binoculars when you need maximum accuracy and have >=32GB RAM / a GPU.
  - We keep the model *lazy-loaded* so the core package stays dependency-free.

Labels from the model: "Real" (human) and "Fake" (AI-generated).
"""

from typing import Any, Optional

from synthscan.core.detector import register_backend, verdict_for
from synthscan.core.result import ScanResult, SegmentResult
from synthscan.core.segmenter import chunk_long

_MIN_SEGMENT_CHARS = 40
_MODEL = "roberta-base-openai-detector"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class RobertaDetector:
    """CPU-friendly AI-text detector powered by a small RoBERTa classifier.

    Requires ``transformers`` (part of the ``[ml]`` extra). Lazy-loads the model
    on first use so importing SynthScan stays lightweight.
    """

    name = "roberta"

    def __init__(self, model: str = _MODEL, device: Optional[int] = None, **kwargs: Any) -> None:
        self._pipeline = None
        self._model = model
        self._device = device
        self._kwargs = kwargs

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "The 'roberta' backend requires 'synthscan[ml]'. "
                "Install with:  pip install 'synthscan[ml]'  "
                "(or: pip install torch transformers)"
            ) from exc
        init_kwargs = dict(self._kwargs)
        if self._device is not None:
            init_kwargs["device"] = self._device
        self._pipeline = pipeline("text-classification", model=self._model, **init_kwargs)
        return self._pipeline

    def _score(self, text: str) -> float:
        """Return ai_probability in [0,1] for one chunk (higher = more AI)."""
        pipe = self._load()
        outputs = pipe(text[:510])  # keep within typical max length for the model
        if not outputs:
            return 0.5
        top = outputs[0]
        label = str(top.get("label", ""))
        score = float(top.get("score", 0.5))
        is_ai = "fake" in label.lower() or "ai" in label.lower()
        if is_ai:
            # 'Fake'/'AI' class score is already high => maps directly.
            prob = _clamp(score)
        else:
            # 'Real' class: high score => human => low AI probability.
            prob = _clamp(1.0 - score)
        return prob

    def detect_text(self, text: str, **kwargs) -> ScanResult:
        text = text.strip()
        if not text:
            return ScanResult(
                text="", ai_probability=0.5, verdict=verdict_for(0.5),
                segments=[], backend=self.name,
            )
        prob = self._score(text)
        segments = []
        for seg_text, start, end in chunk_long(text, min_segment_len=_MIN_SEGMENT_CHARS):
            seg_prob = self._score(seg_text)
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


register_backend("roberta", RobertaDetector)
