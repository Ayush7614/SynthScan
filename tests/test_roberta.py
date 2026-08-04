"""Tests for the lightweight RoBERTa backend (mocked, runs everywhere in CI)."""

import pytest

from synthscan.backends.roberta import RobertaDetector


class _FakePipeline:
    """Mimics the transformers text-classification pipeline output."""

    def __init__(self, label: str, score: float):
        self.responses = [{"label": label, "score": score}]

    def __call__(self, text, *args, **kwargs):  # noqa: ARG002
        return self.responses


def _make_detector(label: str, score: float) -> RobertaDetector:
    detector = RobertaDetector()
    detector._load = lambda: _FakePipeline(label, score)  # noqa: SLF001
    return detector


def test_fake_label_high_score_is_ai():
    detector = _make_detector("Fake", 0.93)
    result = detector.detect_text(
        "It is crucial to emphasize that this is a machine-generated paragraph here."
    )
    assert result.is_ai is True
    assert result.verdict.value == "AI-generated"
    assert result.ai_probability >= 0.5


def test_real_label_high_score_is_human():
    detector = _make_detector("Real", 0.91)
    result = detector.detect_text(
        "I walked to the store and bought some milk. It started to rain on the way home."
    )
    assert result.is_ai is False
    assert result.verdict.value == "Human-written"
    assert result.ai_probability < 0.5


def test_backend_name_and_segments():
    detector = _make_detector("Real", 0.8)
    result = detector.detect_text("First sentence here. Second sentence here too.")
    assert result.backend == "roberta"
    assert isinstance(result.segments, list)


def test_helpful_error_when_transformers_missing(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    detector = RobertaDetector()
    with pytest.raises(ImportError) as excinfo:
        detector._load()  # noqa: SLF001
    assert "synthscan[ml]" in str(excinfo.value)


@pytest.mark.heavyml
@pytest.mark.skipif(True, reason="Real model download is validated manually via scripts/benchmark_roberta.py")
def test_real_roberta_end_to_end():
    """(Placeholder) - use scripts/benchmark_roberta.py to run the real model."""
