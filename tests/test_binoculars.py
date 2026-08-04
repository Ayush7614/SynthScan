"""Tests for the Binoculars backend.

These validate our *integration logic* (probability mapping, verdicts, segment
generation, graceful errors) using a mocked Binoculars, so they run everywhere
in CI without the huge Falcon-7B models.

A real end-to-end test requires the actual models (~14GB+, >=32GB RAM or a GPU).
It is opt-in via the ``SYNTHSCAN_RUN_HEAVY`` env var or the ``heavyml`` marker.
"""

import os

import pytest

from synthscan.backends.binoculars import BinocularsDetector


class _FakeBino:
    """Mimics the Binoculars class interface (compute_score / predict)."""

    def __init__(self, score: float, label: str):
        self.score = score
        self.label = label

    def compute_score(self, text):  # noqa: ARG002
        return self.score

    def predict(self, text):  # noqa: ARG002
        return self.label


def _make_detector(score: float, label: str) -> BinocularsDetector:
    detector = BinocularsDetector()
    detector._load = lambda: _FakeBino(score, label)  # noqa: SLF001
    return detector


def test_high_score_is_ai():
    # Binoculars: high score + AI label -> AI-generated.
    detector = _make_detector(0.8, "Most likely AI-Generated")
    result = detector.detect_text("This is a sentence. And another one here.")
    assert result.is_ai is True
    assert result.verdict.value == "AI-generated"
    assert 0.5 <= result.ai_probability <= 1.0


def test_low_score_with_human_label_is_human():
    detector = _make_detector(0.2, "Most likely Human-Written")
    result = detector.detect_text("I walked to the store and bought some milk. It started to rain.")
    assert result.is_ai is False
    assert result.verdict.value == "Human-written"
    assert result.ai_probability < 0.5


def test_label_overrides_ambiguous_score():
    # If score < 0.5 but label says AI, probability gets pushed >= 0.5.
    detector = _make_detector(0.3, "Most likely AI-Generated")
    result = detector.detect_text("Some text here to check the logic. And more text.")
    assert result.is_ai is True


def test_backend_name_and_segments():
    detector = _make_detector(0.9, "Most likely AI-Generated")
    result = detector.detect_text("First sentence here. Second sentence here too.")
    assert result.backend == "binoculars"
    assert isinstance(result.segments, list)


def test_helpful_error_when_missing(monkeypatch):
    """When the real 'binoculars' module can't be imported, we must raise a
    clear message telling the user how to install it (not a cryptic error)."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "binoculars" or name.startswith("binoculars"):
            raise ImportError("No module named 'binoculars'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    detector = BinocularsDetector()
    with pytest.raises(ImportError) as excinfo:
        detector._load()  # noqa: SLF001 - triggers lazy import
    assert "synthscan[ml]" in str(excinfo.value)


@pytest.mark.heavyml
@pytest.mark.skipif(
    not os.environ.get("SYNTHSCAN_RUN_HEAVY"),
    reason="Set SYNTHSCAN_RUN_HEAVY=1 to run real Binoculars (needs models + RAM/GPU).",
)
def test_real_binoculars_end_to_end():
    """Opt-in real run - only on hardware that can host the Falcon-7B models."""
    detector = BinocularsDetector()
    result = detector.detect_text(
        "The quick brown fox jumps over the lazy dog. It is a classic pangram."
    )
    assert 0.0 <= result.ai_probability <= 1.0
