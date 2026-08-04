"""Tests for the SynthScan scanning pipeline (heuristic backend - no heavy deps)."""

import pytest

from synthscan.core.detector import get_backend, list_backends, scan_text
from synthscan.core.result import Verdict
from synthscan.core.segmenter import split_sentences


def test_list_backends_contains_core():
    backends = list_backends()
    assert "heuristic" in backends
    assert "binoculars" in backends


def test_scan_returns_result_structure():
    result = scan_text("The quick brown fox jumps over the lazy dog.", backend="heuristic")
    assert result.backend == "heuristic"
    assert 0.0 <= result.ai_probability <= 1.0
    assert isinstance(result.verdict, Verdict)
    # Every top-level result includes per-segment data.
    assert isinstance(result.segments, list)


def test_scan_empty_text_is_neutral():
    result = scan_text("   ", backend="heuristic")
    assert result.ai_probability == 0.5
    assert result.segments == []


def test_version_range_is_valid():
    result = get_backend("heuristic").detect_text("Some ordinary human sentence here.")
    assert 0.0 <= result.ai_probability <= 1.0


def test_split_sentences_basic():
    text = "Hello there. This is a second sentence! And a third?"
    sentences = split_sentences(text)
    joined = " ".join(s for s, _, _ in sentences)
    assert "Hello" in joined
    assert "third" in joined
    assert len(sentences) == 3


def test_split_sentences_offsets_are_contiguous():
    text = "First sentence. Second sentence. Third sentence."
    sentences = split_sentences(text)
    for s, start, end in sentences:
        assert text[start:end] == s
        assert start < end


def test_unknown_backend_raises():
    from synthscan.core.detector import scan_text as st

    with pytest.raises(ValueError):
        st("hello", backend="does-not-exist")


def test_segments_cover_no_gaps_for_plain_text():
    text = "One short sentence. Another short sentence here. And one more here too."
    result = scan_text(text, backend="heuristic")
    if result.segments:
        total = sum(len(seg.text) for seg in result.segments)
        assert total <= len(text)
