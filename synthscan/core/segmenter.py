"""Sentence / segment splitting utilities.

Used to produce the sentence-level probabilities that make results actionable
(highlight *which* parts look AI-generated).
"""

import re
from typing import List, Tuple

# Splits on sentence-ending punctuation followed by whitespace and a capital
# letter or a digit. Keeps abbreviations like "Dr." reasonably intact because
# they aren't followed by a capital/digit+space pattern that matches.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split ``text`` into (sentence, start_offset, end_offset) tuples.

    ``start_offset``/``end_offset`` are character offsets into the original
    ``text`` so callers can highlight exact spans.
    """
    text = text.strip()
    if not text:
        return []

    sentences: List[Tuple[str, int, int]] = []
    cursor = 0
    for match in _SENTENCE_RE.finditer(text):
        sentence = text[cursor : match.start()]
        sentences.append((sentence, cursor, match.start()))
        cursor = match.end()
    last = text[cursor:]
    if last.strip():
        sentences.append((last, cursor, len(text)))
    return sentences


def chunk_long(text: str, min_segment_len: int = 20) -> List[Tuple[str, int, int]]:
    """Return segments of at least ``min_segment_len`` chars.

    Very short sentences (headers, single words) are merged into the following
    segment so per-sentence probability estimates are more stable.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    merged: List[Tuple[str, int, int]] = []
    current_text, current_start, current_end = sentences[0]
    for seg_text, seg_start, seg_end in sentences[1:]:
        if len(current_text) < min_segment_len:
            # append with a space to keep token boundaries sane
            current_text = f"{current_text} {seg_text}".strip()
            current_end = seg_end
        else:
            merged.append((current_text, current_start, current_end))
            current_text, current_start, current_end = seg_text, seg_start, seg_end
    merged.append((current_text, current_start, current_end))
    return merged
