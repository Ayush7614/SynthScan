#!/usr/bin/env python3
"""Benchmark / sanity-check the lightweight RoBERTa backend with the real model.

    pip install "synthscan[ml]"
    python scripts/benchmark_roberta.py

This downloads the small model (~500 MB) on first run and prints confidence on a
fixed human vs. AI-style example set. It runs on plain CPU with a few GB of RAM.
"""

import time

from synthscan.backends.roberta import RobertaDetector

SAMPLES = [
    ("human", "I walked down to the cafe on the corner and ordered a coffee. "
              "The barista remembered my name, which was a small but nice "
              "surprise. I sat by the window and watched the rain."),
    ("human", "Cooking dinner is usually a mess in our house. Someone burns "
              "the toast, the onions make everyone cry, and the dog begs "
              "under the table for scraps. It is chaotic, but I love it."),
    ("ai", "It is crucial to emphasize the importance of leveraging a holistic "
           "strategy to navigate the intricate landscape of modern challenges, "
           "fostering synergy and underscoring a comprehensive paradigm shift."),
    ("ai", "Furthermore, it is important to note that a robust and seamless "
           "approach can elevate the overall experience, delivering a "
           "comprehensive solution that is both transformative and pivotal."),
]


def main() -> int:
    print("Loading roberta-base-openai-detector (first run downloads ~500MB)...")
    start = time.perf_counter()
    detector = RobertaDetector()
    detector._load()  # noqa: SLF001 - trigger model load up front
    print(f"Loaded in {time.perf_counter() - start:.1f}s\n")

    for expected, text in SAMPLES:
        r = detector.detect_text(text)
        match = "OK" if (expected == "human" and not r.is_ai) or \
                        (expected == "ai" and r.is_ai) else "MISMATCH"
        print(f"[{match:>8}] expected={expected:<6} got={r.verdict.value:<14} "
              f"AI={r.ai_probability:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
