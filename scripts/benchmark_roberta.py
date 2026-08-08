#!/usr/bin/env python3
"""Benchmark / sanity-check the lightweight RoBERTa backend with the real model.

    pip install "synthscan[ml]"
    python scripts/benchmark_roberta.py [--corpus benchmarks/corpus/modern-corpus.json] [--report /path/report.json]

This downloads the small model (~500 MB) on first run and prints confidence on a
human vs. AI-style example set. It runs on plain CPU with a few GB of RAM.

Accuracy caveat (be honest): ``roberta-base-openai-detector`` was trained to
catch GPT-2-era outputs, so it is a *baseline* - not current-LLM grade. Real
accuracy for current models needs the Binoculars backend. This script exists to
publish real numbers so we never overclaim.

By default it also writes a machine-readable report to ./benchmark-report.json,
which CI jobs consume to publish real accuracy numbers.
"""

import argparse
import json
import time
from pathlib import Path

from synthscan.backends.roberta import RobertaDetector


def _load_corpus(path: str) -> list[dict]:
    """Load samples from a corpus JSON file (same format as the inline default)."""
    data = json.loads(Path(path).read_text())
    return data["samples"]


SAMPLES = [
    {"expected": "human", "text": "I walked down to the cafe on the corner and ordered a coffee. "
        "The barista remembered my name, which was a small but nice surprise. "
        "I sat by the window and watched the rain."},
    {"expected": "human", "text": "Cooking dinner is usually a mess in our house. Someone burns "
        "the toast, the onions make everyone cry, and the dog begs under the table "
        "for scraps. It is chaotic, but I love it."},
    {"expected": "ai", "text": "It is crucial to emphasize the importance of leveraging a holistic "
        "strategy to navigate the intricate landscape of modern challenges, fostering "
        "synergy and underscoring a comprehensive paradigm shift."},
    {"expected": "ai", "text": "Furthermore, it is important to note that a robust and seamless approach "
        "can elevate the overall experience, delivering a comprehensive solution that "
        "is both transformative and pivotal."},
]


def run(detector, samples: list[dict], corpus: str) -> dict:
    results = []
    correct = 0
    for sample in samples:
        r = detector.detect_text(sample["text"])
        expected = sample["expected"]
        is_human = not r.is_ai
        ok = (expected == "human" and is_human) or (expected == "ai" and r.is_ai)
        correct += int(ok)
        results.append({
            "expected": expected,
            "verdict": r.verdict.value,
            "ai_probability": round(r.ai_probability, 4),
            "ok": ok,
        })
        print(f"[{'OK' if ok else 'MISMATCH':>8}] expected={expected:<6} "
              f"got={r.verdict.value:<14} AI={r.ai_probability:.2f}")
    total = len(samples)
    return {
        "backend": "roberta",
        "model": "roberta-base-openai-detector",
        "corpus": corpus or "(built-in 4-sample sanity set)",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "results": results,
    }


def main(corpus: str | None, report: str | None) -> int:
    print("Loading roberta-base-openai-detector (first run downloads ~500MB)...")
    start = time.perf_counter()
    detector = RobertaDetector()
    detector._load()  # noqa: SLF001 - trigger model load up front
    print(f"Loaded in {time.perf_counter() - start:.1f}s\n")

    samples = _load_corpus(corpus) if corpus else SAMPLES

    report_data = run(detector, samples, corpus)

    out_path = Path(report or "benchmark-report.json")
    out_path.write_text(json.dumps(report_data, indent=2))
    print(f"\nReport written to {out_path}")
    print(f"Accuracy: {report_data['accuracy']:.1%} "
          f"({report_data['correct']}/{report_data['total']})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None,
                        help="Path to corpus JSON (with a 'samples' array). "
                             "Defaults to a tiny 4-sample sanity set.")
    parser.add_argument("--report", default=None, help="Path to write JSON report")
    args = parser.parse_args()
    raise SystemExit(main(args.corpus, args.report))
