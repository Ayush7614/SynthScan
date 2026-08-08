#!/usr/bin/env python3
"""Benchmark / sanity-check the real Binoculars backend.

Two modes:

1. Stock bf16 (default, most accurate, = published baseline):
   Needs the two Falcon-7B models in bfloat16 (~28GB VRAM, or >=32GB RAM):

       pip install "synthscan[ml]"
       python scripts/benchmark_binoculars.py --report report.json

2. Quantized (fits a free 16GB GPU such as a Colab/Kaggle T4):

       pip install "synthscan[ml]" bitsandbytes
       python scripts/benchmark_binoculars.py --quantize 4bit --report report.json

Note: upstream Binoculars picks its device itself (cuda:0 when present, else
cpu), so the script does NOT pass a device flag into the detector; ``--device``
here is only recorded in the report. The stock bfloat16 path requires >=32GB RAM
/ a large GPU; try ``--quantize 4bit`` on small GPUs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from synthscan.backends.binoculars import BinocularsDetector

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


def run(detector, samples: list[dict], device: str, quantization: str) -> dict:
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
        "backend": "binoculars",
        "model": "Falcon-7B + Falcon-7B-Instruct (zero-shot)",
        "quantization": quantization or "bf16 (stock)",
        "device": device,  # note: upstream auto-selects cuda:0/cpu; recorded here
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "results": results,
    }


def main(device: str | None, quantize: str, report: str | None) -> int:
    print("Loading Binoculars (first run downloads Falcon-7B models, ~14GB)...")
    if quantize:
        print(f"Quantized mode: {quantize} (fits a 16GB GPU)")
    start = time.perf_counter()
    detector = BinocularsDetector(device=device, quantization=quantize or None)
    detector._load()  # noqa: SLF001 - trigger the model load up front
    print(f"Loaded in {time.perf_counter() - start:.1f}s\n")

    report_data = run(detector, SAMPLES, device or "auto", quantize)

    out_path = Path(report or "benchmark-binoculars-report.json")
    out_path.write_text(json.dumps(report_data, indent=2))
    print(f"\nReport written to {out_path}")
    print(f"Accuracy: {report_data['accuracy']:.1%} "
          f"({report_data['correct']}/{report_data['total']})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None,
                        help="recorded device string for the report (auto-detected by upstream)")
    parser.add_argument("--quantize", default="",
                        choices=["", "none", "4bit", "8bit"],
                        help="load Falcon-7B weights with bitsandbytes (4bit/8bit) to fit"
                             " a 16GB GPU (default: stock bfloat16, needs >=32GB RAM)")
    parser.add_argument("--report", default=None, help="Path to write JSON report")
    args = parser.parse_args()
    sys.exit(main(device=args.device, quantize=args.quantize, report=args.report))