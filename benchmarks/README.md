# SynthScan benchmark reports

Real, reproducible accuracy numbers. We publish them because open source only
matters if the detection genuinely works — and because honesty about limits is
the point of the project.

## Latest results

### CPU / RoBERTa baseline (`benchmarks/cpu/`)

`roberta-base-openai-detector` — trained 2021, GPT-2 era.

| Corpus | Accuracy | Human @15 | AI @15 | Verdict |
|--------|----------|-----------|--------|---------|
| [modern-corpus.json](corpus/modern-corpus.json) | **50%** (15/30) | 11/15 | 4/15 | Coin flip on current-LLM text. Catches obvious filler, misses most realistic modern AI output. |

**Interpretation (be honest):** this backend is a *baseline*, not a product
answer. GPT-2-era RoBERTa cannot reliably detect modern LLMs. It is kept because
it is tiny and CPU-friendly, but it is **not** the accuracy path.

### GPU / Binoculars (`benchmarks/bf16/`, `benchmarks/quantized/`)

Placeholders. Run on a >=32GB GPU (bf16) or a free 16GB T4 (quantized):

- Kaggle one-click: `scripts/benchmark_binoculars_kaggle.ipynb`
- CLI: `python scripts/benchmark_binoculars.py --quantize 4bit --report benchmarks/quantized/T4-4bit.json`

Binoculars (ICML 2024, zero-shot) is the real accuracy path; these numbers are
what the README should quote once produced.

## Reproducing

```bash
pip install "synthscan[ml]"
python scripts/benchmark_roberta.py --corpus benchmarks/corpus/modern-corpus.json --report benchmarks/cpu/roberta-modern-corpus.json
```

## Contributing

Add samples to `benchmarks/corpus/modern-corpus.json` (human + AI, realistic
writing, no filler-bombs). We especially want AI samples from *recent* models
so the corpus stays honest about what detectors face today.
