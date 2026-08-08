"""Production-grade zero-shot detection via the open-source Binoculars method.

Binoculars ("Spotting LLMs With Binoculars", ICML 2024) is a zero-shot,
domain-agnostic detector that needs no training data for the target text. It is
open source (BSD-3-Clause).

This backend lazily loads Binoculars (which pulls Falcon-7B-class models) so the
core package installs and runs without any ML dependencies; the heavy pieces are
only fetched when this backend is actually used.

Add it with:

    pip install "synthscan[ml]"

or manually:  pip install torch transformers binoculars bitsandbytes

Per Binoculars convention, a *lower* binoculars score is more AI-like, but its
``predict`` returns a direct label. We rely on the label for the verdict and map
the raw score to a monotonic ai_probability in [0,1] where higher = more AI.

**Quantization for 16GB free GPUs.** The stock Binoculars loads both Falcon-7B
models in bfloat16 (~28GB VRAM), which does not fit a T4/P100. Pass
``quantization="4bit"`` (or ``"8bit"``) to load them with bitsandbytes instead
(~4GB / ~9GB total) so the detector runs on a single free 16GB GPU. This changes
the inference baseline vs. the published bf16 accuracy, so benchmark reports are
labelled to reflect it.

Note: upstream Binoculars auto-detects the device itself (``cuda:0`` when a GPU
is present, else ``cpu``); its constructor has no ``device`` kwarg, so we do not
forward one to it.
"""

from typing import Any, Optional

from synthscan.core.detector import register_backend, verdict_for
from synthscan.core.result import ScanResult, SegmentResult
from synthscan.core.segmenter import chunk_long

_MIN_SEGMENT_CHARS = 40


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


_QUANT_VALUES = ("", "none", "4bit", "8bit")


def _norm_quant(quantization: Optional[str]) -> str:
    q = (quantization or "").strip().lower()
    if q == "none":
        q = ""
    if q not in _QUANT_VALUES:
        choices = ", ".join(repr(_) for _ in _QUANT_VALUES if _)
        raise ValueError(
            f"Invalid quantization {quantization!r}. Choices: {choices} or None."
        )
    return q


class _QuantizedBinoculars:
    """Binoculars scoring with bitsandbytes-quantized Falcon-7B weights.

    Mirrors the algorithm and thresholds of ahans30/Binoculars (imported from the
    installed upstream package) but loads both observer/performer models with
    bitsandbytes quantization so the pair fits on a single 16GB GPU (T4/P100).
    Exposes the same ``compute_score`` / ``predict`` interface as the stock
    ``Binoculars`` class, so it slots straight into ``BinocularsDetector``.
    """

    def __init__(
        self,
        quant_type: str = "4bit",
        max_token_observed: int = 512,
        mode: str = "low-fpr",
    ) -> None:
        import torch  # deferred: only when the ml extra is present
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        # Reuse the upstream algorithm internals (thresholds + scoring) so our
        # quantized run is the same method, just lower-precision weights.
        from binoculars.detector import (
            BINOCULARS_ACCURACY_THRESHOLD,
            BINOCULARS_FPR_THRESHOLD,
        )
        from binoculars.metrics import entropy, perplexity

        quant_type = _norm_quant(quant_type)
        if quant_type == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8bit
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
        )

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b", **load_kwargs
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b-instruct", **load_kwargs
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed
        self._perplexity = perplexity
        self._entropy = entropy
        self.threshold = (
            BINOCULARS_FPR_THRESHOLD
            if mode == "low-fpr"
            else BINOCULARS_ACCURACY_THRESHOLD
        )

        # Determine the active device from the loaded (quantized) model.
        if hasattr(self.observer_model, "hf_device_map"):
            self.device = next(
                (d for d in self.observer_model.hf_device_map.values() if d != "cpu"),
                "cuda:0",
            )
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _tokenize(self, batch: list[str]):
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.device)
        return encodings

    def _get_logits(self, encodings):
        import torch

        observer_logits = self.observer_model(**encodings.to(self.device)).logits
        performer_logits = self.performer_model(**encodings.to(self.device)).logits
        if self.device != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = self._perplexity(encodings, performer_logits)
        x_ppl = self._entropy(
            observer_logits, performer_logits, encodings, self.tokenizer.pad_token_id
        )
        scores = ppl / x_ppl
        scores = scores.tolist()
        return scores[0] if isinstance(input_text, str) else scores

    def predict(self, input_text):
        import numpy as np

        scores = np.array(self.compute_score(input_text))
        pred = np.where(
            scores < self.threshold,
            "Most likely AI-generated",
            "Most likely human-generated",
        ).tolist()
        return pred


class BinocularsDetector:
    """Zero-shot AI-text detector powered by Binoculars.

    Requires the optional ``ml`` extra. Raises a helpful ImportError at
    instantiation (not at import time) if Binoculars is unavailable.

    Args:
        device: Ignored by the upstream library (it auto-detects CUDA/CPU). Kept
            for API compatibility and for the report's ``device`` field.
        quantization: One of ``None``, ``"4bit"``, or ``"8bit"``. When set, the
            two Falcon-7B models are loaded with bitsandbytes so the detector
            fits on a 16GB GPU. Default (None) uses the stock bfloat16 load.
        **kwargs: Forwarded to the upstream ``Binoculars`` constructor.
    """

    name = "binoculars"

    def __init__(
        self,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._bino = None
        self._device = device
        self._quantization = _norm_quant(quantization)
        self._kwargs = kwargs

    @property
    def quantization(self) -> str:
        """Normalized quantization mode (``""`` = none)."""
        return self._quantization

    def _load(self):
        if self._bino is not None:
            return self._bino
        try:
            import binoculars  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "The 'binoculars' backend requires 'synthscan[ml]'. "
                "Install with:  pip install 'synthscan[ml]'  "
                "or: pip install torch transformers binoculars"
            ) from exc

        if self._quantization:
            from binoculars import Binoculars as _  # noqa: F401  (sanity import)

            init_kwargs = dict(self._kwargs)
            quant_type = self._quantization
            mode = init_kwargs.pop("mode", "low-fpr")
            max_tok = init_kwargs.pop("max_token_observed", 512)
            if init_kwargs:
                names = ", ".join(sorted(init_kwargs))
                raise ValueError(
                    f"Unsupported kwargs for quantized binoculars: {names}."
                )
            self._bino = _QuantizedBinoculars(
                quant_type=quant_type, mode=mode, max_token_observed=max_tok
            )
        else:
            from binoculars import Binoculars

            init_kwargs = dict(self._kwargs)
            # Upstream auto-detects its own device; never forward a device kwarg.
            init_kwargs.pop("device", None)
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