"""Detector protocol, backend registry, and the top-level scan_text() entry point.

Backends conform to the ``Detector`` protocol so new detection strategies
(deep-learning, multimodal image, code, etc.) can be added without changing the
public API.
"""

from typing import Dict, List, Protocol, Type

from synthscan.core.result import ScanResult, Verdict

# The canonical verdict lookup, given a probability in [0, 1].
def verdict_for(ai_probability: float) -> Verdict:
    """Map a raw probability to a human verdict.

    Boundaries:
      - >= 0.7 : AI-generated
      - >= 0.4 : AI-assisted (a mixed signal - investigate further)
      - else   : Human-written
    """
    if ai_probability >= 0.7:
        return Verdict.AI_GENERATED
    if ai_probability >= 0.4:
        return Verdict.AI_ASSISTED
    return Verdict.HUMAN


class Detector(Protocol):
    """Interface every detection backend must implement."""

    name: str

    def detect_text(self, text: str, **kwargs) -> ScanResult:
        """Return a :class:`ScanResult` for ``text``.

        Implementations MUST label their results with the distance/confidence
        as an *ai_probability* in [0, 1], and SHOULD populate per-segment
        probabilities.
        """
        ...


_registry: Dict[str, Type[Detector]] = {}


def register_backend(name: str, backend_cls: Type[Detector]) -> None:
    """Register a detector backend class under ``name``."""
    _registry[name] = backend_cls


def list_backends() -> List[str]:
    """Return all registered backend names."""
    return sorted(_registry)


def get_backend(name: str = "heuristic", **kwargs) -> Detector:
    """Instantiate the backend registered as ``name``."""
    if name not in _registry:
        known = ", ".join(list_backends()) or "(none registered)"
        raise ValueError(f"Unknown backend '{name}'. Available backends: {known}")
    return _registry[name](**kwargs)


def scan_text(text: str, backend: str = "heuristic", **kwargs) -> ScanResult:
    """High-level convenience: scan ``text`` with the named backend."""
    return get_backend(backend, **kwargs).detect_text(text)


# Register the built-in backends (importing them wires the registry).
from synthscan.backends.heuristic import HeuristicDetector  # noqa: E402
from synthscan.backends.binoculars import BinocularsDetector  # noqa: E402
from synthscan.backends.roberta import RobertaDetector  # noqa: E402
