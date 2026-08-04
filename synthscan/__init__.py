"""SynthScan - open-source, self-hostable AI-content authenticity scanner.

Detects synthetic (AI-generated) content. Core text detection works without any
heavy machine-learning dependencies via the heuristic backend; production-grade
detection uses the optional Binoculars zero-shot backend.
"""

from synthscan.core.result import ScanResult, SegmentResult, Verdict
from synthscan.core.detector import (
    Detector,
    get_backend,
    list_backends,
    register_backend,
    scan_text,
)
from synthscan.backends.binoculars import BinocularsDetector  # noqa: F401
from synthscan.backends.roberta import RobertaDetector  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ScanResult",
    "SegmentResult",
    "Verdict",
    "Detector",
    "get_backend",
    "list_backends",
    "register_backend",
    "scan_text",
]
