"""FastAPI server exposing SynthScan as a self-hostable JSON API.

Endpoints:
    GET  /health          -> liveness + backend registry
    POST /scan/text       -> scan a string of text
Payload for POST /scan/text:
    {
      "text": "the text to analyze",
      "backend": "heuristic"   // optional; defaults to the server's default
    }
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from synthscan import __version__
from synthscan.core.detector import get_backend, list_backends


class ScanRequest(BaseModel):
    text: str
    backend: Optional[str] = None


class SegmentSchema(BaseModel):
    text: str
    start: int
    end: int
    ai_probability: float


class ScanResponse(BaseModel):
    verdict: str
    ai_probability: float
    is_ai: bool
    backend: str
    segments: list[SegmentSchema]
    synthscan_version: str


def create_app(default_backend: str = "heuristic") -> FastAPI:
    app = FastAPI(
        title="SynthScan API",
        description="Open-source, self-hostable AI-content authenticity scanner.",
        version=__version__,
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "synthscan_version": __version__,
            "backends": list_backends(),
        }

    @app.post("/scan/text", response_model=ScanResponse)
    def scan_text_endpoint(request: ScanRequest):
        backend = request.backend or default_backend
        if backend not in list_backends():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown backend '{backend}'. Available: {list_backends()}",
            )
        detector = get_backend(backend)
        result = detector.detect_text(request.text)
        return ScanResponse(
            verdict=result.verdict.value,
            ai_probability=result.ai_probability,
            is_ai=result.is_ai,
            backend=result.backend,
            segments=[
                SegmentSchema(
                    text=s.text, start=s.start, end=s.end,
                    ai_probability=s.ai_probability,
                )
                for s in result.segments
            ],
            synthscan_version=__version__,
        )

    return app


app = create_app()
