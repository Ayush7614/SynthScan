"""Command-line interface for SynthScan.

Usage:
    synthscan scan "your text here" [--backend heuristic|binoculars] [--json]
    echo "some text" | synthscan scan --backend heuristic
    synthscan scan --file doc.txt
    synthscan serve [--host 127.0.0.1] [--port 8000] [--backend heuristic]
    synthscan list-backends
"""

import argparse
import json
import sys
from typing import List, Optional

from synthscan import __version__
from synthscan.core.detector import get_backend, list_backends, scan_text


def _read_text(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "file", None):
        with open(args.file, "r", encoding="utf-8") as handle:
            return handle.read()
    if getattr(args, "text", None):
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


def _print_result(result) -> None:
    print(result.human_readable())
    print("-" * 40)
    for seg in result.segments:
        marker = "AI" if seg.ai_probability >= 0.7 else (
            "ai?" if seg.ai_probability >= 0.4 else "hum"
        )
        print(f"[{marker}] ({seg.ai_probability:.0%}) {seg.text.strip()}")


def cmd_scan(args: argparse.Namespace) -> int:
    text = _read_text(args)
    if not text:
        print("No input provided. Pass --text, --file, or pipe text on stdin.",
              file=sys.stderr)
        return 2
    result = scan_text(text, backend=args.backend)
    if args.json:
        print(json.dumps({
            "verdict": result.verdict.value,
            "ai_probability": result.ai_probability,
            "is_ai": result.is_ai,
            "backend": result.backend,
            "segments": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "ai_probability": s.ai_probability,
                }
                for s in result.segments
            ],
        }, indent=2))
    else:
        _print_result(result)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from uvicorn import run as uvicorn_run
    except ImportError as exc:  # pragma: no cover
        print("The API requires 'synthscan[api]'. Install with: "
              "pip install 'synthscan[api]'", file=sys.stderr)
        return 2

    from synthscan.api.server import create_app

    app = create_app(default_backend=args.backend)
    uvicorn_run(app, host=args.host, port=args.port)
    return 0


def cmd_list_backends(args: argparse.Namespace) -> int:  # noqa: ARG001
    for name in list_backends():
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthscan",
        description="Open-source, self-hostable AI-content authenticity scanner.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan text for AI-generated content.")
    scan.add_argument("text", nargs="?")
    scan.add_argument("--file")
    scan.add_argument("--backend", default="heuristic",
                      choices=list_backends() or ["heuristic", "binoculars"])
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=cmd_scan)

    serve = sub.add_parser("serve", help="Run the SynthScan API server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--backend", default="heuristic")
    serve.set_defaults(func=cmd_serve)

    sub.add_parser("list-backends", help="List available detection backends.").set_defaults(
        func=cmd_list_backends
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
