#!/usr/bin/env python3
"""Generate the SynthScan browser-extension icons as PNGs.

Reproducible, dependency-free (stdlib only). Usage:
    python scripts/generate_icons.py [out_dir]

Creates icon16.png, icon48.png, icon128.png (a simple blue "scan bar" mark).
"""

import struct
import sys
import zlib
from pathlib import Path

# SynthScan brand-ish blue.
BG = (74, 108, 247)      # blue
BAR = (255, 255, 255)    # white scan bar
LENS = (255, 224, 130)   # amber lens dot


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, size: int, pixel_fn) -> None:
    rows = bytearray()
    for y in range(size):
        rows.append(0)  # filter 0
        for x in range(size):
            r, g, b = pixel_fn(x, y, size)
            rows += bytes((r, g, b))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def make_pixel_fn():
    def pixel(x, y, size):
        # Vertical "scan bar" in the middle.
        bar_w = max(1, int(size * 0.16))
        cx = size // 2
        in_bar = abs(x - cx) <= bar_w // 2 and y >= int(size * 0.2) and y <= int(size * 0.8)

        # A lens dot above the bar.
        lens_cx, lens_cy = cx, int(size * 0.5)
        lens_r = max(2, int(size * 0.16))
        dx, dy = x - lens_cx, y - lens_cy
        in_lens = dx * dx + dy * dy <= lens_r * lens_r

        if in_lens:
            return LENS
        if in_bar:
            return BAR
        return BG
    return pixel


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pixel = make_pixel_fn()
    for size, name in [(16, "icon16.png"), (48, "icon48.png"), (128, "icon128.png")]:
        write_png(out / name, size, pixel)
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    default = str(Path(__file__).resolve().parent.parent / "extension" / "icons")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else default
    raise SystemExit(main(out_dir))
