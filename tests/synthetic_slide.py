"""Generate a synthetic pyramidal tiled TIFF that OpenSlide can open.

There is no real slide on the build machine, and orientation is the single
highest-risk thing in this port (``04-DESIGN-DECISIONS.md`` §1).  So the test
slide carries **unambiguous corner markers** — a different pure colour in each
corner plus an upward-pointing wedge — which turns "is the slide upright?" from
an eyeball judgement into an assertion.

Corner colours (in level-0, top-left-origin space):

    top-left  = red      top-right    = green
    bottom-left = blue   bottom-right = yellow
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

#: Side length of each corner marker square, in level-0 px.
MARKER = 512

CORNER_COLORS = {
    "top_left": (220, 30, 30),
    "top_right": (30, 200, 30),
    "bottom_left": (30, 60, 220),
    "bottom_right": (230, 210, 30),
}


def corner_centre(name: str, width: int, height: int) -> tuple[int, int]:
    """Centre of a corner marker in level-0 coordinates."""
    half = MARKER // 2
    return {
        "top_left": (half, half),
        "top_right": (width - half, half),
        "bottom_left": (half, height - half),
        "bottom_right": (width - half, height - half),
    }[name]


def _base_image(width: int, height: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]

    # A pale eosin-ish background with gentle structure, so the image looks
    # like tissue rather than noise at low zoom.
    image = np.empty((height, width, 3), dtype=np.float32)
    image[..., 0] = 235 - 25 * np.sin(xs / 220.0) * np.cos(ys / 260.0)
    image[..., 1] = 210 - 30 * np.cos(ys / 190.0)
    image[..., 2] = 228 - 20 * np.sin((xs + ys) / 300.0)

    # Scattered haematoxylin-dark "nuclei", denser in the upper half so the
    # vertical asymmetry is visible even without the corner markers.
    count = 26_000
    nx = rng.integers(0, width, count)
    ny = (rng.beta(1.6, 3.0, count) * height).astype(int).clip(0, height - 1)
    image[ny, nx] = (90, 60, 140)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            image[(ny + dy).clip(0, height - 1), (nx + dx).clip(0, width - 1)] = (110, 75, 155)

    # An upward-pointing wedge: apex at the top, base at the bottom.
    apex_x, apex_y = width // 2, height // 8
    base_y = height - height // 8
    half_base = width // 4
    t = (ys - apex_y) / max(base_y - apex_y, 1)
    inside = (t >= 0) & (t <= 1) & (np.abs(xs - apex_x) <= half_base * t)
    image[inside] = (150, 100, 180)

    for name, color in CORNER_COLORS.items():
        cx, cy = corner_centre(name, width, height)
        y0, y1 = cy - MARKER // 2, cy + MARKER // 2
        x0, x1 = cx - MARKER // 2, cx + MARKER // 2
        image[y0:y1, x0:x1] = color

    return image.clip(0, 255).astype(np.uint8)


def _downsample(image: np.ndarray) -> np.ndarray:
    """Exact 2x box downsample (dimensions are kept even by the caller)."""
    h, w = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    cropped = image[:h, :w].astype(np.uint16)
    return ((cropped[0::2, 0::2] + cropped[1::2, 0::2]
             + cropped[0::2, 1::2] + cropped[1::2, 1::2]) // 4).astype(np.uint8)


def write_synthetic_slide(path: str | Path, width: int = 4096, height: int = 3072,
                          levels: int = 4, tile: int = 256) -> Path:
    """Write a multi-level tiled TIFF openable by OpenSlide's generic-tiff driver.

    Each pyramid level is a reduced-resolution IFD, which is what that driver
    looks for.  ``MPP`` is declared via the TIFF resolution tags so
    ``SlideImage.mpp_x`` has something to read.
    """
    path = Path(path)
    image = _base_image(width, height)

    # 0.5 um/px  ->  20000 px per cm  ->  resolution in px/cm.
    mpp = 0.5
    pixels_per_cm = 10_000.0 / mpp

    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        current = image
        for level in range(levels):
            writer.write(
                current,
                tile=(tile, tile),
                photometric="rgb",
                # zlib throughout: JPEG would need the optional imagecodecs
                # package, and lossless keeps the corner-marker assertions exact.
                compression="zlib",
                resolution=(pixels_per_cm / (2 ** level), pixels_per_cm / (2 ** level)),
                resolutionunit="CENTIMETER",
                subfiletype=1 if level else 0,  # 1 = reduced-resolution image
                metadata=None,
            )
            if level + 1 < levels:
                current = _downsample(current)

    return path


if __name__ == "__main__":  # manual generation for eyeballing in the app
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic-slide.tif")
    write_synthetic_slide(out, width=8192, height=6144, levels=5)
    print("wrote", out.resolve())
