"""OpenSlide wrapper — ported from ``Models/SlideImage.swift``.

Reproduces the Swift ``SlideImage`` interface so the rest of the port reads the
same way, with one important behavioural note carried over verbatim:

    ``read_region(x, y, level, w, h)``: **x, y are level-0 coordinates while
    w, h are in the target level's coordinates.**

That asymmetry is OpenSlide's own convention and every caller depends on it.
OpenSlide fills out-of-bounds reads with background rather than failing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openslide


class SlideError(Exception):
    """Raised when a slide cannot be opened or a region cannot be read."""


@dataclass(frozen=True, slots=True)
class Size:
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)


class SlideImage:
    """A whole-slide image backed by an OpenSlide handle.

    OpenSlide reads are thread-safe on a single handle, so the tile renderer
    may call :meth:`read_region` from worker threads.  We still guard handle
    *lifetime* with a lock so closing while tiles are in flight is safe.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self._osr = openslide.OpenSlide(str(self.path))
        except openslide.OpenSlideUnsupportedFormatError as exc:
            raise SlideError(f"Not a recognised whole-slide image: {self.path.name}") from exc
        except openslide.OpenSlideError as exc:
            raise SlideError(f"OpenSlide error opening {self.path.name}: {exc}") from exc
        except OSError as exc:
            raise SlideError(f"Could not open {self.path.name}: {exc}") from exc

        self._lock = threading.RLock()
        self._closed = False

        w0, h0 = self._osr.dimensions
        self.dimensions = Size(int(w0), int(h0))
        self.level_count: int = int(self._osr.level_count)
        self.level_downsamples: tuple[float, ...] = tuple(
            float(d) for d in self._osr.level_downsamples
        )
        self.level_dimensions: tuple[Size, ...] = tuple(
            Size(int(w), int(h)) for (w, h) in self._osr.level_dimensions
        )
        self.properties: dict[str, str] = dict(self._osr.properties)

    # -- metadata ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def mpp_x(self) -> float | None:
        """Microns per pixel at level 0, or ``None`` if the slide omits it."""
        return _float_or_none(self.properties.get(openslide.PROPERTY_NAME_MPP_X))

    @property
    def mpp_y(self) -> float | None:
        return _float_or_none(self.properties.get(openslide.PROPERTY_NAME_MPP_Y))

    def best_level_for_downsample(self, downsample: float) -> int:
        """Pyramid level whose downsample is closest below *downsample*."""
        level = int(self._osr.get_best_level_for_downsample(max(downsample, 1.0)))
        return max(0, min(level, self.level_count - 1))

    def level_for_mpp(self, target_mpp: float, *, fallback_mpp: float = 0.25) -> int:
        """Pyramid level closest to *target_mpp* microns/px.

        Used by the geometry pipeline, which must analyse every annotation at a
        fixed physical resolution — see ``04-DESIGN-DECISIONS.md`` §2.
        """
        base = self.mpp_x or fallback_mpp
        if base <= 0:
            base = fallback_mpp
        best, best_err = 0, float("inf")
        for level, downsample in enumerate(self.level_downsamples):
            err = abs(base * downsample - target_mpp)
            if err < best_err:
                best, best_err = level, err
        return best

    # -- pixels -----------------------------------------------------------

    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        """Read a region as an ``(height, width, 3)`` uint8 RGB array.

        *x*, *y* are **level-0** pixel coordinates; *width*, *height* are in
        *level*'s coordinates.  Alpha is composited over white, matching how a
        pathology slide's background reads, so downstream whiteness filters and
        stain deconvolution see the same pixels the user sees.
        """
        if width <= 0 or height <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        with self._lock:
            if self._closed:
                raise SlideError("Slide is closed")
            try:
                tile = self._osr.read_region((int(x), int(y)), int(level), (int(width), int(height)))
            except openslide.OpenSlideError as exc:
                raise SlideError(f"read_region failed at ({x},{y}) level {level}: {exc}") from exc

        rgba = np.asarray(tile, dtype=np.uint8)
        return _composite_over_white(rgba)

    def read_region_rgba(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        """As :meth:`read_region` but keeps the alpha channel, ``(h, w, 4)``."""
        if width <= 0 or height <= 0:
            return np.zeros((0, 0, 4), dtype=np.uint8)
        with self._lock:
            if self._closed:
                raise SlideError("Slide is closed")
            try:
                tile = self._osr.read_region((int(x), int(y)), int(level), (int(width), int(height)))
            except openslide.OpenSlideError as exc:
                raise SlideError(f"read_region failed at ({x},{y}) level {level}: {exc}") from exc
        return np.asarray(tile, dtype=np.uint8)

    def thumbnail(self, max_size: int = 1024) -> np.ndarray:
        """A whole-slide overview image, at most *max_size* on the long edge."""
        with self._lock:
            if self._closed:
                raise SlideError("Slide is closed")
            image = self._osr.get_thumbnail((max_size, max_size))
        return _composite_over_white(np.asarray(image.convert("RGBA"), dtype=np.uint8))

    # -- lifetime ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._osr.close()

    def __enter__(self) -> "SlideImage":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"SlideImage({self.name!r}, {self.dimensions.width}x{self.dimensions.height}, "
                f"{self.level_count} levels)")


def _composite_over_white(rgba: np.ndarray) -> np.ndarray:
    """Flatten RGBA onto a white background, returning RGB uint8."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return np.ascontiguousarray(rgba[..., :3])
    rgb = rgba[..., :3].astype(np.uint16)
    alpha = rgba[..., 3:4].astype(np.uint16)
    # OpenSlide returns premultiplied alpha, so compositing over white is just
    # out = premultiplied_rgb + (255 - alpha) * 255 / 255.
    out = rgb + (255 - alpha)
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_format(path: str | Path) -> str | None:
    """Vendor/driver name OpenSlide would use, or ``None`` if unsupported."""
    return openslide.OpenSlide.detect_format(str(path))


#: Extensions OpenSlide can open, for file dialogs.
SUPPORTED_EXTENSIONS = (
    ".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu",
    ".scn", ".mrxs", ".bif", ".svslide", ".dcm",
)
