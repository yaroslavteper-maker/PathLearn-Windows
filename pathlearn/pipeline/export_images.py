"""Save each annotation as a JPEG cut from the slide.

For getting traced regions out of PathLearn and into something else — a
figure, a slide deck, a colleague's inbox, or another training pipeline that
wants a folder of images per class.

The image is the *pixels under the outline*, not a screenshot of the canvas:
read straight from the slide at a chosen resolution, so it does not depend on
the zoom you happened to be at.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..io.slide import SlideImage
from ..models.annotation import Annotation

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]
CancelFn = Callable[[], bool]

#: Beyond this, a level-0 crop of a large region becomes unwieldy — a 20,000px
#: JPEG is slow to write and refuses to open in most viewers.
DEFAULT_MAX_EDGE = 4096

#: JPEG's own hard ceiling per side.
JPEG_MAX_EDGE = 65_500


class ExportError(RuntimeError):
    """Raised when nothing could be exported at all."""


@dataclass(frozen=True, slots=True)
class ExportSettings:
    """How to cut and write the images."""

    #: Pyramid level to read. None picks the finest level whose crop fits
    #: within ``max_edge``, which is almost always what you want.
    level: int | None = None
    #: Extra context around the bounding box, in level-0 pixels.
    margin: int = 0
    #: Longest side of the written image; larger crops are downscaled.
    max_edge: int = DEFAULT_MAX_EDGE
    quality: int = 92
    #: Paint everything outside the polygon white, so only the traced region
    #: shows. Off by default: the surrounding tissue is usually the context
    #: that makes the region readable.
    mask_outside: bool = False
    #: Draw the outline onto the image in the class colour.
    draw_outline: bool = False
    #: Write into one subfolder per class — the layout most image classifiers
    #: expect, and the reason this export usually exists.
    folder_per_class: bool = True


@dataclass(frozen=True, slots=True)
class ExportedImage:
    path: Path
    annotation_id: object
    classification: str
    width: int
    height: int
    level: int


@dataclass
class ExportReport:
    written: list[ExportedImage] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (name, reason)
    cancelled: bool = False

    @property
    def bytes_written(self) -> int:
        return sum(i.path.stat().st_size for i in self.written if i.path.exists())

    def summary(self) -> str:
        if not self.written and not self.skipped:
            return "Nothing to export."
        text = f"Wrote {len(self.written)} image(s)"
        if self.written:
            text += f" ({self.bytes_written / 1e6:.1f} MB)"
        if self.skipped:
            text += f"; skipped {len(self.skipped)}"
        return text + ("  (cancelled early)" if self.cancelled else ".")


def export_annotation_images(slide: SlideImage, annotations: Sequence[Annotation],
                             out_dir: Path,
                             settings: ExportSettings | None = None,
                             *, progress: ProgressFn | None = None,
                             should_cancel: CancelFn | None = None) -> ExportReport:
    """Write one JPEG per annotation into *out_dir*."""
    from PIL import Image

    settings = settings or ExportSettings()
    report = ExportReport()
    usable = [a for a in annotations if len(a.points) >= 3]
    if not usable:
        return report

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()

    for index, annotation in enumerate(usable):
        if should_cancel and should_cancel():
            report.cancelled = True
            break
        if progress:
            progress(index, len(usable),
                     f"{annotation.classification}: {len(report.written)} written")

        try:
            pixels, level = _crop(slide, annotation, settings)
        except Exception as exc:                     # noqa: BLE001 - reported
            log.debug("crop failed for %s: %s", annotation.id, exc)
            report.skipped.append((_label(annotation, index), str(exc)))
            continue
        if pixels.size == 0:
            report.skipped.append((_label(annotation, index), "empty crop"))
            continue

        folder = out_dir / _safe(annotation.classification) if settings.folder_per_class \
            else out_dir
        folder.mkdir(parents=True, exist_ok=True)
        path = _unique(folder, _filename(slide, annotation, index), used_names)

        image = Image.fromarray(pixels)
        try:
            image.save(path, "JPEG", quality=settings.quality, optimize=True)
        except OSError as exc:
            report.skipped.append((_label(annotation, index), f"could not write: {exc}"))
            continue
        report.written.append(ExportedImage(
            path=path, annotation_id=annotation.id,
            classification=annotation.classification,
            width=pixels.shape[1], height=pixels.shape[0], level=level))

    if progress:
        progress(len(usable), len(usable), report.summary())
    return report


# -- cutting -----------------------------------------------------------------

def _crop(slide: SlideImage, annotation: Annotation,
          settings: ExportSettings) -> tuple[np.ndarray, int]:
    """The pixels for one annotation, plus the level they were read at."""
    min_x, min_y, width, height = annotation.bounding_box
    min_x = int(min_x) - settings.margin
    min_y = int(min_y) - settings.margin
    width = int(round(width)) + 2 * settings.margin
    height = int(round(height)) + 2 * settings.margin
    if width <= 0 or height <= 0:
        raise ExportError("degenerate bounding box")

    level = settings.level
    if level is None:
        # Finest level whose crop still fits the cap, so a big lesion comes
        # back readable rather than as a 20,000px file nothing will open.
        longest = max(width, height)
        needed = longest / max(settings.max_edge, 1)
        level = slide.best_level_for_downsample(max(needed, 1.0))
    level = max(0, min(int(level), slide.level_count - 1))

    downsample = slide.level_downsamples[level]
    level_w = max(1, int(round(width / downsample)))
    level_h = max(1, int(round(height / downsample)))
    if max(level_w, level_h) > JPEG_MAX_EDGE:
        raise ExportError(f"{level_w}x{level_h} exceeds JPEG's {JPEG_MAX_EDGE}px limit; "
                          f"pick a coarser pyramid level")

    pixels = slide.read_region(min_x, min_y, level, level_w, level_h)
    if pixels.size == 0:
        return pixels, level

    if settings.mask_outside or settings.draw_outline:
        pixels = _apply_polygon(pixels, annotation, min_x, min_y, downsample, settings)

    if max(pixels.shape[:2]) > settings.max_edge:
        pixels = _downscale(pixels, settings.max_edge)
    return pixels, level


def _apply_polygon(pixels: np.ndarray, annotation: Annotation,
                   origin_x: int, origin_y: int, downsample: float,
                   settings: ExportSettings) -> np.ndarray:
    """Mask outside the outline and/or draw it, in image coordinates."""
    from PIL import Image, ImageDraw

    points = [((p.x - origin_x) / downsample, (p.y - origin_y) / downsample)
              for p in annotation.points]
    image = Image.fromarray(pixels)

    if settings.mask_outside:
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        # White rather than black: it reads as slide background, which is what
        # every downstream whiteness filter already expects.
        white = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.composite(image, white, mask)

    if settings.draw_outline:
        colour = (annotation.color.r, annotation.color.g, annotation.color.b)
        thickness = max(1, round(min(image.size) / 300))
        ImageDraw.Draw(image).line(points + [points[0]], fill=colour, width=thickness)

    return np.asarray(image, dtype=np.uint8)


def _downscale(pixels: np.ndarray, max_edge: int) -> np.ndarray:
    from PIL import Image

    height, width = pixels.shape[:2]
    scale = max_edge / max(height, width)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return np.asarray(Image.fromarray(pixels).resize(size, Image.LANCZOS),
                      dtype=np.uint8)


# -- naming ------------------------------------------------------------------

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe(text: str, fallback: str = "unlabelled") -> str:
    """A filename component Windows will accept.

    Class names are user-typed and routinely contain characters Windows
    forbids, so this is not decoration — without it the export fails on the
    first slash in a name.
    """
    cleaned = _UNSAFE.sub("-", (text or "").strip()).strip(". ")
    return cleaned or fallback


def _label(annotation: Annotation, index: int) -> str:
    return annotation.display_name or f"{annotation.classification} #{index + 1}"


def _filename(slide: SlideImage, annotation: Annotation, index: int) -> str:
    parts = [_safe(Path(slide.name).stem, "slide"), _safe(annotation.classification)]
    if annotation.display_name:
        parts.append(_safe(annotation.display_name))
    else:
        parts.append(f"{index + 1:03d}")
    return "_".join(parts) + ".jpg"


def _unique(folder: Path, name: str, used: set[str]) -> Path:
    """Never overwrite: two regions can share a class and a name."""
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = name, 2
    while str(folder / candidate) in used or (folder / candidate).exists():
        candidate = f"{stem}-{n}{suffix}"
        n += 1
    used.add(str(folder / candidate))
    return folder / candidate
