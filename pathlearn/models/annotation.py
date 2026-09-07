"""Annotation model — ported from ``Models/Annotation.swift``.

Coordinates are **level-0 slide pixels, top-left origin, Y down** (see
:mod:`pathlearn.coords`).  This is the single coordinate space in the Windows
build; unlike the macOS app there is no mirrored "view-Y" space.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Iterable, NamedTuple, Sequence


class Point(NamedTuple):
    """A point in level-0 slide pixels."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class AnnotationColor:
    """RGB colour, components 0-255."""

    r: int
    g: int
    b: int

    @staticmethod
    def default() -> "AnnotationColor":
        return AnnotationColor(200, 60, 60)

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @classmethod
    def from_sequence(cls, seq: Sequence[float]) -> "AnnotationColor":
        r, g, b = (int(round(float(v))) for v in seq[:3])
        return cls(_clamp8(r), _clamp8(g), _clamp8(b))


def _clamp8(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v


@dataclass(slots=True)
class Annotation:
    """A polygon region drawn on a slide.

    ``is_subtractive`` polygons carve regions *out* of enclosing annotations:
    during patch extraction any patch whose centre falls inside a subtractive
    polygon is cancelled before it reaches the bank (e.g. to exclude empty
    lumen space inside a lesion).  Subtractive annotations are never themselves
    extracted.
    """

    points: list[Point]
    classification: str
    color: AnnotationColor
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str | None = None
    #: Canvas display only. macOS-era sidecars carry this flag freely — an
    #: annotation hidden to see the tissue beneath was saved as isVisible=false
    #: — so it must NEVER gate analysis.
    is_visible: bool = True
    #: Whether pipelines should act on this annotation. Separate from display
    #: on purpose: conflating them made every previously-hidden annotation in
    #: an existing sidecar silently drop out of extraction and geometry.
    #: Defaults True so legacy files, which have no such key, stay usable.
    is_selected: bool = True
    is_subtractive: bool = False

    def __post_init__(self) -> None:
        # Accept any (x, y) iterable but always store Points.
        self.points = [p if isinstance(p, Point) else Point(float(p[0]), float(p[1]))
                       for p in self.points]

    # -- geometry ---------------------------------------------------------

    @property
    def area_in_slide_pixels(self) -> float:
        """Shoelace area in level-0 px^2.  Assumes a simple polygon."""
        pts = self.points
        if len(pts) < 3:
            return 0.0
        total = 0.0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            total += pts[i].x * pts[j].y
            total -= pts[j].x * pts[i].y
        return abs(total) / 2.0

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, width, height)`` in level-0 px."""
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test, matching the Swift overlay."""
        pts = self.points
        n = len(pts)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                # x of the edge at height y
                t = (y - yi) / (yj - yi)
                if x < xi + t * (xj - xi):
                    inside = not inside
            j = i
        return inside

    @property
    def area_short(self) -> str:
        """Compact area string like ``1.2M`` / ``340K`` / ``120``."""
        a = self.area_in_slide_pixels
        if a >= 1_000_000:
            return f"{a / 1_000_000:.1f}M"
        if a >= 1_000:
            return f"{a / 1_000:.0f}K"
        return f"{a:.0f}"

    @property
    def display_name(self) -> str:
        return self.name or ""

    # -- transforms -------------------------------------------------------

    def mirrored_y(self, slide_height: float) -> "Annotation":
        """Return a copy with every ``y`` replaced by ``slide_height - y``.

        Only used at the legacy-migration boundary — see
        :mod:`pathlearn.coords`.  Unlike the Swift ``mirrorYForImport`` this
        preserves ``is_subtractive`` (the Swift version silently dropped it).
        """
        return replace(self, points=[Point(p.x, slide_height - p.y) for p in self.points])

    def copy_with_new_id(self) -> "Annotation":
        return replace(self, id=uuid.uuid4())


def mirror_all_y(annotations: Iterable[Annotation], slide_height: float) -> list[Annotation]:
    """Mirror Y for a whole collection."""
    return [a.mirrored_y(slide_height) for a in annotations]
