"""What fraction of your tracing each class accounts for.

This is about the annotations *you drew*, not about anything a model predicted
— the question "of everything I have outlined on this slide, how much is
PanIN-2?".  For the model's answer, see ``pipeline/composition.py``.

TWO ANSWERS, AND THEY DISAGREE ON PURPOSE
=========================================
*Count share* is how many regions carry each class.  It is the literal reading
of "percentage of each class", and it is the right number when each traced
region is one lesion and you are asking how the lesions are distributed.

*Area share* is how much slide each class covers.  It is the right number when
you are asking how much tissue is involved.

They separate sharply in PanIN work, and the reason is measured rather than
theoretical: region area rises steeply with grade, by close to an order of
magnitude across the range.  A slide with twenty small 1a ducts and two large
grade-3 lesions is over 90% 1a by count and can be mostly grade 3 by area.
Quoting one without the other is how that slide gets described two opposite
ways, so both are always reported.

SUBTRACTIVE POLYGONS ARE HOLES, NOT REGIONS
===========================================
A subtractive annotation carves space *out* of the region enclosing it — empty
lumen inside a lesion, say — and is never itself extracted.  So it is:

* **not counted** as a region of its own class, and
* **deducted** from the area of the enclosing annotation's class.

Which annotation encloses it is decided by the innermost positive region
containing its centroid, so nested tracings deduct from the region actually
carved.  A subtractive polygon inside nothing is an orphan: it is reported and
deducts nothing, because guessing which class it meant would be inventing data.

OVERLAPPING TRACES ARE COUNTED TWICE
====================================
Area here is the sum of polygon areas.  Two hand traces that overlap contribute
their overlap to both, so the total exceeds the tissue actually covered.  Unlike
the tile grid in ``composition.py`` — where an exact union is cheap — resolving
polygon overlap properly needs real polygon clipping, and silently approximating
it would be worse than saying so.  Hand-drawn regions rarely overlap; when they
do, the breakdown says how many pairs and the numbers should be read as upper
bounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..models.annotation import Annotation
from .composition import format_area

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClassTally:
    """One class's share of the tracing."""

    label: str
    count: int
    area_px: float
    count_fraction: float
    area_fraction: float
    mean_area_px: float
    area_mm2: float | None = None
    mean_area_mm2: float | None = None
    #: Area removed from this class by subtractive polygons inside it.
    carved_px: float = 0.0

    @property
    def count_percent(self) -> float:
        return self.count_fraction * 100.0

    @property
    def area_percent(self) -> float:
        return self.area_fraction * 100.0


@dataclass
class AnnotationBreakdown:
    """Every class's count and area share, largest area first."""

    tallies: list[ClassTally] = field(default_factory=list)
    total_count: int = 0
    total_area_px: float = 0.0
    total_area_mm2: float | None = None
    mpp: float | None = None
    scope: str = "all annotations"
    #: Holes: excluded from the counts, deducted from their enclosing class.
    subtractive_count: int = 0
    carved_px: float = 0.0
    #: Subtractive polygons inside no region — they deduct from nothing.
    orphan_subtractive: int = 0
    #: Pairs of positive annotations whose outlines may overlap.
    overlapping_pairs: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.tallies

    def tally_for(self, label: str) -> ClassTally | None:
        return next((t for t in self.tallies if t.label == label), None)

    def area_text(self) -> str:
        return format_area(self.total_area_px, self.total_area_mm2)

    def summary(self) -> str:
        if self.is_empty:
            return "No annotations to break down."
        parts = ", ".join(f"{t.label} {t.count_percent:.1f}%"
                          for t in self.tallies)
        text = (f"{self.total_count} annotation(s) over {self.area_text()} "
                f"— by count: {parts}")
        if self.subtractive_count:
            text += (f"  ({self.subtractive_count} subtractive polygon(s) "
                     "excluded and deducted)")
        return text + "."

    def to_csv(self) -> str:
        lines = ["Class,Annotations,Count %,Area (px^2),Area (mm^2),Area %,"
                 "Mean area (px^2),Mean area (mm^2),Carved out (px^2)"]
        for tally in self.tallies:
            mm2 = "" if tally.area_mm2 is None else f"{tally.area_mm2:.4f}"
            mean_mm2 = ("" if tally.mean_area_mm2 is None
                        else f"{tally.mean_area_mm2:.4f}")
            lines.append(
                f"{_csv(tally.label)},{tally.count},{tally.count_percent:.2f},"
                f"{tally.area_px:.0f},{mm2},{tally.area_percent:.2f},"
                f"{tally.mean_area_px:.0f},{mean_mm2},{tally.carved_px:.0f}")
        total_mm2 = ("" if self.total_area_mm2 is None
                     else f"{self.total_area_mm2:.4f}")
        lines.append(f"Total,{self.total_count},100.00,{self.total_area_px:.0f},"
                     f"{total_mm2},100.00,,,{self.carved_px:.0f}")
        mpp_text = "unknown" if self.mpp is None else f"{self.mpp:.4f}"
        lines += [
            "",
            f"# Scope,{_csv(self.scope)}",
            "# Source,Hand-drawn annotations - not model predictions",
            "# Count % versus Area %,"
            "\"Count is how many regions carry the class; area is how much "
            "slide they cover. In PanIN work these differ sharply because "
            "region size rises steeply with grade.\"",
            f"# Subtractive polygons excluded,{self.subtractive_count}",
            f"# Subtractive area deducted (px^2),{self.carved_px:.0f}",
            f"# Subtractive polygons inside no region,{self.orphan_subtractive}",
            f"# Possibly overlapping pairs,{self.overlapping_pairs}",
            f"# Microns per pixel,{mpp_text}",
        ]
        return "\n".join(lines) + "\n"


def _csv(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(character in text for character in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _to_mm2(area_px: float, mpp: float) -> float:
    return area_px * (mpp * mpp) / 1_000_000.0


def _centroid(annotation: Annotation) -> tuple[float, float]:
    """Shoelace centroid, falling back to the mean point for a degenerate one."""
    points = annotation.points
    if not points:
        return (0.0, 0.0)
    if len(points) < 3:
        return (sum(p.x for p in points) / len(points),
                sum(p.y for p in points) / len(points))
    twice_area = 0.0
    cx = cy = 0.0
    count = len(points)
    for i in range(count):
        j = (i + 1) % count
        cross = points[i].x * points[j].y - points[j].x * points[i].y
        twice_area += cross
        cx += (points[i].x + points[j].x) * cross
        cy += (points[i].y + points[j].y) * cross
    if twice_area == 0.0:
        return (sum(p.x for p in points) / count,
                sum(p.y for p in points) / count)
    return (cx / (3.0 * twice_area), cy / (3.0 * twice_area))


def _boxes_overlap(a: Annotation, b: Annotation) -> bool:
    ax, ay, aw, ah = a.bounding_box
    bx, by, bw, bh = b.bounding_box
    return not (ax + aw <= bx or bx + bw <= ax
                or ay + ah <= by or by + bh <= ay)


def breakdown(annotations: Iterable[Annotation], *, mpp: float | None = None,
              scope: str = "all annotations") -> AnnotationBreakdown:
    """Break a set of hand-drawn annotations down by class.

    *mpp* is microns per level-0 pixel; without it areas stay in pixels.
    """
    everything = list(annotations)
    positives = [a for a in everything if not a.is_subtractive]
    holes = [a for a in everything if a.is_subtractive]

    report = AnnotationBreakdown(mpp=mpp, scope=scope,
                                 subtractive_count=len(holes))
    if not positives:
        # Holes with nothing to carve are all orphans, and there is no
        # breakdown to give — but say why rather than showing an empty table.
        report.orphan_subtractive = len(holes)
        return report

    counts: dict[str, int] = {}
    areas: dict[str, float] = {}
    carved: dict[str, float] = {}
    colours: dict[str, object] = {}
    for annotation in positives:
        label = annotation.classification or "Unlabeled"
        counts[label] = counts.get(label, 0) + 1
        areas[label] = areas.get(label, 0.0) + annotation.area_in_slide_pixels
        carved.setdefault(label, 0.0)
        colours.setdefault(label, annotation.color)

    for hole in holes:
        owner = _enclosing(hole, positives)
        if owner is None:
            report.orphan_subtractive += 1
            continue
        label = owner.classification or "Unlabeled"
        area = hole.area_in_slide_pixels
        carved[label] = carved.get(label, 0.0) + area
        # Clamp: a hole larger than its parent is a tracing mistake, and a
        # negative class area would be worse than an obviously-zero one.
        areas[label] = max(0.0, areas.get(label, 0.0) - area)
        report.carved_px += area

    total_count = sum(counts.values())
    total_area = sum(areas.values())
    report.total_count = total_count
    report.total_area_px = total_area
    if mpp:
        report.total_area_mm2 = _to_mm2(total_area, mpp)

    tallies: list[ClassTally] = []
    for label, count in counts.items():
        area = areas[label]
        mean = area / count if count else 0.0
        tallies.append(ClassTally(
            label=label, count=count, area_px=area,
            count_fraction=count / total_count if total_count else 0.0,
            # A slide whose every region was carved to nothing has no area to
            # apportion; report zero rather than dividing by it.
            area_fraction=area / total_area if total_area else 0.0,
            mean_area_px=mean,
            area_mm2=_to_mm2(area, mpp) if mpp else None,
            mean_area_mm2=_to_mm2(mean, mpp) if mpp else None,
            carved_px=carved.get(label, 0.0)))

    tallies.sort(key=lambda t: (-t.area_fraction, -t.count, t.label))
    report.tallies = tallies
    report.overlapping_pairs = _count_overlaps(positives)
    return report


def _enclosing(hole: Annotation,
               positives: Sequence[Annotation]) -> Annotation | None:
    """The innermost positive region containing *hole*'s centroid.

    Innermost — smallest by area — so that a hole inside a lesion inside a
    block deducts from the lesion, which is the region actually carved.
    """
    x, y = _centroid(hole)
    containing = [a for a in positives if a.contains(x, y)]
    if not containing:
        return None
    return min(containing, key=lambda a: a.area_in_slide_pixels)


#: Above this many regions the pairwise overlap check is skipped: it is O(n^2)
#: and a stitched slide can carry thousands of regions, where a several-second
#: freeze to compute an advisory footnote is a bad trade.
MAX_OVERLAP_CHECK = 400


def _count_overlaps(positives: Sequence[Annotation]) -> int:
    """Pairs whose bounding boxes intersect — an upper bound, deliberately.

    Bounding boxes, not outlines: this only decides whether to show a caveat,
    and a cheap over-estimate that occasionally warns unnecessarily beats an
    exact answer nobody waits for.
    """
    if len(positives) > MAX_OVERLAP_CHECK:
        return 0
    pairs = 0
    for index, first in enumerate(positives):
        for second in positives[index + 1:]:
            if _boxes_overlap(first, second):
                pairs += 1
    return pairs
