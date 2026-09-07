"""Turn predicted tiles of one class into annotations, one per connected region.

After prediction, the tiles carrying a given label are scattered across the
slide in clumps. This joins each clump into a polygon you can describe with the
geometry pipeline, and keeps clumps that do not touch as separate annotations
of the same class — which is the point: two lesions are two lesions.

HOW ADJACENCY IS DECIDED
========================
Tiles are rasterised onto a grid before grouping, rather than compared
pairwise. Prediction can run several passes at different patch sizes, and with
a stride below the patch size the tiles overlap, so "do these two touch" has no
clean answer at the tile level. On a raster it is exactly connected-component
labelling, and overlapping or differently-sized tiles all reduce to covered
cells.

Default connectivity is 4 (edge-sharing). Two tiles meeting only at a corner
are not one lesion in any sense a pathologist would accept, so diagonal
touching is opt-in.

THE STAIRCASE PROBLEM — READ BEFORE FEEDING THESE TO A SHAPE MODEL
==================================================================
A stitched outline follows tile edges, so it is a staircase of right angles.
The outline-shape descriptor measures things like boundary complexity and
radial roughness, and a staircase scores high on both for reasons that are
about the tile grid rather than the tissue. A model trained on smooth hand
traces will therefore read stitched regions as more irregular than they are.
``simplify_tolerance`` removes the steps, but it also moves the boundary, so
it is **off by default** — the region you get is exactly the tiles you chose.
Smoothing does not make a stitched outline equivalent to a hand trace either,
and the two should not be mixed in one training set without checking that
assumption first.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..models.annotation import Annotation, AnnotationColor, Point
from ..models.prediction import PatchPrediction, PredictionSet

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]

#: Fewer cells than this and the outline is noise rather than a region.
DEFAULT_MIN_CELLS = 2

#: Douglas-Peucker tolerance in grid cells. **Zero by default**: smoothing
#: moves the boundary, and measured on an L-shaped region a tolerance of 0.9
#: shaved the inner corner and lost 10% of the area. A faithful region is the
#: better default; smoothing the staircase is a deliberate trade the caller
#: makes with its eyes open.
DEFAULT_SIMPLIFY = 0.0


class StitchError(ValueError):
    """Raised when predictions cannot be stitched at all."""


@dataclass(frozen=True, slots=True)
class StitchSettings:
    """How to group tiles and trace their outlines."""

    #: Only tiles at or above this confidence are used.
    min_confidence: float = 0.0
    #: 4 = edge-sharing only; 8 also joins tiles meeting at a corner.
    connectivity: int = 4
    #: Drop regions smaller than this many grid cells.
    min_cells: int = DEFAULT_MIN_CELLS
    #: Douglas-Peucker tolerance in cells; 0 keeps the raw staircase.
    simplify_tolerance: float = DEFAULT_SIMPLIFY
    #: Grid cell size in level-0 px. None infers it from the tiles.
    cell_size: int | None = None


@dataclass
class StitchReport:
    annotations: list[Annotation] = field(default_factory=list)
    #: Regions dropped for being smaller than ``min_cells``.
    too_small: int = 0
    tiles_used: int = 0
    cell_size: int = 0
    cancelled: bool = False

    def summary(self) -> str:
        if self.cancelled and not self.annotations:
            # Not "nothing matched": it was stopped, which is a different
            # thing and the user needs to know which happened.
            return "Stopped early — no regions were created."
        if not self.annotations and not self.too_small:
            return "No tiles of that class passed the filters."
        text = (f"{len(self.annotations)} region(s) from {self.tiles_used} tile(s) "
                f"at {self.cell_size} px/cell")
        if self.too_small:
            text += f"; {self.too_small} region(s) too small"
        return text + ("  (stopped early)." if self.cancelled else ".")


def stitch_class(predictions: PredictionSet, label: str,
                 settings: StitchSettings | None = None,
                 *, progress: ProgressFn | None = None,
                 should_cancel: Callable[[], bool] | None = None) -> StitchReport:
    """Join the tiles predicted as *label* into one annotation per clump."""
    settings = settings or StitchSettings()
    tiles = [p for p in predictions.predictions
             if p.label == label and p.confidence >= settings.min_confidence]
    report = StitchReport(tiles_used=len(tiles))
    if not tiles:
        return report

    cell = settings.cell_size or infer_cell_size(tiles)
    if cell <= 0:
        raise StitchError("Could not work out a tile grid from these predictions.")
    report.cell_size = cell

    if progress:
        progress(0, 3, f"Rasterising {len(tiles)} tile(s)…")
    mask, origin_x, origin_y = rasterise(tiles, cell)

    if progress:
        progress(1, 3, "Grouping adjacent tiles…")
    from scipy import ndimage

    structure = (np.ones((3, 3), dtype=bool) if settings.connectivity == 8
                 else ndimage.generate_binary_structure(2, 1))
    labelled, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return report

    if progress:
        progress(2, 3, f"Tracing {count} region(s)…")
    colour = predictions.color_for(label)
    # Crop each region to its own bounding box before tracing. Passing the
    # full grid made the boundary walk cost O(whole slide) PER REGION: 6,000
    # regions on a 100x60 grid meant 36 million iterations and a nine-second
    # freeze. find_objects gives each box in one pass.
    boxes = ndimage.find_objects(labelled)
    sizes = np.bincount(labelled.ravel(), minlength=count + 1)
    for index in range(1, count + 1):
        if should_cancel and should_cancel():
            report.cancelled = True
            break
        if progress and index % 200 == 0:
            progress(index, count, f"Tracing region {index} of {count}…")
        if int(sizes[index]) < settings.min_cells:
            report.too_small += 1
            continue
        box = boxes[index - 1]
        if box is None:
            continue
        rows, cols = box
        component = labelled[rows, cols] == index
        polygon = trace_outline(
            component, cell,
            origin_x + cols.start * cell, origin_y + rows.start * cell,
            settings.simplify_tolerance)
        if polygon is None:
            report.too_small += 1
            continue
        report.annotations.append(Annotation(
            points=polygon, classification=label, color=colour,
            name=f"{label} stitched {len(report.annotations) + 1}"))

    if progress:
        progress(3, 3, report.summary())
    return report


def infer_cell_size(tiles: Sequence[PatchPrediction]) -> int:
    """The grid pitch these tiles sit on, in level-0 pixels.

    Taken from the smallest gap between distinct origins rather than from the
    patch size: with a stride below the patch size the tiles overlap, and the
    pitch — not the tile — is the grid the predictions actually live on.
    """
    from collections import Counter

    sizes = [t.size_level0 for t in tiles if t.size_level0 > 0]
    smallest_tile = min(sizes) if sizes else 0

    # The MOST COMMON gap, not the smallest. Taking the minimum meant a
    # single stray pair of origins one pixel apart — which multi-pass
    # prediction at different levels readily produces — collapsed the cell
    # size to 1 px and asked for a 2.4 GB raster. The mode is the grid the
    # tiles actually sit on; outliers cannot move it.
    gaps: Counter[int] = Counter()
    for values in (sorted({t.x for t in tiles}), sorted({t.y for t in tiles})):
        gaps.update(b - a for a, b in zip(values, values[1:]) if b > a)
    pitch = gaps.most_common(1)[0][0] if gaps else 0

    if pitch and smallest_tile:
        return int(min(pitch, smallest_tile))
    return int(pitch or smallest_tile)


#: A raster larger than this is refused. At 224 px cells this is a grid of
#: roughly 9,000 x 9,000 tiles — far beyond any real slide, so hitting it
#: means the cell size is wrong rather than the slide being large.
MAX_RASTER_CELLS = 80_000_000


def rasterise(tiles: Sequence[PatchPrediction],
              cell: int) -> tuple[np.ndarray, int, int]:
    """A boolean grid of covered cells, plus its level-0 origin."""
    origin_x = min(t.x for t in tiles)
    origin_y = min(t.y for t in tiles)
    far_x = max(t.x + max(t.size_level0, cell) for t in tiles)
    far_y = max(t.y + max(t.size_level0, cell) for t in tiles)

    width = max(1, int(math.ceil((far_x - origin_x) / cell)))
    height = max(1, int(math.ceil((far_y - origin_y) / cell)))
    if width * height > MAX_RASTER_CELLS:
        raise StitchError(
            f"These predictions imply a {width:,} x {height:,} grid at "
            f"{cell} px per cell, which is far too large to be right. The "
            f"tile positions are probably not on a regular grid — set the "
            f"cell size explicitly.")
    mask = np.zeros((height, width), dtype=bool)

    for tile in tiles:
        size = max(tile.size_level0, cell)
        x0 = int((tile.x - origin_x) // cell)
        y0 = int((tile.y - origin_y) // cell)
        x1 = int(math.ceil((tile.x - origin_x + size) / cell))
        y1 = int(math.ceil((tile.y - origin_y + size) / cell))
        mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
    return mask, origin_x, origin_y


def trace_outline(component: np.ndarray, cell: int, origin_x: int, origin_y: int,
                  tolerance: float) -> list[Point] | None:
    """The outer boundary of one component, in level-0 slide coordinates.

    Walks cell edges exactly rather than using ``skimage.find_contours``.
    Marching squares interpolates, so it chamfers every corner: the contour of
    a 2x2 block comes back as an octagon, and the traced region is then 62% of
    the area it should be. Tiles are squares and their union has square
    corners, so the boundary is walked directly.

    Interior holes are not carved out: the annotation is the outline of the
    region, and a gap of unpredicted tiles inside a lesion is far more often a
    confidence dropout than a genuine void.
    """
    loops = _boundary_loops(component)
    if not loops:
        return None
    # Largest enclosed area is the outer boundary; the rest are holes.
    outer = max(loops, key=lambda loop: abs(_signed_area(loop)))
    outer = _drop_collinear(outer)
    if tolerance > 0 and len(outer) > 4:
        outer = _simplify(outer, tolerance)
    if len(outer) < 3:
        return None
    return [Point(float(origin_x + col * cell), float(origin_y + row * cell))
            for row, col in outer]


def _boundary_loops(component: np.ndarray) -> list[list[tuple[int, int]]]:
    """Closed loops of cell-corner coordinates around a filled region.

    Each filled cell contributes a directed edge for every side facing empty
    space, wound so the loops chain head-to-tail. Corners are exact integers,
    which is the whole reason for doing it this way.
    """
    filled = np.asarray(component, dtype=bool)
    height, width = filled.shape
    padded = np.pad(filled, 1)

    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(height):
        for c in range(width):
            if not filled[r, c]:
                continue
            pr, pc = r + 1, c + 1
            if not padded[pr - 1, pc]:              # nothing above
                edges.setdefault((r, c), []).append((r, c + 1))
            if not padded[pr, pc + 1]:              # nothing to the right
                edges.setdefault((r, c + 1), []).append((r + 1, c + 1))
            if not padded[pr + 1, pc]:              # nothing below
                edges.setdefault((r + 1, c + 1), []).append((r + 1, c))
            if not padded[pr, pc - 1]:              # nothing to the left
                edges.setdefault((r + 1, c), []).append((r, c))

    loops: list[list[tuple[int, int]]] = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        current = start
        while True:
            outgoing = edges.get(current)
            if not outgoing:
                break
            nxt = outgoing.pop()
            if not outgoing:
                del edges[current]
            if nxt == start:
                break
            loop.append(nxt)
            current = nxt
        if len(loop) >= 4:
            loops.append(loop)
    return loops


def _signed_area(loop: Sequence[tuple[int, int]]) -> float:
    """Shoelace area of a loop of (row, col) points."""
    total = 0.0
    for (r1, c1), (r2, c2) in zip(loop, list(loop[1:]) + [loop[0]]):
        total += c1 * r2 - c2 * r1
    return total / 2.0


def _drop_collinear(loop: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse runs along one cell edge into a single straight segment."""
    if len(loop) < 3:
        return loop
    kept: list[tuple[int, int]] = []
    count = len(loop)
    for i in range(count):
        prev = loop[i - 1]
        here = loop[i]
        nxt = loop[(i + 1) % count]
        cross = ((here[0] - prev[0]) * (nxt[1] - here[1])
                 - (here[1] - prev[1]) * (nxt[0] - here[0]))
        if cross != 0:
            kept.append(here)
    return kept or loop


def _simplify(loop: list[tuple[int, int]], tolerance: float) -> list[tuple[int, int]]:
    """Douglas-Peucker, which flattens staircases but keeps real corners.

    A one-cell step along a diagonal deviates about half a cell from the chord
    and disappears; the right angle of a genuinely rectangular region deviates
    much further and stays.
    """
    from skimage.measure import approximate_polygon

    closed = np.array(loop + [loop[0]], dtype=float)
    simplified = approximate_polygon(closed, tolerance=tolerance)
    if len(simplified) < 4:
        return loop
    points = [(float(r), float(c)) for r, c in simplified]
    if points[0] == points[-1]:
        points.pop()
    return points if len(points) >= 3 else loop
