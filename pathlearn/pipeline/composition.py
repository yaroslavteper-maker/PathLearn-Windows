"""What fraction of the predicted tissue each class occupies.

After a prediction run the question a pathologist actually asks is not "how
many tiles" but "how much of this lesion is PanIN-2".  This turns a
``PredictionSet`` into that breakdown.

WHAT THE DENOMINATOR IS — READ THIS BEFORE QUOTING A NUMBER
===========================================================
The total here is **the tissue that was predicted over**, which is not the same
as the tissue on the slide:

* Prediction only samples inside the regions you chose, so a slide-wide
  percentage is only slide-wide if you predicted over the whole slide.
* The sampler already dropped tiles that were too white or too nuclei-poor to
  be tissue.  Those tiles are absent from the ``PredictionSet`` entirely and
  cannot be counted here.  That is the behaviour you want — blank glass in the
  denominator would deflate every class by however much background the region
  happened to enclose — but it does mean this is a percentage of *sampled
  tissue*, not of *slide area*.
* Tiles below the confidence threshold are excluded and reported separately
  rather than silently dropped, because a run where a third of the tissue was
  too marginal to call is a different result from one where it was not.

Classes hidden in the overlay are **still counted**.  Hiding is a drawing
choice; letting a checkbox move a quantitative result would be a trap.

TILE SHARE VERSUS AREA SHARE
============================
Both are reported and they are not the same number.

*Tile share* is the fraction of classified tiles.  It is exact and needs no
assumptions, but it treats every tile as equal — and a multi-pass run mixes
patch sizes, so a large tile and a small one count the same.

*Area share* rasterises the tiles onto their common grid and measures covered
cells, which is what you want when tiles differ in size or overlap.  Overlap
forces a choice: a cell covered by two tiles of different classes has to go to
one of them, and it goes to the **more confident** tile.  That keeps the class
areas summing to the total instead of double-counting the overlap, which a
naive per-class sum of tile areas does not.

With a single patch size and no overlap the two agree exactly, and a test pins
that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..models.prediction import PatchPrediction, PredictionSet
from .stitch import MAX_RASTER_CELLS, infer_cell_size

log = logging.getLogger(__name__)


class CompositionError(ValueError):
    """Raised when a breakdown cannot be computed at all."""


@dataclass(frozen=True, slots=True)
class ClassShare:
    """One class's slice of the predicted tissue."""

    label: str
    tiles: int
    #: Grid cells this class won outright.
    cells: int
    area_px: float
    tile_fraction: float
    area_fraction: float
    mean_confidence: float
    #: None when the slide does not report a pixel size.
    area_mm2: float | None = None

    @property
    def area_percent(self) -> float:
        return self.area_fraction * 100.0

    @property
    def tile_percent(self) -> float:
        return self.tile_fraction * 100.0


@dataclass
class CompositionReport:
    """The full breakdown, in the order a reader wants it: largest first."""

    shares: list[ClassShare] = field(default_factory=list)
    total_tiles: int = 0
    total_cells: int = 0
    total_area_px: float = 0.0
    total_area_mm2: float | None = None
    cell_size: int = 0
    #: Tiles present in the set but under the threshold, so not counted.
    excluded_low_confidence: int = 0
    min_confidence: float = 0.0
    mpp: float | None = None
    slide_name: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.shares

    def share_for(self, label: str) -> ClassShare | None:
        return next((s for s in self.shares if s.label == label), None)

    def summary(self) -> str:
        if self.is_empty:
            if self.excluded_low_confidence:
                return (f"No tiles at or above {self.min_confidence:.2f} "
                        f"confidence — {self.excluded_low_confidence} were below it.")
            return "No predictions to break down."
        parts = ", ".join(f"{s.label} {s.area_percent:.1f}%" for s in self.shares)
        text = f"{self.total_tiles} tiles over {self.area_text()} — {parts}"
        if self.excluded_low_confidence:
            text += (f"  ({self.excluded_low_confidence} tiles below "
                     f"{self.min_confidence:.2f} excluded)")
        return text + "."

    def area_text(self) -> str:
        return format_area(self.total_area_px, self.total_area_mm2)

    def to_csv(self) -> str:
        """A spreadsheet-ready table, with the denominator stated in it.

        The provenance rows are part of the file on purpose: a bare percentage
        column outlives the memory of what it was a percentage *of*.
        """
        lines = ["Class,Tiles,Tile %,Cells,Area (px^2),Area (mm^2),Area %,"
                 "Mean confidence"]
        for share in self.shares:
            mm2 = "" if share.area_mm2 is None else f"{share.area_mm2:.4f}"
            lines.append(
                f"{_csv(share.label)},{share.tiles},{share.tile_percent:.2f},"
                f"{share.cells},{share.area_px:.0f},{mm2},"
                f"{share.area_percent:.2f},{share.mean_confidence:.4f}")
        total_mm2 = ("" if self.total_area_mm2 is None
                     else f"{self.total_area_mm2:.4f}")
        lines.append(f"Total,{self.total_tiles},100.00,{self.total_cells},"
                     f"{self.total_area_px:.0f},{total_mm2},100.00,")
        mpp_text = "unknown" if self.mpp is None else f"{self.mpp:.4f}"
        lines += [
            "",
            f"# Slide,{_csv(self.slide_name)}",
            "# Denominator,Predicted tissue only - not whole-slide area",
            f"# Confidence threshold,{self.min_confidence:.2f}",
            f"# Tiles below threshold (excluded),{self.excluded_low_confidence}",
            f"# Grid cell (level-0 px),{self.cell_size}",
            f"# Microns per pixel,{mpp_text}",
        ]
        return "\n".join(lines) + "\n"


def format_area(area_px: float, area_mm2: float | None) -> str:
    """One area, in the largest unit that still shows a non-zero figure.

    Fixed two decimals would render anything under 5,000 px² as "0.00 mm2",
    which reads as *no area at all* rather than *a small one* — so the
    precision follows the magnitude, and small areas drop to um2 outright.
    """
    if area_mm2 is None:
        return f"{area_px:,.0f} px²"
    if area_mm2 >= 1.0:
        return f"{area_mm2:.2f} mm²"
    if area_mm2 >= 0.01:
        return f"{area_mm2:.3f} mm²"
    return f"{area_mm2 * 1_000_000:,.0f} µm²"


def _csv(value: str) -> str:
    text = str(value)
    if any(character in text for character in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def composition(predictions: PredictionSet, *,
                min_confidence: float | None = None,
                mpp: float | None = None,
                cell_size: int | None = None) -> CompositionReport:
    """Break the predicted tissue down by class.

    *min_confidence* defaults to the set's own threshold, so the table matches
    what the heatmap is showing.  *mpp* is microns per level-0 pixel; without
    it the report is in pixels only.
    """
    threshold = (predictions.min_confidence if min_confidence is None
                 else float(min_confidence))
    kept = [p for p in predictions.predictions if p.confidence >= threshold]
    excluded = len(predictions.predictions) - len(kept)

    report = CompositionReport(
        total_tiles=len(kept), excluded_low_confidence=excluded,
        min_confidence=threshold, mpp=mpp,
        slide_name=predictions.slide_path)
    if not kept:
        return report

    cell = cell_size or infer_cell_size(kept)
    if cell <= 0:
        raise CompositionError(
            "Could not work out a tile grid from these predictions.")
    report.cell_size = cell

    owner, labels = _assign_cells(kept, cell)
    counts = np.bincount(owner.ravel(), minlength=len(labels) + 1)
    # Index 0 is "no tile covered this cell" — the gaps between the clumps.
    total_cells = int(counts[1:].sum())
    if total_cells == 0:
        return report

    cell_area = float(cell) * float(cell)
    report.total_cells = total_cells
    report.total_area_px = total_cells * cell_area
    if mpp:
        report.total_area_mm2 = _to_mm2(report.total_area_px, mpp)

    tiles_by_label: dict[str, list[PatchPrediction]] = {}
    for tile in kept:
        tiles_by_label.setdefault(tile.label, []).append(tile)

    shares: list[ClassShare] = []
    for label, tiles in tiles_by_label.items():
        cells = int(counts[labels.index(label) + 1])
        area_px = cells * cell_area
        shares.append(ClassShare(
            label=label,
            tiles=len(tiles),
            cells=cells,
            area_px=area_px,
            tile_fraction=len(tiles) / len(kept),
            area_fraction=cells / total_cells,
            mean_confidence=float(np.mean([t.confidence for t in tiles])),
            area_mm2=_to_mm2(area_px, mpp) if mpp else None))

    # Largest first, then alphabetically so ties are stable between runs.
    shares.sort(key=lambda share: (-share.area_fraction, share.label))
    report.shares = shares
    return report


def _to_mm2(area_px: float, mpp: float) -> float:
    """Level-0 px² to mm²: 1 px = mpp µm, and 1 mm² = 1e6 µm²."""
    return area_px * (mpp * mpp) / 1_000_000.0


def _assign_cells(tiles: Sequence[PatchPrediction],
                  cell: int) -> tuple[np.ndarray, list[str]]:
    """Paint tiles onto a grid; each cell ends up owned by one class.

    Returns a grid of 1-based class indices (0 = uncovered) and the label list
    those index into.  Overlapping tiles are painted in ascending confidence
    order so the most confident call is the one left standing — the cheapest
    correct way to resolve overlap without an argmax per cell.
    """
    origin_x = min(t.x for t in tiles)
    origin_y = min(t.y for t in tiles)
    far_x = max(t.x + max(t.size_level0, cell) for t in tiles)
    far_y = max(t.y + max(t.size_level0, cell) for t in tiles)
    columns = max(1, math.ceil((far_x - origin_x) / cell))
    rows = max(1, math.ceil((far_y - origin_y) / cell))
    if rows * columns > MAX_RASTER_CELLS:
        raise CompositionError(
            f"That would need a {rows} x {columns} grid — "
            "the tile size looks wrong.")

    labels: list[str] = []
    for tile in tiles:
        if tile.label not in labels:
            labels.append(tile.label)

    owner = np.zeros((rows, columns), dtype=np.int32)
    for tile in sorted(tiles, key=lambda t: t.confidence):
        column = (tile.x - origin_x) // cell
        row = (tile.y - origin_y) // cell
        span = max(1, round(max(tile.size_level0, cell) / cell))
        owner[row:row + span, column:column + span] = labels.index(tile.label) + 1
    return owner, labels
