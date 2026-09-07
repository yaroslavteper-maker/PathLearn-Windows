"""Lesion **outline** descriptors — the shape of a PaNIN, not its texture.

THE IDEA
========
PaNIN grade is partly a statement about duct *shape*.  A PaNIN-1 duct is a
simple, smooth, roughly convex tube.  As grade rises the epithelium throws up
papillae and cribriform bridges, so the traced outline becomes lobulated,
indented and convoluted.  That information lives in the polygon the annotator
drew, and a 224 px tile crop cannot see it — the tile sees texture inside a
window, never the boundary of the whole lesion.

This module measures the outline directly.  It complements
:mod:`pathlearn.core.geometry`, which measures interior architecture, and has
three properties that matter for this dataset:

* **No pixels are read**, so it is instant and cannot fail on staining.
* **Nothing is gated on nucleus counts**, so it works on every annotation —
  including the ~70% that interior analysis rejects for being single ducts too
  small to hold 20 nuclei.
* **The primary terms are scale-invariant**, so they cannot relearn "bigger
  lesion ⇒ higher grade", which is the confound that collapsed descriptor v1
  (``04-DESIGN-DECISIONS.md`` §2).

DRAWING STYLE MUST NOT LEAK IN
==============================
A lasso-drawn polygon has hundreds of jittery vertices; a click-drawn one has
eight clean ones.  Perimeter, roughness and convexity all respond strongly to
that, so a naive descriptor would partly encode *which tool the user picked* —
and if grades were annotated in different sessions, that becomes a phantom
signal.  Every outline is therefore **resampled to a fixed number of
equally-spaced points** before anything is measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..models.annotation import Annotation, Point

#: Scale-invariant outline descriptors, in fixed order.
SHAPE_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "circularity",          # 0  4*pi*A / P^2 — 1 is a circle, lower is convoluted
    "solidity",             # 1  A / convex-hull A — indentation / papillae
    "convexity",            # 2  hull perimeter / perimeter — boundary excess
    "elongation",           # 3  1 - minor/major axis (PCA)
    "extent",               # 4  A / bounding-box A
    "radialCV",             # 5  CV of centroid distance — lobulation
    "radialRoughness",      # 6  mean |d r| along the outline, scale-normalised
    "concavityFraction",    # 7  share of resampled points lying off the hull
    "concavityDepthMean",   # 8  mean indentation depth, scale-normalised
    "boundaryComplexity",   # 9  perimeter / smoothed perimeter
    "lobeCount",            # 10 radial maxima per unit boundary — papillae proxy
    "eccentricity",         # 11 sqrt(1 - (minor/major)^2)
)

SHAPE_DIMENSION = len(SHAPE_DESCRIPTOR_NAMES)

#: Size terms, kept separate because they are **not** scale-invariant.  Useful
#: to inspect, dangerous to train on — see the v1 collapse.
SIZE_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "areaUm2",
    "perimeterUm",
    "equivalentDiameterUm",
)

#: Bump when the descriptor set or its semantics change.
SHAPE_VERSION = 1

#: Outlines are resampled to this many equally-spaced points before measurement,
#: so lasso-vs-polygon drawing style cannot leak into the numbers.
RESAMPLE_POINTS = 256

#: Window (in resampled points) used to smooth the outline for the complexity
#: and lobe terms.  About 5% of the boundary.
SMOOTH_WINDOW = 13


@dataclass(frozen=True, slots=True)
class ShapeResult:
    features: np.ndarray
    size: np.ndarray

    def as_dict(self) -> dict[str, float]:
        out = {n: float(v) for n, v in zip(SHAPE_DESCRIPTOR_NAMES, self.features)}
        out.update({n: float(v) for n, v in zip(SIZE_DESCRIPTOR_NAMES, self.size)})
        return out


def describe_shape(annotation: Annotation, mpp: float = 1.0) -> ShapeResult | None:
    """Outline descriptors for *annotation*, or ``None`` if it is degenerate.

    *mpp* converts the size terms to microns; the scale-invariant block is
    unaffected by it.
    """
    points = np.array([(p.x, p.y) for p in annotation.points], dtype=np.float64)
    if points.shape[0] < 3:
        return None

    area_px = _polygon_area(points)
    if area_px <= 0:
        return None

    outline = _resample_closed(points, RESAMPLE_POINTS)
    if outline is None:
        return None

    perimeter_px = _closed_perimeter(outline)
    if perimeter_px <= 0:
        return None

    hull = _convex_hull(outline)
    hull_area = _polygon_area(hull) if hull.shape[0] >= 3 else area_px
    hull_perimeter = _closed_perimeter(hull) if hull.shape[0] >= 3 else perimeter_px

    centroid = outline.mean(axis=0)
    radii = np.linalg.norm(outline - centroid, axis=1)
    mean_radius = float(radii.mean()) or 1.0

    major, minor = _principal_axes(outline)
    smoothed = _smooth_closed(outline, SMOOTH_WINDOW)
    smoothed_perimeter = _closed_perimeter(smoothed) or perimeter_px

    # Indentation: how far each outline point sits inside the convex hull,
    # normalised by lesion size so it stays scale-free.
    depths = _hull_depths(outline, hull)
    off_hull = depths > (0.01 * mean_radius)

    values = [
        min(1.0, 4.0 * math.pi * area_px / (perimeter_px ** 2)),
        min(1.0, area_px / hull_area) if hull_area > 0 else 1.0,
        min(1.0, hull_perimeter / perimeter_px) if perimeter_px > 0 else 1.0,
        1.0 - (minor / major) if major > 0 else 0.0,
        area_px / _bounding_box_area(outline) if _bounding_box_area(outline) > 0 else 0.0,
        float(radii.std() / mean_radius),
        float(np.abs(np.diff(np.concatenate([radii, radii[:1]]))).mean() / mean_radius),
        float(off_hull.mean()),
        float(depths.mean() / mean_radius),
        float(perimeter_px / smoothed_perimeter) if smoothed_perimeter > 0 else 1.0,
        _lobe_count(radii),
        math.sqrt(max(0.0, 1.0 - (minor / major) ** 2)) if major > 0 else 0.0,
    ]

    features = np.asarray(values, dtype=np.float64)
    features = np.where(np.isfinite(features), features, 0.0).astype(np.float32)

    size = np.asarray([
        area_px * mpp * mpp,
        perimeter_px * mpp,
        2.0 * math.sqrt(area_px / math.pi) * mpp,
    ], dtype=np.float32)

    return ShapeResult(features, size)


# -- geometry helpers -----------------------------------------------------

def _polygon_area(points: np.ndarray) -> float:
    """Shoelace area, always positive."""
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _closed_perimeter(points: np.ndarray) -> float:
    diffs = np.diff(np.vstack([points, points[:1]]), axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray | None:
    """Equally-spaced points around the closed outline.

    This is what makes the descriptor independent of how the polygon was drawn:
    a 400-vertex lasso trace and an 8-vertex clicked polygon of the same shape
    resample to the same outline.
    """
    closed = np.vstack([points, points[:1]])
    segment = np.hypot(*np.diff(closed, axis=0).T)
    total = segment.sum()
    if total <= 0:
        return None
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    targets = np.linspace(0.0, total, count, endpoint=False)
    x = np.interp(targets, cumulative, closed[:, 0])
    y = np.interp(targets, cumulative, closed[:, 1])
    return np.column_stack([x, y])


def _smooth_closed(points: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average — the reference for boundary complexity."""
    if window < 3 or points.shape[0] < window:
        return points
    kernel = np.ones(window) / window
    padded = np.vstack([points[-window:], points, points[:window]])
    x = np.convolve(padded[:, 0], kernel, mode="same")[window:-window]
    y = np.convolve(padded[:, 1], kernel, mode="same")[window:-window]
    return np.column_stack([x, y])


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, counter-clockwise."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]
    if ordered.shape[0] < 3:
        return ordered

    def build(sequence: np.ndarray) -> list:
        chain: list = []
        for point in sequence:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    lower = build(ordered)
    upper = build(ordered[::-1])
    return np.array(lower[:-1] + upper[:-1]) if len(lower) + len(upper) > 3 else ordered


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull_depths(outline: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Distance from each outline point to the convex hull boundary."""
    if hull.shape[0] < 3:
        return np.zeros(outline.shape[0])
    starts = hull
    ends = np.roll(hull, -1, axis=0)
    edge = ends - starts
    length2 = np.einsum("ij,ij->i", edge, edge)
    length2[length2 == 0] = 1e-12

    # Point-to-segment distance for every (point, hull edge) pair.
    delta = outline[:, None, :] - starts[None, :, :]
    t = np.clip(np.einsum("pij,ij->pi", delta, edge) / length2, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * edge[None, :, :]
    return np.linalg.norm(outline[:, None, :] - closest, axis=2).min(axis=1)


def _principal_axes(points: np.ndarray) -> tuple[float, float]:
    """Major and minor axis lengths from the covariance eigenvalues."""
    centred = points - points.mean(axis=0)
    covariance = np.cov(centred.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    minor, major = math.sqrt(eigenvalues[0]), math.sqrt(eigenvalues[1])
    return (major or 1e-9), minor


def _bounding_box_area(points: np.ndarray) -> float:
    span = points.max(axis=0) - points.min(axis=0)
    return float(span[0] * span[1])


def _lobe_count(radii: np.ndarray) -> float:
    """Radial maxima per unit boundary — a papillae / lobulation proxy.

    Counted on a smoothed radius profile so pixel-level jitter does not read as
    dozens of lobes, and normalised by point count so it stays scale-free.
    """
    if radii.size < 8:
        return 0.0
    window = max(3, radii.size // 32)
    kernel = np.ones(window) / window
    padded = np.concatenate([radii[-window:], radii, radii[:window]])
    smooth = np.convolve(padded, kernel, mode="same")[window:-window]

    # A lobe apex is the local maximum of its neighbourhood AND rises far
    # enough above the surrounding trough to be a real lobe rather than ripple.
    #
    # Prominence must be measured against the local *minimum*. Comparing a peak
    # against its neighbouring points instead fails completely: on a smoothed
    # profile a peak exceeds its own shoulders by a tiny margin, so any
    # meaningful threshold rejects everything. That bug scored a clean 6-lobed
    # star as zero lobes.
    span = max(2, radii.size // 16)
    offsets = [o for o in range(-span, span + 1) if o != 0]
    shifted = [np.roll(smooth, o) for o in offsets]
    neighbourhood_max = np.maximum.reduce(shifted)
    neighbourhood_min = np.minimum.reduce(shifted + [smooth])

    prominence = 0.05 * (smooth.mean() or 1.0)
    peaks = (smooth >= neighbourhood_max) & (smooth - neighbourhood_min >= prominence)
    return float(peaks.sum()) / radii.size * 100.0


def shape_matrix(annotations: Sequence[Annotation], mpp: float = 1.0
                 ) -> tuple[np.ndarray, list[str], list[Annotation]]:
    """Shape descriptors for many annotations, skipping degenerate ones."""
    rows, labels, kept = [], [], []
    for annotation in annotations:
        if annotation.is_subtractive:
            continue
        result = describe_shape(annotation, mpp)
        if result is None:
            continue
        rows.append(result.features)
        labels.append(annotation.classification)
        kept.append(annotation)
    if not rows:
        return np.zeros((0, SHAPE_DIMENSION), dtype=np.float32), [], []
    return np.stack(rows).astype(np.float32), labels, kept
