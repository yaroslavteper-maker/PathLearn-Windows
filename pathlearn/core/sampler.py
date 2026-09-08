"""Patch sampling — ported from ``Models/PatchSampler.swift``.

Turns a polygon into a list of patch origins on a regular grid.  Only patches
whose **centre** is inside the polygon and whose rect fits inside the slide are
emitted.

Difference from the Swift: it took ``polygonOverlayY`` and mirrored Y to reach
slide-data space.  Here polygons are already in the one true space
(:mod:`pathlearn.coords`), so there is no mirror — everything in and out is
level-0, top-left origin, Y down.

The grid is anchored at the polygon's bounding box, not at a global origin, so
two annotations sample on independent lattices.  That matches the Swift and
keeps a patch grid stable when unrelated annotations change.

It also always reaches the far side of that bounding box.  Stepping by stride
alone stops at the last whole stride, which left a band of up to one stride
along the right and bottom of every region whose extent was not an exact
multiple — visible in the app as a heatmap the shape of the annotation that
came up short of it.  ``_grid`` snaps a final origin flush to the far edge,
which overlaps its neighbour and is worth it.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from ..models.annotation import Point


def sample_patches(polygon: Sequence[Point],
                   slide_width: int,
                   slide_height: int,
                   patch_size_level: int,
                   stride_level: int,
                   level_downsample: float) -> list[tuple[int, int]]:
    """Grid origins (level-0 px) for patches covering *polygon*.

    *patch_size_level* and *stride_level* are in the sampled level's pixels;
    they are scaled by *level_downsample* to reach level-0, matching
    ``patchSizeLevel0 = round(patchSizeLevel * downsample)``.
    """
    if len(polygon) < 3 or patch_size_level <= 0 or stride_level <= 0 or level_downsample <= 0:
        return []

    xs = np.fromiter((p[0] for p in polygon), dtype=np.float64, count=len(polygon))
    ys = np.fromiter((p[1] for p in polygon), dtype=np.float64, count=len(polygon))

    patch_level0 = int(round(patch_size_level * level_downsample))
    stride_level0 = int(round(stride_level * level_downsample))
    if patch_level0 <= 0 or stride_level0 <= 0:
        return []

    x_start = int(math.floor(xs.min()))
    y_start = int(math.floor(ys.min()))
    x_end = int(math.ceil(xs.max())) - patch_level0
    y_end = int(math.ceil(ys.max())) - patch_level0
    if x_end < x_start or y_end < y_start:
        return []

    grid_x = _grid(x_start, x_end, stride_level0)
    grid_y = _grid(y_start, y_end, stride_level0)
    if grid_x.size == 0 or grid_y.size == 0:
        return []

    # Keep only origins whose whole rect lies on the slide.
    grid_x = grid_x[(grid_x >= 0) & (grid_x + patch_level0 <= slide_width)]
    grid_y = grid_y[(grid_y >= 0) & (grid_y + patch_level0 <= slide_height)]
    if grid_x.size == 0 or grid_y.size == 0:
        return []

    origin_x, origin_y = np.meshgrid(grid_x, grid_y)
    origin_x = origin_x.ravel()
    origin_y = origin_y.ravel()

    half = patch_level0 / 2.0
    inside = points_in_polygon(origin_x + half, origin_y + half, xs, ys)
    return [(int(x), int(y)) for x, y in zip(origin_x[inside], origin_y[inside])]


def _grid(start: int, end: int, stride: int) -> np.ndarray:
    """Origins from *start* to *end* inclusive, always reaching *end*.

    ``arange`` alone stops at the last whole stride, so unless the span is an
    exact multiple of the stride the final row or column never reaches *end*
    and a band of up to ``stride - 1`` pixels along the right and bottom edges
    of the region is never sampled at all. On a large annotation that is a
    visible strip of tissue with no heatmap over it, in the shape of the
    region — which is exactly what it looks like in the app.

    Snapping a final origin flush to *end* costs one extra row and column and
    overlaps its neighbour, which is harmless: prediction resolves overlapping
    tiles by confidence, and for extraction an extra edge patch is better than
    a missed one.
    """
    grid = np.arange(start, end + 1, stride, dtype=np.int64)
    if grid.size and grid[-1] < end:
        grid = np.append(grid, np.int64(end))
    return grid

def cancel_origins(origins: Sequence[tuple[int, int]],
                   patch_size_level0: int,
                   subtractive_polygons: Iterable[Sequence[Point]]) -> list[tuple[int, int]]:
    """Drop origins whose patch **centre** falls inside any subtractive polygon.

    Used to carve lumen / empty space out of a primary annotation before the
    patches ever reach the bank.
    """
    origins = list(origins)
    polygons = [p for p in subtractive_polygons if len(p) >= 3]
    if not origins or not polygons:
        return origins

    ox = np.fromiter((o[0] for o in origins), dtype=np.float64, count=len(origins))
    oy = np.fromiter((o[1] for o in origins), dtype=np.float64, count=len(origins))
    half = patch_size_level0 / 2.0
    cx, cy = ox + half, oy + half

    cancelled = np.zeros(len(origins), dtype=bool)
    for polygon in polygons:
        xs = np.fromiter((p[0] for p in polygon), dtype=np.float64, count=len(polygon))
        ys = np.fromiter((p[1] for p in polygon), dtype=np.float64, count=len(polygon))
        cancelled |= points_in_polygon(cx, cy, xs, ys)

    return [o for o, drop in zip(origins, cancelled) if not drop]


def estimated_patch_count(polygon: Sequence[Point],
                          stride_level: int,
                          level_downsample: float) -> int:
    """Cheap bbox/stride² upper bound, for UI preview only.

    Deliberately ignores the polygon shape — it is an estimate shown while the
    user drags a slider, not a promise.
    """
    if len(polygon) < 3 or stride_level <= 0 or level_downsample <= 0:
        return 0
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    stride_level0 = stride_level * level_downsample
    nx = max(0, int(math.floor((max(xs) - min(xs)) / stride_level0)))
    ny = max(0, int(math.floor((max(ys) - min(ys)) / stride_level0)))
    return nx * ny


def points_in_polygon(px: np.ndarray, py: np.ndarray,
                      poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting test for many points against one polygon.

    Semantics match the scalar Swift ``pointInPolygon`` (and
    ``Annotation.contains``): a crossing counts when the edge straddles *py* per
    the half-open rule ``(yi > y) != (yj > y)``, which makes shared vertices and
    horizontal edges behave consistently.
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    if px.size == 0:
        return np.zeros(0, dtype=bool)

    # Edge i goes from vertex j (previous) to vertex i, as in the Swift loop.
    xi, yi = poly_x[:, None], poly_y[:, None]
    xj, yj = np.roll(poly_x, 1)[:, None], np.roll(poly_y, 1)[:, None]

    straddles = (yi > py[None, :]) != (yj > py[None, :])
    # Guard the divide: where it does not straddle, the value is masked out
    # anyway, so a dummy denominator avoids a spurious warning.
    dy = np.where(straddles, yj - yi, 1.0)
    x_cross = (xj - xi) * (py[None, :] - yi) / dy + xi
    crossings = straddles & (px[None, :] < x_cross)
    return (np.count_nonzero(crossings, axis=0) % 2).astype(bool)
