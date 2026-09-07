"""Connected components — from ``Models/ConnectedComponents.swift``.

4-connected labelling over a boolean mask, returning area, centroid and
perimeter per blob.  Reused for both nuclei (haematoxylin mask) and lumina
(whitespace mask).

The Swift did an explicit flood fill; ``scipy.ndimage.label`` does the same
labelling far faster.  The one definition worth stating precisely is
**perimeter**, which the Swift computes as *the number of pixels in the blob
that have at least one 4-neighbour outside the mask, counting the image border
as outside*.  That is a pixel count, not an arc length, and the geometry
descriptors' circularity term depends on it — so it is reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

#: 4-connectivity, matching the Swift's left/right/up/down traversal.
_STRUCTURE = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)


@dataclass(frozen=True, slots=True)
class Blob:
    """One connected region."""

    area: int          # pixel count
    cx: float          # centroid x
    cy: float          # centroid y
    perimeter: int     # boundary pixel count (>= 1)


def label_blobs(mask: np.ndarray, min_area: int = 1) -> list[Blob]:
    """Label *mask* and return blobs of at least *min_area* pixels."""
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0 or not mask.any():
        return []

    labels, count = ndimage.label(mask, structure=_STRUCTURE)
    if count == 0:
        return []

    index = np.arange(1, count + 1)
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]

    # Centroids: sum of coordinates per label / area.
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    flat_labels = labels[ys, xs]
    sum_x = np.bincount(flat_labels, weights=xs, minlength=count + 1)[1:]
    sum_y = np.bincount(flat_labels, weights=ys, minlength=count + 1)[1:]

    boundary = _boundary_mask(mask)
    perimeters = np.bincount(labels[boundary], minlength=count + 1)[1:]

    keep = areas >= min_area
    return [
        Blob(area=int(areas[i]),
             cx=float(sum_x[i] / areas[i]),
             cy=float(sum_y[i] / areas[i]),
             perimeter=int(max(perimeters[i], 1)))
        for i in np.flatnonzero(keep)
    ]


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Mask pixels with a 4-neighbour outside the mask, image edge included.

    Padding with ``False`` makes the image border count as outside, which is
    what the Swift's ``else { isBoundary = true }`` branches do.
    """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (padded[:-2, 1:-1]    # up
                & padded[2:, 1:-1]   # down
                & padded[1:-1, :-2]  # left
                & padded[1:-1, 2:])  # right
    return mask & ~interior


def circularity(area: int, perimeter: int) -> float:
    """``4*pi*area / perimeter^2``, capped at 1.

    Low values mean an irregular boundary — the cribriforming proxy in the
    geometry descriptor.
    """
    if perimeter <= 0:
        return 0.0
    return float(min(1.0, 4.0 * np.pi * area / (perimeter * perimeter)))
