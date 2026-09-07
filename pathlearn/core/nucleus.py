"""Nucleus segmentation — from ``Models/NucleusSegmenter.swift``.

The geometry descriptors depend only on the :class:`NucleusSegmenter` protocol,
so a learned instance segmenter (StarDist / HoVer-Net via ONNX) can replace the
classical backend without touching callers.  That swap is the recommended next
lever if grading underperforms — the classical detector under-segments touching
nuclei and therefore under-measures nucleus *size*
(``04-DESIGN-DECISIONS.md`` §2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .components import label_blobs
from .stain import StainResult

DEFAULT_THRESHOLD_K = 0.5
DEFAULT_MIN_NUCLEUS_AREA = 6


@dataclass(frozen=True, slots=True)
class DetectedNucleus:
    area: int      # pixels at the analysis level
    cx: float
    cy: float


@dataclass(frozen=True, slots=True)
class NucleiResult:
    nuclei: list[DetectedNucleus]
    #: Total thresholded nucleus pixels — counted **before** the min-area
    #: filter, because it feeds the crowding descriptor rather than instance
    #: counts.  The Swift does the same.
    nucleus_pixel_count: int


@runtime_checkable
class NucleusSegmenter(Protocol):
    def segment(self, stain: StainResult, tissue_mask: np.ndarray) -> NucleiResult:
        """Detect nuclei where *tissue_mask* is True."""
        ...


class ClassicalNucleusSegmenter:
    """Threshold the haematoxylin channel, then connected components.

    The threshold is adaptive — ``mean + k*std`` over tissue pixels only — so it
    tolerates staining and scanner variation.  Fast and dependency-free.
    """

    def __init__(self, threshold_k: float = DEFAULT_THRESHOLD_K,
                 min_nucleus_area: int = DEFAULT_MIN_NUCLEUS_AREA) -> None:
        self.threshold_k = threshold_k
        self.min_nucleus_area = min_nucleus_area

    def segment(self, stain: StainResult, tissue_mask: np.ndarray) -> NucleiResult:
        tissue_mask = np.asarray(tissue_mask, dtype=bool)
        hematoxylin = stain.hematoxylin
        if tissue_mask.shape != hematoxylin.shape:
            raise ValueError("tissue_mask shape does not match the stain result")

        tissue_values = hematoxylin[tissue_mask]
        if tissue_values.size == 0:
            return NucleiResult([], 0)

        threshold = float(tissue_values.mean() + self.threshold_k * tissue_values.std())
        nucleus_mask = tissue_mask & (hematoxylin >= threshold)
        pixel_count = int(np.count_nonzero(nucleus_mask))

        blobs = label_blobs(nucleus_mask, min_area=self.min_nucleus_area)
        nuclei = [DetectedNucleus(area=b.area, cx=b.cx, cy=b.cy) for b in blobs]
        return NucleiResult(nuclei, pixel_count)


def nearest_neighbour_distances(nuclei: list[DetectedNucleus], mpp: float,
                                cap: int = 2000) -> np.ndarray:
    """Nearest-neighbour centroid distances in microns.

    When there are more than *cap* nuclei the largest ones are kept, matching
    the Swift's ``sorted { $0.area > $1.area }.prefix(cap)`` — this bounds an
    O(n^2) pass while keeping the best-segmented objects.
    """
    if len(nuclei) < 2:
        return np.zeros(0, dtype=np.float64)

    subset = nuclei
    if len(nuclei) > cap:
        subset = sorted(nuclei, key=lambda n: n.area, reverse=True)[:cap]

    coords = np.array([(n.cx, n.cy) for n in subset], dtype=np.float64)
    # Chunked to keep the pairwise matrix bounded at large cap values.
    out = np.empty(len(coords), dtype=np.float64)
    chunk = 512
    for start in range(0, len(coords), chunk):
        block = coords[start:start + chunk]
        d2 = ((block[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
        # Exclude self-distance without disturbing genuine coincident points.
        rows = np.arange(block.shape[0])
        d2[rows, start + rows] = np.inf
        out[start:start + block.shape[0]] = np.sqrt(d2.min(axis=1))

    return out * mpp
