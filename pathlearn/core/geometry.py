"""Handcrafted architectural descriptors — from ``Models/PathologyGeometry.swift``.

A compact 14-D vector per **whole annotation**, capturing the features that
separate PaNIN grades — cribriforming, luminal complexity, nuclear crowding,
pleomorphism, loss of order — which a single tile cannot see
(``04-DESIGN-DECISIONS.md`` §2).

FIXED PHYSICAL RESOLUTION — DO NOT SKIP
=======================================
Every annotation is analysed at the pyramid level nearest ``target_mpp``
(~1 µm/px), tiled into fixed-size windows.  Descriptor **version 1** read each
annotation at "whatever level fits 2048 px", so larger lesions (typically
PaNIN-3) were read coarser and µm-normalisation inflated their absolute
descriptors.  The classifier duly learned "big nuclei ⇒ PaNIN-3" and collapsed —
the same failure mode as the deep features it was meant to fix.  Version 2 fixes
the resolution instead.  Records whose version differs are rejected by the
trainer rather than silently mixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..io.slide import SlideImage
from ..models.annotation import Annotation, Point
from .components import circularity, label_blobs
from .nucleus import (ClassicalNucleusSegmenter, NucleusSegmenter,
                      nearest_neighbour_distances)
from .sampler import points_in_polygon
from .stain import deconvolve

#: Ordered descriptor names — indices match the produced vector.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "nucleiPerMm2",           # 0  nuclear density
    "nuclearAreaMeanUm2",     # 1  nuclear size
    "nuclearAreaCV",          # 2  pleomorphism
    "nuclearPixelFraction",   # 3  crowding (nucleus px / tissue px)
    "nnDistMeanUm",           # 4  mean nearest-neighbour spacing
    "nnDistCV",               # 5  spacing disorder
    "lumenAreaFraction",      # 6  luminal space fraction
    "lumenCountPerMm2",       # 7  lumen density (cribriforming proxy)
    "lumenSizeMeanUm2",       # 8  mean lumen size
    "lumenSizeCV",            # 9  lumen size variability
    "lumenCircularityMean",   # 10 low = irregular / cribriform boundaries
    "lumenCircularityCV",     # 11 lumen shape variability
    "eosinToHemaRatio",       # 12 cytoplasm vs nuclei
    "tissueFraction",         # 13 tissue px / inside-polygon px
)

DIMENSION = len(DESCRIPTOR_NAMES)

#: Bump when the descriptor set or semantics change, so stale bank rows can be
#: detected.
#:
#: * v2 = fixed-resolution windowing (replacing v1's "whatever level fits").
#: * v3 = the level is now chosen NEAREST the target mpp. v2 used OpenSlide's
#:   best_level_for_downsample, which rounds toward finer and so analysed at
#:   ~0.25 um/px instead of 1.0 on typical slides — measuring chromatin rather
#:   than nuclei. v2 records are not comparable with v3 and are rejected.
VERSION = 3


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    #: Analysis resolution in µm/px.  ~1.0 keeps nuclei resolvable (~7 px)
    #: while a window still spans whole glands.
    target_mpp: float = 1.0
    #: Side length of each analysis window, in analysis-level pixels.
    window_px: int = 1024
    #: Max windows per annotation (evenly spaced if more would fit).
    max_windows: int = 16
    #: Level-0 µm/px to assume when the slide omits ``openslide.mpp-x``.
    fallback_mpp: float = 0.25
    #: Minimum lumen blob area (µm²) — filters inter-cell gaps.
    min_lumen_area_um2: float = 40.0
    #: Require at least this many nuclei before emitting a descriptor.
    #:
    #: The Swift used 20. Measured on real PanIN annotations (single duct
    #: cross-sections of 1,500-3,900 um^2, physically holding 10-20 nuclei)
    #: that rejected roughly 70% of them. 10 matches the scale of the data;
    #: below that the per-annotation CV terms get too noisy to mean much.
    min_nuclei: int = 10
    #: Cap for the per-window O(n²) nearest-neighbour pass.
    max_nuclei_for_nn: int = 2000


def describe(slide: SlideImage,
             annotation: Annotation,
             config: GeometryConfig | None = None,
             segmenter: NucleusSegmenter | None = None) -> np.ndarray | None:
    """The 14-D descriptor for *annotation*, or ``None`` if it is unusable.

    Returns ``None`` when the polygon is degenerate or too few nuclei are found
    — a caller should treat that as "no descriptor", never as a zero vector.
    """
    config = config or GeometryConfig()
    segmenter = segmenter or ClassicalNucleusSegmenter()

    if len(annotation.points) < 3:
        return None

    poly_x = np.array([p.x for p in annotation.points], dtype=np.float64)
    poly_y = np.array([p.y for p in annotation.points], dtype=np.float64)
    min_x, max_x = poly_x.min(), poly_x.max()
    min_y, max_y = poly_y.min(), poly_y.max()
    if max_x <= min_x or max_y <= min_y:
        return None

    # -- fixed analysis level ---------------------------------------------
    base_mpp = slide.mpp_x or config.fallback_mpp
    if base_mpp <= 0:
        base_mpp = config.fallback_mpp
    # NEAREST level to the target, not `best_level_for_downsample`.
    #
    # OpenSlide's best_level_for_downsample never returns a level *coarser*
    # than requested. On a typical slide here (0.2527 um/px, downsamples
    # 1/4/16) a 1.0 um/px target asks for downsample 3.96, so it returns
    # level 0 — analysing four times finer than intended. At that scale the
    # classical segmenter resolves sub-nuclear chromatin rather than whole
    # nuclei: measured on real slides it reported ~37,000 "nuclei"/mm^2 of
    # ~1.6 um^2 each, both physically impossible, and the descriptor became
    # noise.
    #
    # The Swift used best_level_for_downsample and inherited this, but
    # `04-DESIGN-DECISIONS.md` §2 says "nearest", and fixed *physical*
    # resolution is the whole point of descriptor v2.
    level = slide.level_for_mpp(config.target_mpp, fallback_mpp=config.fallback_mpp)
    downsample = slide.level_downsamples[level]
    mpp = base_mpp * downsample          # µm per pixel at the analysis level
    px_area_um2 = mpp * mpp

    windows = _plan_windows(slide, config, downsample,
                            min_x, min_y, max_x, max_y, poly_x, poly_y)
    if not windows:
        return None

    acc = _Accumulator()
    min_lumen_px = max(1, int(round(config.min_lumen_area_um2 / px_area_um2)))

    for wx, wy in windows:
        try:
            rgb = slide.read_region(wx, wy, level, config.window_px, config.window_px)
        except Exception:
            continue
        if rgb.size == 0:
            continue

        stain = deconvolve(rgb)
        inside = _inside_mask(wx, wy, stain.height, stain.width, downsample,
                              poly_x, poly_y)
        if not inside.any():
            continue

        tissue = inside & ~stain.is_background
        lumen_raw = inside & stain.is_background

        acc.inside_count += int(np.count_nonzero(inside))
        acc.tissue_count += int(np.count_nonzero(tissue))
        acc.hema_sum += float(stain.hematoxylin[tissue].sum())
        acc.eosin_sum += float(stain.eosin[tissue].sum())

        nuclei_result = segmenter.segment(stain, tissue)
        acc.nucleus_pixel_count += nuclei_result.nucleus_pixel_count
        acc.nucleus_areas_px.extend(n.area for n in nuclei_result.nuclei)
        acc.nn_distances_um.append(
            nearest_neighbour_distances(nuclei_result.nuclei, mpp,
                                        cap=config.max_nuclei_for_nn)
        )

        blobs = label_blobs(lumen_raw, min_area=min_lumen_px)
        acc.lumen_count += len(blobs)
        acc.lumen_pixel_count += sum(b.area for b in blobs)
        acc.lumen_areas_px.extend(b.area for b in blobs)
        acc.lumen_circularities.extend(circularity(b.area, b.perimeter) for b in blobs)

    if (acc.inside_count == 0 or acc.tissue_count == 0
            or len(acc.nucleus_areas_px) < config.min_nuclei):
        return None

    return _assemble(acc, px_area_um2)


def _plan_windows(slide: SlideImage, config: GeometryConfig, downsample: float,
                  min_x: float, min_y: float, max_x: float, max_y: float,
                  poly_x: np.ndarray, poly_y: np.ndarray) -> list[tuple[int, int]]:
    """Tile the bbox into windows whose centre is inside the polygon."""
    window_level0 = int(round(config.window_px * downsample))
    if window_level0 <= 0:
        return []

    slide_w, slide_h = slide.dimensions.width, slide.dimensions.height
    xs = np.arange(int(math.floor(min_x)), int(math.ceil(max_x)), window_level0, dtype=np.int64)
    ys = np.arange(int(math.floor(min_y)), int(math.ceil(max_y)), window_level0, dtype=np.int64)

    windows: list[tuple[int, int]] = []
    if xs.size and ys.size:
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
        half = window_level0 / 2.0
        inside = points_in_polygon(grid_x + half, grid_y + half, poly_x, poly_y)
        windows = [
            (int(np.clip(x, 0, max(0, slide_w - window_level0))),
             int(np.clip(y, 0, max(0, slide_h - window_level0))))
            for x, y in zip(grid_x[inside], grid_y[inside])
        ]

    if not windows:
        # A small annotation may have no window centre inside it; analyse one
        # clamped window so it still gets a descriptor.
        windows = [(int(np.clip(int(math.floor(min_x)), 0, max(0, slide_w - window_level0))),
                    int(np.clip(int(math.floor(min_y)), 0, max(0, slide_h - window_level0))))]

    if len(windows) > config.max_windows:
        step = len(windows) / config.max_windows
        windows = [windows[int(i * step)] for i in range(config.max_windows)]
    return windows


def _inside_mask(window_x: int, window_y: int, height: int, width: int,
                 downsample: float, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Boolean mask of window pixels whose centre lies inside the polygon."""
    px = window_x + (np.arange(width, dtype=np.float64) + 0.5) * downsample
    py = window_y + (np.arange(height, dtype=np.float64) + 0.5) * downsample
    grid_x, grid_y = np.meshgrid(px, py)
    return points_in_polygon(grid_x.ravel(), grid_y.ravel(),
                             poly_x, poly_y).reshape(height, width)


@dataclass
class _Accumulator:
    """Running totals pooled across all windows of one annotation."""

    nucleus_areas_px: list[int] = None
    nn_distances_um: list[np.ndarray] = None
    lumen_areas_px: list[int] = None
    lumen_circularities: list[float] = None
    nucleus_pixel_count: int = 0
    tissue_count: int = 0
    inside_count: int = 0
    lumen_pixel_count: int = 0
    lumen_count: int = 0
    hema_sum: float = 0.0
    eosin_sum: float = 0.0

    def __post_init__(self) -> None:
        self.nucleus_areas_px = []
        self.nn_distances_um = []
        self.lumen_areas_px = []
        self.lumen_circularities = []


def _assemble(acc: _Accumulator, px_area_um2: float) -> np.ndarray:
    tissue_area_mm2 = acc.tissue_count * px_area_um2 / 1_000_000.0

    nucleus_areas_um2 = np.asarray(acc.nucleus_areas_px, dtype=np.float64) * px_area_um2
    nuc_area_mean, nuc_area_cv = _mean_cv(nucleus_areas_um2)

    nn = (np.concatenate(acc.nn_distances_um)
          if acc.nn_distances_um else np.zeros(0, dtype=np.float64))
    nn_mean, nn_cv = _mean_cv(nn)

    lumen_areas_um2 = np.asarray(acc.lumen_areas_px, dtype=np.float64) * px_area_um2
    lumen_mean, lumen_cv = _mean_cv(lumen_areas_um2)

    circ = np.asarray(acc.lumen_circularities, dtype=np.float64)
    circ_mean, circ_cv = _mean_cv(circ)

    values = [
        len(acc.nucleus_areas_px) / tissue_area_mm2 if tissue_area_mm2 > 0 else 0.0,
        nuc_area_mean,
        nuc_area_cv,
        acc.nucleus_pixel_count / acc.tissue_count if acc.tissue_count > 0 else 0.0,
        nn_mean,
        nn_cv,
        acc.lumen_pixel_count / acc.inside_count if acc.inside_count > 0 else 0.0,
        acc.lumen_count / tissue_area_mm2 if tissue_area_mm2 > 0 else 0.0,
        lumen_mean,
        lumen_cv,
        circ_mean,
        circ_cv,
        acc.eosin_sum / acc.hema_sum if acc.hema_sum > 0 else 0.0,
        acc.tissue_count / acc.inside_count if acc.inside_count > 0 else 0.0,
    ]
    out = np.asarray(values, dtype=np.float64)
    # Any non-finite value becomes 0, as in the Swift's final map.
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def _mean_cv(values: np.ndarray) -> tuple[float, float]:
    """Population mean and coefficient of variation (std / mean)."""
    if values.size == 0:
        return 0.0, 0.0
    mean = float(values.mean())
    std = float(values.std())
    return mean, (std / mean if mean > 0 else 0.0)
