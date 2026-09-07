"""PanIN architectural descriptor — polarity and lumen shape.

Derived from a study of 62 annotated regions across 8 slides
(an unpublished internal study). The two blocks here are the ones that carried
scale-free signal and that no existing descriptor measures:

**Polarity.** Where nuclei sit relative to the lumen. In low-grade PanIN they
are basal and few touch the luminal surface; by PanIN-3 a large minority do,
rising monotonically through the grades — the strongest single axis found in
development. This is the computable form of "loss of polarity and
pseudostratification".

**Lumen shape.** The shape of the space *inside* the duct, which the outline
descriptor in :mod:`pathlearn.core.shape` structurally cannot see: that one
describes the outer traced boundary. PanIN-1a has a round open lumen
(circularity 0.34) and PanIN-1b a branched slit (0.17) with otherwise
identical nuclei, so lumen shape is the only route to that boundary.

WHY THIS IS NOT WINDOWED
========================
:mod:`pathlearn.core.geometry` tiles an annotation into fixed windows and
accumulates. That cannot work here: a distance transform from the lumen is a
**global** property of the lesion, and computing it per window would measure
distance to the nearest lumen *in that window*, which is a different and
meaningless quantity near a window edge. So the whole annotation is read as
one image, at the same fixed physical resolution the geometry descriptor uses.

SIZE IS DELIBERATELY ABSENT
===========================
Every feature is a fraction, a ratio, or a normalised shape. Region area rises
steeply with PanIN grade, and counts inherit that: in development a
luminal-debris *count* ranked among the strongest features available until it
was normalised per unit area, at which point it collapsed and reversed
direction. It was measuring how big a region had been traced, not what was in
it. A test enforces the rule, so a raw count cannot be added back by accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..io.slide import SlideImage
from ..models.annotation import Annotation
from .components import label_blobs
from .geometry import GeometryConfig
from .nucleus import ClassicalNucleusSegmenter, NucleusSegmenter
from .sampler import points_in_polygon
from .stain import deconvolve

#: Ordered descriptor names — indices match the produced vector.
PANIN_DESCRIPTOR_NAMES: tuple[str, ...] = (
    # -- polarity: where nuclei sit relative to the lumen ------------------
    "nucleiAtLumen",            # 0  fraction of nuclei on the luminal surface
    "nucleusToLumenMean",       # 1  mean nucleus->lumen distance, normalised
    "nucleusToLumenCV",         # 2  disorder of that arrangement
    "epitheliumThicknessMean",  # 3  relative thickness of the epithelial band
    "epitheliumThicknessMax",   # 4  thickest point, relative
    # -- lumen shape: the space inside the duct ---------------------------
    "lumenFraction",            # 5  lumen px / inside-polygon px
    "lumenInteriorFraction",    # 6  how much lumen survives erosion (open vs slit)
    "lumenCircularity",         # 7  1.0 = round; a branched slit is far below
    "lumenSolidity",            # 8  lumen area / its convex hull
    "lumenElongation",          # 9  principal-axis ratio of the lumen
    "lumenBoundaryComplexity",  # 10 perimeter / sqrt(area) — convolution
    "lumenBranchIndex",         # 11 branchedness: 1a round vs 1b stellate
    "lumenCountPerArea",        # 12 fragmentation, normalised by lesion area
    # -- nuclei ------------------------------------------------------------
    "nucleusAreaCV",            # 13 pleomorphism
    "nucleusArea90over50",      # 14 upper-tail nuclear enlargement
    "nuclearHyperchromasia",    # 15 nuclear vs tissue haematoxylin
    "tissueFraction",           # 16 tissue px / inside-polygon px
)

PANIN_DIMENSION = len(PANIN_DESCRIPTOR_NAMES)

#: Bump when the descriptor set or its semantics change, so stale bank rows
#: are rejected rather than silently mixed with fresh ones.
PANIN_VERSION = 1

#: Largest analysis image per annotation. At ~1 µm/px even a very large lesion
#: lands far below this; the cap only guards against a pathological polygon.
MAX_ANALYSIS_EDGE = 4096

#: A nucleus centroid within this many pixels of lumen counts as "at the
#: luminal surface". At the ~1 µm/px analysis resolution a nucleus is about
#: 6 px across, so this is roughly half a nucleus.
AT_LUMEN_PX = 3.0

#: Lumen blobs below this share of the lesion are stain gaps between cells,
#: not lumen.
MIN_LUMEN_FRACTION = 0.002


@dataclass(frozen=True, slots=True)
class PaninResult:
    features: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(PANIN_DESCRIPTOR_NAMES, self.features)}


def describe_panin(slide: SlideImage, annotation: Annotation,
                   config: GeometryConfig | None = None,
                   segmenter: NucleusSegmenter | None = None
                   ) -> np.ndarray | None:
    """The 17-D PanIN descriptor, or ``None`` when it cannot be measured.

    ``None`` means "no descriptor" and must never be replaced by zeros: a zero
    vector is indistinguishable from a real measurement of an empty region.
    """
    config = config or GeometryConfig()
    segmenter = segmenter or ClassicalNucleusSegmenter()

    if len(annotation.points) < 3:
        return None
    poly_x = np.array([p.x for p in annotation.points], dtype=np.float64)
    poly_y = np.array([p.y for p in annotation.points], dtype=np.float64)
    min_x, max_x = float(poly_x.min()), float(poly_x.max())
    min_y, max_y = float(poly_y.min()), float(poly_y.max())
    if max_x <= min_x or max_y <= min_y:
        return None

    level = slide.level_for_mpp(config.target_mpp, fallback_mpp=config.fallback_mpp)
    downsample = slide.level_downsamples[level]
    width = int(math.ceil((max_x - min_x) / downsample))
    height = int(math.ceil((max_y - min_y) / downsample))
    if width < 8 or height < 8:
        return None
    if max(width, height) > MAX_ANALYSIS_EDGE:
        return None

    try:
        rgb = slide.read_region(int(min_x), int(min_y), level, width, height)
    except Exception:
        return None
    if rgb.size == 0:
        return None

    stain = deconvolve(rgb)
    inside = _inside_mask(min_x, min_y, stain.height, stain.width, downsample,
                          poly_x, poly_y)
    inside_count = int(np.count_nonzero(inside))
    if inside_count < 64:
        return None

    tissue = inside & ~stain.is_background
    lumen_raw = inside & stain.is_background
    tissue_count = int(np.count_nonzero(tissue))
    if tissue_count < 32:
        return None

    min_lumen_px = max(4, int(MIN_LUMEN_FRACTION * inside_count))
    lumen = _significant_lumen(lumen_raw, min_lumen_px)
    lumen_blobs = label_blobs(lumen, min_area=min_lumen_px)
    lumen_count = int(np.count_nonzero(lumen))

    out = np.zeros(PANIN_DIMENSION, dtype=np.float32)
    scale = math.sqrt(inside_count)      # the lesion's own length unit

    # -- polarity ---------------------------------------------------------
    nuclei = segmenter.segment(stain, tissue).nuclei
    if nuclei and lumen_count:
        distance = _distance_to(lumen)
        h, w = distance.shape
        d = np.array([distance[int(np.clip(n.cy, 0, h - 1)),
                               int(np.clip(n.cx, 0, w - 1))] for n in nuclei],
                     dtype=np.float64)
        out[0] = float((d <= AT_LUMEN_PX).mean())
        normalised = d / scale
        out[1] = float(normalised.mean())
        out[2] = float(normalised.std() / max(normalised.mean(), 1e-9))

    if tissue_count:
        thickness = _distance_to(~tissue)[tissue]
        out[3] = float(thickness.mean() / scale)
        out[4] = float(thickness.max() / scale)

    # -- lumen shape -------------------------------------------------------
    out[5] = lumen_count / inside_count
    out[16] = tissue_count / inside_count
    if lumen_blobs:
        biggest = max(lumen_blobs, key=lambda b: b.area)
        out[6] = _interior_fraction(lumen)
        out[7] = min(1.0, 4 * math.pi * biggest.area / max(biggest.perimeter, 1) ** 2)
        largest_mask = _largest_component(lumen)
        out[8] = _solidity(largest_mask)
        out[9] = _elongation(largest_mask)
        out[10] = biggest.perimeter / max(math.sqrt(biggest.area), 1.0)
        out[11] = _branch_index(largest_mask)
        # Per unit area, so fragmentation is not just "a bigger region".
        out[12] = len(lumen_blobs) / (inside_count / 1e4)

    # -- nuclei ------------------------------------------------------------
    areas = np.array([n.area for n in nuclei], dtype=np.float64)
    if areas.size >= 5:
        out[13] = float(areas.std() / max(areas.mean(), 1e-9))
        out[14] = float(np.percentile(areas, 90) / max(np.percentile(areas, 50), 1e-9))
    hema = stain.hematoxylin
    tissue_values = hema[tissue]
    if tissue_values.size:
        threshold = float(tissue_values.mean() + tissue_values.std())
        dark = tissue & (hema >= threshold)
        if dark.any():
            out[15] = float(hema[dark].mean() / max(tissue_values.mean(), 1e-9))

    if not np.all(np.isfinite(out)):
        return None
    return out


# -- helpers ------------------------------------------------------------------

def _inside_mask(origin_x: float, origin_y: float, height: int, width: int,
                 downsample: float, poly_x: np.ndarray,
                 poly_y: np.ndarray) -> np.ndarray:
    """Boolean mask of the polygon, in analysis-image coordinates."""
    ys, xs = np.mgrid[0:height, 0:width]
    slide_x = origin_x + (xs.ravel() + 0.5) * downsample
    slide_y = origin_y + (ys.ravel() + 0.5) * downsample
    return points_in_polygon(slide_x, slide_y, poly_x, poly_y).reshape(height, width)


def _significant_lumen(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop speckle, then keep only blobs worth calling lumen."""
    from scipy import ndimage

    opened = ndimage.binary_opening(mask, np.ones((3, 3)))
    labelled, count = ndimage.label(opened)
    if count == 0:
        return opened
    sizes = np.bincount(labelled.ravel())
    keep = np.zeros(sizes.size, dtype=bool)
    keep[sizes >= min_area] = True
    keep[0] = False
    return keep[labelled]


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest True in *mask*."""
    from scipy import ndimage

    return ndimage.distance_transform_edt(~mask)


def _interior_fraction(lumen: np.ndarray) -> float:
    """Share of the lumen that survives erosion.

    An open round lumen keeps most of itself; a narrow branching slit is all
    edge and nearly vanishes. This is what separates PanIN-1b from PanIN-2.
    """
    from scipy import ndimage

    total = int(np.count_nonzero(lumen))
    if total == 0:
        return 0.0
    eroded = ndimage.binary_erosion(lumen, np.ones((5, 5)))
    return float(np.count_nonzero(eroded) / total)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labelled, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(sizes.argmax())


def _solidity(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull, QhullError
        try:
            hull = ConvexHull(np.column_stack([xs, ys]))
        except QhullError:          # collinear points have no area
            return 0.0
        return float(min(1.0, mask.sum() / max(hull.volume, 1e-9)))
    except ImportError:
        return 0.0


def _elongation(mask: np.ndarray) -> float:
    """Ratio of principal axes; 1.0 is isotropic."""
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 0.0
    coords = np.column_stack([xs - xs.mean(), ys - ys.mean()]).astype(float)
    covariance = coords.T @ coords / len(coords)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(math.sqrt(max(eigenvalues[1], 1e-9) / max(eigenvalues[0], 1e-9)))


def _branch_index(mask: np.ndarray) -> float:
    """How branched the lumen is, without needing a skeletoniser.

    Erode until the shape nearly disappears, and measure how many separate
    pieces it breaks into on the way. A round lumen shrinks to one blob; a
    stellate slit splits into its arms. That difference is the PanIN-1a
    versus 1b distinction, which nuclear features do not carry.
    """
    from scipy import ndimage

    total = int(np.count_nonzero(mask))
    if total < 25:
        return 0.0
    most = 1
    current = mask
    for _ in range(6):
        current = ndimage.binary_erosion(current, np.ones((3, 3)))
        remaining = int(np.count_nonzero(current))
        if remaining < max(4, total * 0.02):
            break
        _, pieces = ndimage.label(current)
        most = max(most, pieces)
    return float(most)
