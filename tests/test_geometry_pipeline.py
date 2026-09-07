"""End-to-end geometry descriptor tests against a slide with known architecture.

This is the test that matters for grading. It builds a slide containing two
regions whose architecture differs in exactly the ways the descriptor claims to
measure, then asserts the descriptor moves in the right direction:

* **crowded** — dense small nuclei, many small irregular lumina (PaNIN-3-like)
* **orderly** — sparse larger nuclei, few large round lumina (PaNIN-1-like)

It also pins the fixed-resolution property from ``04-DESIGN-DECISIONS.md`` §2:
the same architecture must produce similar descriptors regardless of how large
the annotation is. Version 1 failed exactly there.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from pathlearn.core.geometry import (DESCRIPTOR_NAMES, GeometryConfig, describe)
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point

CYTOPLASM = (215, 150, 185)
NUCLEUS = (70, 55, 125)
LUMEN = (252, 252, 252)

#: Level-0 microns per pixel of the generated slide.
MPP = 0.5
SLIDE_W, SLIDE_H = 4096, 2048
#: The two regions sit side by side, each half the slide.
CROWDED_BOX = (0, 0, 2048, 2048)
ORDERLY_BOX = (2048, 0, 4096, 2048)


def index(name: str) -> int:
    return DESCRIPTOR_NAMES.index(name)


def _stamp(image: np.ndarray, box, cx: int, cy: int, radius: int,
           color, lobes: int = 0, amplitude: float = 0.0) -> None:
    """Paint one blob, optionally lobed so its boundary is irregular.

    ``lobes``/``amplitude`` modulate the radius with angle,
    ``r(theta) = radius * (1 + amplitude*sin(lobes*theta))``, which is what
    makes a cribriform-looking lumen genuinely non-circular rather than merely
    a different size.
    """
    x0, y0, x1, y1 = box
    reach = int(radius * (1 + abs(amplitude))) + 1
    ax0, ax1 = max(x0, cx - reach), min(x1, cx + reach + 1)
    ay0, ay1 = max(y0, cy - reach), min(y1, cy + reach + 1)
    if ax1 <= ax0 or ay1 <= ay0:
        return

    ys, xs = np.ogrid[ay0:ay1, ax0:ax1]
    dx, dy = xs - cx, ys - cy
    dist2 = dx * dx + dy * dy
    if lobes and amplitude:
        local_r = radius * (1.0 + amplitude * np.sin(lobes * np.arctan2(dy, dx)))
        mask = dist2 <= np.maximum(local_r, 1.0) ** 2
    else:
        mask = dist2 <= radius ** 2
    image[ay0:ay1, ax0:ax1][mask] = color


def _paint_region(image: np.ndarray, box, *, nucleus_spacing: int, nucleus_radius: int,
                  lumen_spacing: int, lumen_radius: int, lumen_lobes: int = 0,
                  lumen_amplitude: float = 0.0, seed: int = 0) -> None:
    """Fill *box* with a lattice of nuclei, then lumina painted over them.

    Order matters: lumina are open spaces, so they must be painted *last*.
    Painting nuclei afterwards would punch holes through every lumen and wreck
    both the lumen count and its circularity.
    """
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = CYTOPLASM

    for cy in range(y0 + nucleus_spacing // 2, y1, nucleus_spacing):
        for cx in range(x0 + nucleus_spacing // 2, x1, nucleus_spacing):
            _stamp(image, box, cx, cy, nucleus_radius, NUCLEUS)

    for cy in range(y0 + lumen_spacing // 2, y1, lumen_spacing):
        for cx in range(x0 + lumen_spacing // 2, x1, lumen_spacing):
            _stamp(image, box, cx, cy, lumen_radius, LUMEN,
                   lobes=lumen_lobes, amplitude=lumen_amplitude)


def _write_slide(path, image: np.ndarray, levels: int = 3) -> None:
    pixels_per_cm = 10_000.0 / MPP
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        current = image
        for level in range(levels):
            writer.write(current, tile=(256, 256), photometric="rgb",
                         compression="zlib",
                         resolution=(pixels_per_cm / (2 ** level),
                                     pixels_per_cm / (2 ** level)),
                         resolutionunit="CENTIMETER",
                         subfiletype=1 if level else 0)
            if level + 1 < levels:
                h, w = current.shape[0] // 2 * 2, current.shape[1] // 2 * 2
                c = current[:h, :w].astype(np.uint16)
                current = ((c[0::2, 0::2] + c[1::2, 0::2]
                            + c[0::2, 1::2] + c[1::2, 1::2]) // 4).astype(np.uint8)


@pytest.fixture(scope="module")
def architecture_slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("geometry") / "architecture.tif"
    image = np.full((SLIDE_H, SLIDE_W, 3), CYTOPLASM, dtype=np.uint8)
    # Sizes are chosen so nuclei stay resolvable at the 1 um/px analysis level:
    # a level-0 radius of 6 px (0.5 um/px) is 3 px at analysis, so a 6 px
    # diameter -- comparable to a real ~7 um nucleus. Anything much smaller is
    # sub-resolution and the classical segmenter returns fragments.
    #
    # Crowded: nuclei every 20 px (10 um), many small strongly lobed lumina.
    _paint_region(image, CROWDED_BOX, nucleus_spacing=20, nucleus_radius=6,
                  lumen_spacing=64, lumen_radius=14, lumen_lobes=6,
                  lumen_amplitude=0.45, seed=1)
    # Orderly: nuclei every 44 px (22 um) and larger, few big round lumina.
    _paint_region(image, ORDERLY_BOX, nucleus_spacing=44, nucleus_radius=10,
                  lumen_spacing=220, lumen_radius=60, seed=2)
    _write_slide(path, image)
    with SlideImage(path) as slide:
        yield slide


def region_annotation(box, inset: int = 200, name: str = "r") -> Annotation:
    x0, y0, x1, y1 = box
    return Annotation(
        points=[Point(x0 + inset, y0 + inset), Point(x1 - inset, y0 + inset),
                Point(x1 - inset, y1 - inset), Point(x0 + inset, y1 - inset)],
        classification=name, color=AnnotationColor.default(), name=name,
    )


@pytest.fixture(scope="module")
def descriptors(architecture_slide):
    config = GeometryConfig(window_px=512, max_windows=8)
    crowded = describe(architecture_slide, region_annotation(CROWDED_BOX, name="crowded"),
                       config)
    orderly = describe(architecture_slide, region_annotation(ORDERLY_BOX, name="orderly"),
                       config)
    assert crowded is not None and orderly is not None
    return crowded, orderly


class TestDescriptorShape:
    def test_is_14d_float32_and_finite(self, descriptors):
        for vector in descriptors:
            assert vector.shape == (14,)
            assert vector.dtype == np.float32
            assert np.all(np.isfinite(vector))

    def test_fractions_are_in_range(self, descriptors):
        for vector in descriptors:
            for name in ("nuclearPixelFraction", "lumenAreaFraction", "tissueFraction",
                         "lumenCircularityMean"):
                value = vector[index(name)]
                assert 0.0 <= value <= 1.0, f"{name} = {value}"


class TestArchitectureDiscrimination:
    """Each assertion names the architectural property it is testing."""

    def test_crowded_region_has_higher_nuclear_density(self, descriptors):
        crowded, orderly = descriptors
        assert crowded[index("nucleiPerMm2")] > orderly[index("nucleiPerMm2")]

    def test_crowded_region_has_smaller_nuclei(self, descriptors):
        crowded, orderly = descriptors
        assert crowded[index("nuclearAreaMeanUm2")] < orderly[index("nuclearAreaMeanUm2")]

    def test_crowded_region_has_tighter_nuclear_spacing(self, descriptors):
        crowded, orderly = descriptors
        assert crowded[index("nnDistMeanUm")] < orderly[index("nnDistMeanUm")]

    def test_crowded_region_has_more_lumina(self, descriptors):
        """Lumen count per mm^2 is the cribriforming proxy."""
        crowded, orderly = descriptors
        assert crowded[index("lumenCountPerMm2")] > orderly[index("lumenCountPerMm2")]

    def test_orderly_region_has_larger_lumina(self, descriptors):
        crowded, orderly = descriptors
        assert orderly[index("lumenSizeMeanUm2")] > crowded[index("lumenSizeMeanUm2")]

    def test_orderly_lumina_are_rounder(self, descriptors):
        """Low circularity = irregular/cribriform boundaries."""
        crowded, orderly = descriptors
        assert orderly[index("lumenCircularityMean")] > crowded[index("lumenCircularityMean")]

    def test_nuclear_density_is_physically_plausible(self, descriptors):
        """Nuclei every 5 um is on the order of 1e4-1e5 per mm^2."""
        crowded, _ = descriptors
        assert 5_000 < crowded[index("nucleiPerMm2")] < 200_000


class TestFixedResolutionInvariance:
    """The v2 fix: descriptors must not depend on annotation size.

    In v1, larger annotations were read at a coarser pyramid level, which
    inflated absolute micron-denominated descriptors and taught the classifier
    that big lesions are high grade. These assertions would fail under that bug.
    """

    @pytest.mark.parametrize("name", ["nuclearAreaMeanUm2", "nnDistMeanUm", "nucleiPerMm2"])
    def test_absolute_descriptors_survive_a_size_change(self, architecture_slide, name):
        config = GeometryConfig(window_px=512, max_windows=8)
        small = describe(architecture_slide,
                         region_annotation(CROWDED_BOX, inset=700), config)
        large = describe(architecture_slide,
                         region_annotation(CROWDED_BOX, inset=200), config)
        assert small is not None and large is not None

        i = index(name)
        ratio = small[i] / large[i]
        assert 0.7 < ratio < 1.4, f"{name} changed by {ratio:.2f}x with annotation size"

    def test_analysis_level_is_chosen_from_mpp_not_size(self, architecture_slide):
        """At 0.5 um/px base and a 1.0 um/px target, that is downsample 2."""
        level = architecture_slide.best_level_for_downsample(
            max(1.0, GeometryConfig().target_mpp / MPP))
        assert architecture_slide.level_downsamples[level] == pytest.approx(2.0)


class TestGuards:
    def test_degenerate_polygon_returns_none(self, architecture_slide):
        ann = Annotation(points=[Point(0, 0), Point(10, 0)],
                         classification="x", color=AnnotationColor.default())
        assert describe(architecture_slide, ann) is None

    def test_zero_area_polygon_returns_none(self, architecture_slide):
        ann = Annotation(points=[Point(5, 5), Point(5, 5), Point(5, 5)],
                         classification="x", color=AnnotationColor.default())
        assert describe(architecture_slide, ann) is None

    def test_too_few_nuclei_returns_none(self, architecture_slide):
        """A tiny region cannot support a descriptor, and must say so."""
        ann = Annotation(points=[Point(100, 100), Point(115, 100),
                                 Point(115, 115), Point(100, 115)],
                         classification="x", color=AnnotationColor.default())
        assert describe(architecture_slide, ann,
                        GeometryConfig(window_px=256, min_nuclei=500)) is None

    def test_max_windows_bounds_the_work(self, architecture_slide):
        """Capping windows must still produce a usable descriptor."""
        vector = describe(architecture_slide, region_annotation(CROWDED_BOX),
                          GeometryConfig(window_px=256, max_windows=2))
        assert vector is not None and np.all(np.isfinite(vector))


class TestAnalysisResolution:
    """The level chosen must actually be near the target, not merely legal.

    OpenSlide's ``best_level_for_downsample`` never returns a level coarser than
    requested, so a 1.0 um/px target on a 0.25 um/px slide with 1/4/16
    downsamples silently lands on level 0 — four times too fine. At that scale
    the classical segmenter measures chromatin rather than nuclei.
    """

    def test_chosen_level_is_near_the_target_mpp(self, architecture_slide):
        config = GeometryConfig()
        level = architecture_slide.level_for_mpp(config.target_mpp)
        actual = MPP * architecture_slide.level_downsamples[level]
        assert 0.5 * config.target_mpp <= actual <= 2.0 * config.target_mpp

    def test_nearest_beats_best_level_for_downsample(self, architecture_slide):
        """Documents the discrepancy that made v2 descriptors meaningless."""
        config = GeometryConfig()
        desired = max(1.0, config.target_mpp / MPP)
        rounded_down = architecture_slide.best_level_for_downsample(desired)
        nearest = architecture_slide.level_for_mpp(config.target_mpp)
        error_down = abs(MPP * architecture_slide.level_downsamples[rounded_down]
                         - config.target_mpp)
        error_near = abs(MPP * architecture_slide.level_downsamples[nearest]
                         - config.target_mpp)
        assert error_near <= error_down

    def test_nuclei_are_physically_plausible(self, architecture_slide):
        """A descriptor measuring chromatin instead of nuclei looks like this.

        Real nuclei are ~7-10 um across (roughly 30-80 um^2) at densities well
        under 30,000/mm^2. The v2 bug produced ~37,000/mm^2 of ~1.6 um^2 on real
        slides — impossible, and the tell that the resolution was wrong.
        """
        vector = describe(architecture_slide, region_annotation(CROWDED_BOX),
                          GeometryConfig(window_px=512, max_windows=6))
        assert vector is not None
        assert 5.0 < vector[index("nuclearAreaMeanUm2")] < 300.0
        assert vector[index("nucleiPerMm2")] < 60_000
        assert vector[index("nnDistMeanUm")] > 2.0
