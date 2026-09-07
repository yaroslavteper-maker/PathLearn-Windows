"""Stain deconvolution, connected components, nucleus segmentation, descriptors.

Where possible these assert against *constructed* ground truth — an image with
a known number of nuclei of known size — rather than golden numbers, so a
failure says what actually broke.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.core.components import Blob, circularity, label_blobs
from pathlearn.core.geometry import DESCRIPTOR_NAMES, DIMENSION, VERSION
from pathlearn.core.nucleus import (ClassicalNucleusSegmenter, DetectedNucleus,
                                    nearest_neighbour_distances)
from pathlearn.core.stain import BACKGROUND_THRESHOLD, deconvolve

# Approximate H&E appearance for synthetic images.
NUCLEUS_RGB = (72, 60, 130)     # haematoxylin-dominant
CYTOPLASM_RGB = (220, 150, 190)  # eosin-dominant
WHITE_RGB = (250, 250, 250)


def tissue_image(size: int = 128, nucleus_positions=(), radius: int = 5,
                 lumen_boxes=()) -> np.ndarray:
    """Cytoplasm background with discrete nuclei and white lumina."""
    image = np.full((size, size, 3), CYTOPLASM_RGB, dtype=np.uint8)
    for x0, y0, x1, y1 in lumen_boxes:
        image[y0:y1, x0:x1] = WHITE_RGB
    ys, xs = np.mgrid[0:size, 0:size]
    for cx, cy in nucleus_positions:
        image[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = NUCLEUS_RGB
    return image


class TestStainDeconvolution:
    def test_nuclei_are_haematoxylin_dominant(self):
        result = deconvolve(tissue_image(64, [(32, 32)], radius=8))
        nucleus = result.hematoxylin[32, 32]
        background = result.hematoxylin[5, 5]
        assert nucleus > background

    def test_cytoplasm_is_eosin_dominant(self):
        result = deconvolve(np.full((16, 16, 3), CYTOPLASM_RGB, dtype=np.uint8))
        assert result.eosin.mean() > result.hematoxylin.mean()

    def test_background_detection_uses_the_threshold(self):
        image = np.full((8, 8, 3), BACKGROUND_THRESHOLD, dtype=np.uint8)
        assert deconvolve(image).is_background.all()
        image[..., 0] = BACKGROUND_THRESHOLD - 1
        assert not deconvolve(image).is_background.any()

    def test_concentrations_are_non_negative(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        result = deconvolve(image)
        assert (result.hematoxylin >= 0).all() and (result.eosin >= 0).all()

    def test_pure_black_stays_finite(self):
        """The +1 in OD = -log10((I+1)/256) is what keeps this finite."""
        result = deconvolve(np.zeros((4, 4, 3), dtype=np.uint8))
        assert np.all(np.isfinite(result.hematoxylin))

    def test_shape_and_dtype(self):
        result = deconvolve(tissue_image(32))
        assert result.hematoxylin.shape == (32, 32)
        assert result.hematoxylin.dtype == np.float32
        assert result.is_background.dtype == np.bool_

    def test_rejects_non_rgb(self):
        with pytest.raises(ValueError):
            deconvolve(np.zeros((8, 8), dtype=np.uint8))


class TestConnectedComponents:
    def test_counts_separate_blobs(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:5, 2:5] = True
        mask[10:14, 10:14] = True
        blobs = label_blobs(mask)
        assert len(blobs) == 2
        assert sorted(b.area for b in blobs) == [9, 16]

    def test_four_connectivity_keeps_diagonals_apart(self):
        """Diagonal touching must be two blobs, not one."""
        mask = np.zeros((6, 6), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        assert len(label_blobs(mask)) == 2

    def test_centroid(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 4:8] = True     # rows 2-5, cols 4-7
        [blob] = label_blobs(mask)
        assert blob.cx == pytest.approx(5.5)
        assert blob.cy == pytest.approx(3.5)

    def test_min_area_filter(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[1, 1] = True             # area 1
        mask[10:15, 10:15] = True     # area 25
        assert len(label_blobs(mask, min_area=5)) == 1

    def test_perimeter_counts_boundary_pixels(self):
        """A 4x4 solid square: 12 border pixels, 4 interior."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        [blob] = label_blobs(mask)
        assert blob.area == 16
        assert blob.perimeter == 12

    def test_image_edge_counts_as_boundary(self):
        """A blob flush against the edge has those pixels as boundary too."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[0:2, 0:2] = True
        [blob] = label_blobs(mask)
        assert blob.perimeter == 4      # every pixel touches an edge or outside

    def test_empty_mask(self):
        assert label_blobs(np.zeros((10, 10), dtype=bool)) == []

    def test_circularity_of_a_disc_beats_a_line(self):
        size = 61
        ys, xs = np.mgrid[0:size, 0:size]
        disc = (xs - 30) ** 2 + (ys - 30) ** 2 <= 20 ** 2
        line = np.zeros((size, size), dtype=bool)
        line[30, 5:56] = True

        [disc_blob] = label_blobs(disc)
        [line_blob] = label_blobs(line)
        assert (circularity(disc_blob.area, disc_blob.perimeter)
                > circularity(line_blob.area, line_blob.perimeter))

    def test_circularity_is_capped_at_one(self):
        assert circularity(1000, 1) == 1.0
        assert circularity(10, 0) == 0.0


class TestNucleusSegmenter:
    def test_finds_the_expected_number_of_nuclei(self):
        positions = [(20, 20), (60, 20), (20, 60), (60, 60), (40, 40)]
        image = tissue_image(80, positions, radius=5)
        stain = deconvolve(image)
        tissue = ~stain.is_background

        result = ClassicalNucleusSegmenter().segment(stain, tissue)
        assert len(result.nuclei) == len(positions)

    def test_centroids_land_on_the_nuclei(self):
        positions = [(20, 20), (60, 60)]
        stain = deconvolve(tissue_image(80, positions, radius=6))
        result = ClassicalNucleusSegmenter().segment(stain, ~stain.is_background)

        found = sorted((round(n.cx), round(n.cy)) for n in result.nuclei)
        for (ex, ey), (gx, gy) in zip(sorted(positions), found):
            assert abs(ex - gx) <= 1 and abs(ey - gy) <= 1

    def test_pixel_count_is_pre_filter(self):
        """nucleus_pixel_count feeds crowding, so it ignores min-area."""
        stain = deconvolve(tissue_image(80, [(40, 40)], radius=6))
        result = ClassicalNucleusSegmenter(min_nucleus_area=10_000).segment(
            stain, ~stain.is_background)
        assert result.nuclei == []              # all filtered out as instances
        assert result.nucleus_pixel_count > 0   # but the pixels still counted

    def test_min_area_filters_speckle(self):
        stain = deconvolve(tissue_image(80, [(40, 40)], radius=1))
        lenient = ClassicalNucleusSegmenter(min_nucleus_area=1)
        strict = ClassicalNucleusSegmenter(min_nucleus_area=200)
        mask = ~stain.is_background
        assert len(lenient.segment(stain, mask).nuclei) >= 1
        assert strict.segment(stain, mask).nuclei == []

    def test_empty_tissue_mask_is_safe(self):
        stain = deconvolve(tissue_image(32))
        result = ClassicalNucleusSegmenter().segment(stain, np.zeros((32, 32), dtype=bool))
        assert result.nuclei == [] and result.nucleus_pixel_count == 0

    def test_mismatched_mask_raises(self):
        stain = deconvolve(tissue_image(32))
        with pytest.raises(ValueError):
            ClassicalNucleusSegmenter().segment(stain, np.zeros((8, 8), dtype=bool))

    def test_higher_k_detects_fewer(self):
        positions = [(x * 16 + 8, y * 16 + 8) for x in range(4) for y in range(4)]
        stain = deconvolve(tissue_image(64, positions, radius=3))
        mask = ~stain.is_background
        lenient = len(ClassicalNucleusSegmenter(threshold_k=0.0).segment(stain, mask).nuclei)
        strict = len(ClassicalNucleusSegmenter(threshold_k=3.0).segment(stain, mask).nuclei)
        assert strict <= lenient


class TestNearestNeighbourDistances:
    def test_distance_on_a_known_grid(self):
        nuclei = [DetectedNucleus(10, 0.0, 0.0), DetectedNucleus(10, 3.0, 4.0)]
        distances = nearest_neighbour_distances(nuclei, mpp=1.0)
        assert np.allclose(distances, [5.0, 5.0])

    def test_scales_with_mpp(self):
        nuclei = [DetectedNucleus(10, 0.0, 0.0), DetectedNucleus(10, 0.0, 10.0)]
        assert np.allclose(nearest_neighbour_distances(nuclei, mpp=0.5), [5.0, 5.0])

    def test_fewer_than_two_returns_empty(self):
        assert nearest_neighbour_distances([], mpp=1.0).size == 0
        assert nearest_neighbour_distances([DetectedNucleus(1, 0.0, 0.0)], mpp=1.0).size == 0

    def test_cap_keeps_the_largest(self):
        nuclei = [DetectedNucleus(area=i, cx=float(i), cy=0.0) for i in range(1, 51)]
        assert len(nearest_neighbour_distances(nuclei, mpp=1.0, cap=10)) == 10

    def test_chunking_matches_unchunked(self):
        """The chunked pass must agree with a plain O(n^2) computation."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 500, size=(1500, 2))
        nuclei = [DetectedNucleus(10, float(x), float(y)) for x, y in coords]
        got = nearest_neighbour_distances(nuclei, mpp=1.0, cap=5000)

        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        assert np.allclose(got, np.sqrt(d2.min(axis=1)))


class TestDescriptorContract:
    def test_dimension_and_names_agree(self):
        assert DIMENSION == 14 == len(DESCRIPTOR_NAMES)

    def test_version_is_the_nearest_level_one(self):
        """v1: level confound. v2: best_level_for_downsample chose too fine."""
        assert VERSION == 3

    def test_name_order_matches_the_spec(self):
        assert DESCRIPTOR_NAMES[0] == "nucleiPerMm2"
        assert DESCRIPTOR_NAMES[3] == "nuclearPixelFraction"
        assert DESCRIPTOR_NAMES[10] == "lumenCircularityMean"
        assert DESCRIPTOR_NAMES[13] == "tissueFraction"
