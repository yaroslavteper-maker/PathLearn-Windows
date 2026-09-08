"""Patch sampling and feature pooling."""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.core.pooling import mean_max_std, pooled_dimension
from pathlearn.core.sampler import (cancel_origins, estimated_patch_count,
                                    points_in_polygon, sample_patches)
from pathlearn.models.annotation import Point

SLIDE_W, SLIDE_H = 10_000, 8_000


def square(x0: float, y0: float, side: float) -> list[Point]:
    return [Point(x0, y0), Point(x0 + side, y0),
            Point(x0 + side, y0 + side), Point(x0, y0 + side)]


class TestSamplePatches:
    def test_grid_spacing_and_count(self):
        # 1000-px square, 100-px patches, stride 100 -> 10x10 origins, but the
        # grid stops at max - patch, so the last row/col of centres still fits.
        origins = sample_patches(square(1000, 1000, 1000), SLIDE_W, SLIDE_H,
                                 patch_size_level=100, stride_level=100,
                                 level_downsample=1.0)
        assert origins
        xs = sorted({x for x, _ in origins})
        assert xs[1] - xs[0] == 100

    def test_centre_inside_rule(self):
        """A patch is kept iff its centre is inside — not its corner."""
        origins = sample_patches(square(0, 0, 200), SLIDE_W, SLIDE_H,
                                 patch_size_level=100, stride_level=100,
                                 level_downsample=1.0)
        for x, y in origins:
            assert 0 <= x + 50 <= 200 and 0 <= y + 50 <= 200

    def test_downsample_scales_patch_and_stride(self):
        """patchSizeLevel0 = round(patchSizeLevel * downsample)."""
        polygon = square(0, 0, 8000)
        at_level_0 = sample_patches(polygon, SLIDE_W, SLIDE_H,
                                    patch_size_level=256, stride_level=256,
                                    level_downsample=1.0)
        at_level_2 = sample_patches(polygon, SLIDE_W, SLIDE_H,
                                    patch_size_level=256, stride_level=256,
                                    level_downsample=4.0)
        # 4x coarser -> 4x the level-0 stride -> ~1/16 the patches.
        assert len(at_level_2) < len(at_level_0) / 8
        xs = sorted({x for x, _ in at_level_2})
        assert xs[1] - xs[0] == 1024

    def test_patch_only_just_fits(self):
        """A box exactly one patch wide yields exactly one column."""
        origins = sample_patches(square(0, 0, 1024), SLIDE_W, SLIDE_H,
                                 patch_size_level=256, stride_level=256,
                                 level_downsample=4.0)
        assert sorted({x for x, _ in origins}) == [0]

    def test_the_far_edge_is_covered_when_the_box_is_not_a_whole_number(self):
        """A 2000 px box and a 1024 px patch: one column would miss 976 px.

        ``arange`` stops at the last whole stride, so without a final origin
        snapped flush to the far edge, nearly half this region would never be
        sampled — a band of tissue with no heatmap over it.
        """
        origins = sample_patches(square(0, 0, 2000), SLIDE_W, SLIDE_H,
                                 patch_size_level=256, stride_level=256,
                                 level_downsample=4.0)
        xs = sorted({x for x, _ in origins})
        assert xs == [0, 976]
        assert xs[-1] + 1024 == 2000        # reaches the far edge exactly

    def test_a_whole_number_of_patches_gains_no_extra_column(self):
        """The snap must not add a redundant row when the tiling is exact."""
        origins = sample_patches(square(0, 0, 2048), SLIDE_W, SLIDE_H,
                                 patch_size_level=256, stride_level=256,
                                 level_downsample=4.0)
        assert sorted({x for x, _ in origins}) == [0, 1024]

    def test_patches_never_leave_the_slide(self):
        """A polygon overhanging the edge must not emit out-of-bounds rects."""
        origins = sample_patches(square(SLIDE_W - 300, SLIDE_H - 300, 600),
                                 SLIDE_W, SLIDE_H,
                                 patch_size_level=256, stride_level=128,
                                 level_downsample=1.0)
        for x, y in origins:
            assert x >= 0 and y >= 0
            assert x + 256 <= SLIDE_W and y + 256 <= SLIDE_H

    def test_negative_coordinates_are_rejected(self):
        origins = sample_patches(square(-500, -500, 400), SLIDE_W, SLIDE_H,
                                 patch_size_level=100, stride_level=100,
                                 level_downsample=1.0)
        assert origins == []

    def test_concave_polygon_excludes_the_notch(self):
        """An L-shape must not sample inside its missing quadrant."""
        polygon = [Point(0, 0), Point(1000, 0), Point(1000, 400),
                   Point(400, 400), Point(400, 1000), Point(0, 1000)]
        origins = sample_patches(polygon, SLIDE_W, SLIDE_H,
                                 patch_size_level=50, stride_level=50,
                                 level_downsample=1.0)
        assert origins
        for x, y in origins:
            cx, cy = x + 25, y + 25
            assert not (cx > 400 and cy > 400), f"sampled the notch at ({cx},{cy})"

    @pytest.mark.parametrize("kwargs", [
        dict(patch_size_level=0, stride_level=100, level_downsample=1.0),
        dict(patch_size_level=100, stride_level=0, level_downsample=1.0),
        dict(patch_size_level=100, stride_level=100, level_downsample=0.0),
    ])
    def test_degenerate_parameters_return_empty(self, kwargs):
        assert sample_patches(square(0, 0, 500), SLIDE_W, SLIDE_H, **kwargs) == []

    def test_too_few_points_returns_empty(self):
        assert sample_patches([Point(0, 0), Point(10, 10)], SLIDE_W, SLIDE_H,
                              100, 100, 1.0) == []

    def test_patch_larger_than_polygon_returns_empty(self):
        assert sample_patches(square(0, 0, 50), SLIDE_W, SLIDE_H,
                              patch_size_level=500, stride_level=500,
                              level_downsample=1.0) == []

    def test_no_y_mirror(self):
        """A polygon near the top must sample near the top.

        If anyone reintroduces the macOS mirror this is the test that fails.
        """
        origins = sample_patches(square(100, 100, 400), SLIDE_W, SLIDE_H,
                                 patch_size_level=100, stride_level=100,
                                 level_downsample=1.0)
        assert origins
        assert max(y for _, y in origins) < 600


class TestCancelOrigins:
    def test_subtractive_polygon_removes_covered_origins(self):
        origins = sample_patches(square(0, 0, 1000), SLIDE_W, SLIDE_H,
                                 patch_size_level=100, stride_level=100,
                                 level_downsample=1.0)
        kept = cancel_origins(origins, 100, [square(0, 0, 500)])
        assert len(kept) < len(origins)
        for x, y in kept:
            assert not (0 < x + 50 < 500 and 0 < y + 50 < 500)

    def test_no_polygons_is_identity(self):
        origins = [(0, 0), (100, 100)]
        assert cancel_origins(origins, 100, []) == origins

    def test_degenerate_polygons_are_ignored(self):
        origins = [(0, 0), (100, 100)]
        assert cancel_origins(origins, 100, [[Point(0, 0), Point(1, 1)]]) == origins

    def test_multiple_subtractive_polygons_compose(self):
        origins = sample_patches(square(0, 0, 1000), SLIDE_W, SLIDE_H, 100, 100, 1.0)
        one = cancel_origins(origins, 100, [square(0, 0, 300)])
        both = cancel_origins(origins, 100, [square(0, 0, 300), square(700, 700, 300)])
        assert len(both) < len(one) < len(origins)


class TestPointsInPolygon:
    def test_matches_scalar_annotation_contains(self):
        from pathlearn.models.annotation import Annotation, AnnotationColor

        polygon = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        ann = Annotation(points=polygon, classification="x",
                         color=AnnotationColor.default())
        rng = np.random.default_rng(0)
        pts = rng.uniform(-20, 120, size=(200, 2))

        vector = points_in_polygon(pts[:, 0], pts[:, 1],
                                   np.array([p.x for p in polygon], dtype=float),
                                   np.array([p.y for p in polygon], dtype=float))
        scalar = np.array([ann.contains(x, y) for x, y in pts])
        assert np.array_equal(vector, scalar)

    def test_empty_input(self):
        result = points_in_polygon(np.array([]), np.array([]),
                                   np.array([0.0, 1, 1]), np.array([0.0, 0, 1]))
        assert result.size == 0


class TestEstimatedPatchCount:
    def test_scales_with_stride(self):
        polygon = square(0, 0, 1000)
        assert estimated_patch_count(polygon, 100, 1.0) == 100
        assert estimated_patch_count(polygon, 200, 1.0) == 25

    def test_degenerate_returns_zero(self):
        assert estimated_patch_count([Point(0, 0)], 100, 1.0) == 0
        assert estimated_patch_count(square(0, 0, 100), 0, 1.0) == 0


class TestMeanMaxStd:
    def test_layout_is_mean_then_max_then_std(self):
        vectors = np.array([[1.0, 10.0], [3.0, 20.0]], dtype=np.float32)
        pooled = mean_max_std(vectors)
        assert pooled.shape == (6,)
        assert pooled[:2] == pytest.approx([2.0, 15.0])
        assert pooled[2:4] == pytest.approx([3.0, 20.0])
        assert pooled[4:] == pytest.approx([1.0, 5.0])   # population std

    def test_single_vector_has_zero_std(self):
        pooled = mean_max_std(np.array([[2.0, 4.0]], dtype=np.float32))
        assert pooled[:2] == pytest.approx([2.0, 4.0])
        assert pooled[4:] == pytest.approx([0.0, 0.0])

    def test_empty_bag_returns_none(self):
        assert mean_max_std(np.zeros((0, 8), dtype=np.float32)) is None
        assert mean_max_std(None) is None

    def test_dimension_helper(self):
        assert pooled_dimension(768) == 2304

class TestTheGridCoversTheRegion:
    """The bug this class exists for: a heatmap that stopped short of the
    annotation's right and bottom edges, in the shape of the annotation."""

    @staticmethod
    def covered_fraction(polygon, patch=224, stride=224, downsample=1.0,
                         step=16):
        """What fraction of the polygon's area the sampled patches cover."""
        import numpy as np

        from pathlearn.core.sampler import points_in_polygon

        xs = np.array([p.x for p in polygon], dtype=np.float64)
        ys = np.array([p.y for p in polygon], dtype=np.float64)
        pad = int(patch * downsample)
        gx = np.arange(xs.min() - pad, xs.max() + pad, step) + step / 2
        gy = np.arange(ys.min() - pad, ys.max() + pad, step) + step / 2
        X, Y = np.meshgrid(gx, gy)
        inside = points_in_polygon(X.ravel(), Y.ravel(), xs, ys).reshape(X.shape)

        origins = sample_patches(polygon, SLIDE_W, SLIDE_H, patch, stride,
                                 downsample)
        size = int(round(patch * downsample))
        covered = np.zeros_like(inside, dtype=bool)
        for x, y in origins:
            covered[np.ix_((gy >= y) & (gy < y + size),
                           (gx >= x) & (gx < x + size))] = True
        return float((inside & covered).sum()) / float(inside.sum())

    @pytest.mark.parametrize("side", [500, 1000, 2000, 4000])
    def test_a_square_is_covered_completely(self, side):
        """Whatever the size, an axis-aligned region tiles exactly."""
        assert self.covered_fraction(square(0, 0, side)) == pytest.approx(1.0)

    def test_a_square_at_a_coarse_level_too(self):
        assert self.covered_fraction(square(0, 0, 4000), patch=256,
                                     stride=256, downsample=4.0) ==             pytest.approx(1.0)

    def test_a_big_round_region_is_almost_completely_covered(self):
        """Curved edges keep a thin inset from the centre-inside rule, but
        nothing like the stride-wide band the grid used to leave."""
        import math

        circle = [Point(5_000 + 3000 * math.cos(t), 4_000 + 3000 * math.sin(t))
                  for t in [i * 2 * math.pi / 64 for i in range(64)]]
        assert self.covered_fraction(circle) > 0.98

    def test_overlapping_strides_still_cover(self):
        assert self.covered_fraction(square(0, 0, 1500), stride=112) ==             pytest.approx(1.0)
