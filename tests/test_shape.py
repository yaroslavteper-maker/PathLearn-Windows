"""Outline shape descriptors.

Assertions are against *constructed* shapes with known properties — a circle, a
star with a known lobe count, an elongated ellipse — so a failure says which
geometric property broke rather than that a number moved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pathlearn.core.shape import (RESAMPLE_POINTS, SHAPE_DESCRIPTOR_NAMES,
                                  SHAPE_DIMENSION, SIZE_DESCRIPTOR_NAMES,
                                  describe_shape, shape_matrix)
from pathlearn.models.annotation import Annotation, AnnotationColor, Point


def index(name: str) -> int:
    return SHAPE_DESCRIPTOR_NAMES.index(name)


def polygon(points, classification="x", subtractive=False) -> Annotation:
    return Annotation(points=[Point(float(x), float(y)) for x, y in points],
                      classification=classification,
                      color=AnnotationColor.default(), is_subtractive=subtractive)


def circle(radius=100.0, n=256, cx=0.0, cy=0.0) -> Annotation:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return polygon(np.column_stack([cx + radius * np.cos(t), cy + radius * np.sin(t)]))


def star(lobes=6, radius=100.0, amplitude=0.4, n=300) -> Annotation:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius * (1 + amplitude * np.sin(lobes * t))
    return polygon(np.column_stack([r * np.cos(t), r * np.sin(t)]))


def ellipse(a=200.0, b=50.0, n=256) -> Annotation:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return polygon(np.column_stack([a * np.cos(t), b * np.sin(t)]))


def square(side=100.0) -> Annotation:
    return polygon([(0, 0), (side, 0), (side, side), (0, side)])


class TestBasicShapes:
    def test_circle_is_maximally_circular(self):
        features = describe_shape(circle()).features
        assert features[index("circularity")] == pytest.approx(1.0, abs=0.01)
        assert features[index("solidity")] == pytest.approx(1.0, abs=0.01)
        assert features[index("convexity")] == pytest.approx(1.0, abs=0.01)

    def test_circle_has_no_lobes_or_elongation(self):
        features = describe_shape(circle()).features
        assert features[index("lobeCount")] == pytest.approx(0.0, abs=0.1)
        assert features[index("elongation")] == pytest.approx(0.0, abs=0.05)
        assert features[index("eccentricity")] == pytest.approx(0.0, abs=0.2)

    def test_square_circularity_is_pi_over_four(self):
        """4*pi*A/P^2 for a square is exactly pi/4."""
        features = describe_shape(square()).features
        assert features[index("circularity")] == pytest.approx(math.pi / 4, abs=0.02)

    def test_ellipse_is_elongated(self):
        features = describe_shape(ellipse(a=200, b=50)).features
        assert features[index("elongation")] > 0.6
        assert features[index("eccentricity")] > 0.9

    def test_convex_shapes_have_no_concavity(self):
        for shape in (circle(), square(), ellipse()):
            features = describe_shape(shape).features
            assert features[index("concavityDepthMean")] < 0.02


class TestLobes:
    @pytest.mark.parametrize("lobes", [3, 6, 10])
    def test_lobe_count_tracks_the_real_number(self, lobes):
        """lobeCount is peaks per 100 resampled points."""
        measured = describe_shape(star(lobes)).features[index("lobeCount")]
        expected = lobes / RESAMPLE_POINTS * 100.0
        assert measured == pytest.approx(expected, abs=0.5)

    def test_more_lobes_means_less_circular(self):
        values = [describe_shape(star(n)).features[index("circularity")]
                  for n in (0.0001, 3, 6, 10)]
        assert values == sorted(values, reverse=True)

    def test_star_is_concave(self):
        features = describe_shape(star(6)).features
        assert features[index("solidity")] < 0.9
        assert features[index("concavityDepthMean")] > 0.02


class TestDrawingStyleIndependence:
    """The descriptor must measure the shape, not how it was traced."""

    def test_vertex_count_does_not_change_the_result(self):
        sparse = describe_shape(circle(n=12)).features
        dense = describe_shape(circle(n=2000)).features
        assert np.allclose(sparse, dense, atol=0.05)

    def test_lasso_jitter_barely_moves_it(self):
        """A lasso trace has hundreds of noisy vertices; a clicked polygon does not."""
        rng = np.random.default_rng(0)
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        clean = np.column_stack([100 * np.cos(t), 100 * np.sin(t)])
        jittered = clean + rng.normal(0, 0.4, clean.shape)
        a = describe_shape(polygon(clean)).features
        b = describe_shape(polygon(jittered)).features
        assert np.allclose(a[index("circularity")], b[index("circularity")], atol=0.1)
        assert np.allclose(a[index("solidity")], b[index("solidity")], atol=0.1)

    def test_scale_invariant_block_ignores_size(self):
        """The v1 collapse was size leaking into the descriptor. It must not."""
        small = describe_shape(star(6, radius=50)).features
        large = describe_shape(star(6, radius=500)).features
        assert np.allclose(small, large, atol=0.02)

    def test_translation_invariant(self):
        assert np.allclose(describe_shape(circle(cx=0, cy=0)).features,
                           describe_shape(circle(cx=9000, cy=4000)).features, atol=0.01)

    def test_rotation_barely_matters(self):
        t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        r = 100 * (1 + 0.4 * np.sin(6 * t))
        base = polygon(np.column_stack([r * np.cos(t), r * np.sin(t)]))
        turned = polygon(np.column_stack([r * np.cos(t + 0.7), r * np.sin(t + 0.7)]))
        assert np.allclose(describe_shape(base).features,
                           describe_shape(turned).features, atol=0.05)


class TestSizeBlock:
    def test_area_uses_mpp(self):
        result = describe_shape(circle(radius=100), mpp=0.5)
        expected = math.pi * 100 ** 2 * 0.25
        assert result.size[0] == pytest.approx(expected, rel=0.02)

    def test_equivalent_diameter(self):
        result = describe_shape(circle(radius=100), mpp=1.0)
        assert result.size[2] == pytest.approx(200.0, rel=0.02)

    def test_size_is_reported_separately_from_shape(self):
        """Size must stay out of the trainable block."""
        assert len(SIZE_DESCRIPTOR_NAMES) == 3
        assert not set(SIZE_DESCRIPTOR_NAMES) & set(SHAPE_DESCRIPTOR_NAMES)


class TestGuards:
    def test_degenerate_polygons_return_none(self):
        assert describe_shape(polygon([(0, 0), (1, 1)])) is None
        assert describe_shape(polygon([(0, 0), (1, 0), (2, 0)])) is None

    def test_all_features_finite(self):
        for shape in (circle(), square(), star(6), ellipse()):
            assert np.all(np.isfinite(describe_shape(shape).features))

    def test_dimension_matches_names(self):
        assert SHAPE_DIMENSION == 12 == len(SHAPE_DESCRIPTOR_NAMES)
        assert describe_shape(circle()).features.shape == (SHAPE_DIMENSION,)

    def test_bounded_descriptors_stay_in_range(self):
        for shape in (circle(), square(), star(10), ellipse()):
            features = describe_shape(shape).features
            for name in ("circularity", "solidity", "convexity", "extent",
                         "concavityFraction", "eccentricity"):
                value = features[index(name)]
                assert 0.0 <= value <= 1.0 + 1e-6, f"{name} = {value}"


class TestMatrix:
    def test_stacks_and_labels(self):
        matrix, labels, kept = shape_matrix(
            [circle(), star(6), ellipse()], mpp=0.5)
        assert matrix.shape == (3, SHAPE_DIMENSION)
        assert len(labels) == 3 and len(kept) == 3

    def test_skips_subtractive_and_degenerate(self):
        annotations = [circle(),
                       polygon([(0, 0), (1, 1)]),
                       polygon([(0, 0), (10, 0), (10, 10)], subtractive=True)]
        matrix, labels, _ = shape_matrix(annotations)
        assert matrix.shape[0] == 1

    def test_empty(self):
        matrix, labels, kept = shape_matrix([])
        assert matrix.shape == (0, SHAPE_DIMENSION) and labels == [] and kept == []
