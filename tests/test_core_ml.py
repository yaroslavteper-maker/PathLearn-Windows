"""Logistic regression, t-SNE and centroids."""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.core.centroids import CentroidSet
from pathlearn.core.logistic import (softmax, stratified_split, train, zscore_apply,
                                     zscore_fit)
from pathlearn.core.tsne import SplitMix64, TSNEConfig, TSNEError, embed, stratified_subsample


def blobs(n_per_class: int = 60, dim: int = 8, classes: int = 3, seed: int = 0):
    """Well-separated Gaussian blobs, one per class."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 5, size=(classes, dim)).astype(np.float32)
    x = np.concatenate([centres[k] + rng.normal(0, 0.5, size=(n_per_class, dim))
                        for k in range(classes)]).astype(np.float32)
    y = np.repeat(np.arange(classes), n_per_class)
    return x, y


class TestSoftmax:
    def test_rows_sum_to_one(self):
        p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32))
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_is_shift_invariant(self):
        z = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        assert np.allclose(softmax(z), softmax(z + 100.0), atol=1e-6)

    def test_large_values_do_not_overflow(self):
        p = softmax(np.array([[1000.0, 1001.0]], dtype=np.float32))
        assert np.all(np.isfinite(p))
        assert p[0, 1] > p[0, 0]


class TestLogisticRegression:
    def test_separates_clean_blobs(self):
        x, y = blobs()
        model = train(x, y, class_count=3)
        predicted = np.argmax(model.predict_proba(x), axis=1)
        assert (predicted == y).mean() > 0.98

    def test_loss_decreases(self):
        x, y = blobs()
        early = train(x, y, class_count=3, iterations=1).final_loss
        late = train(x, y, class_count=3, iterations=200).final_loss
        assert late < early

    def test_shapes(self):
        x, y = blobs(dim=16, classes=4)
        model = train(x, y, class_count=4)
        assert model.weights.shape == (16, 4)
        assert model.biases.shape == (4,)
        assert model.feature_dim == 16 and model.class_count == 4

    def test_predict_matches_predict_proba(self):
        x, y = blobs()
        model = train(x, y, class_count=3)
        index, probs = model.predict(x[0])
        assert index == int(np.argmax(probs))
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_l2_shrinks_weights(self):
        x, y = blobs()
        weak = train(x, y, class_count=3, l2=0.0)
        strong = train(x, y, class_count=3, l2=1.0)
        assert np.abs(strong.weights).sum() < np.abs(weak.weights).sum()

    def test_is_deterministic(self):
        x, y = blobs()
        a = train(x, y, class_count=3)
        b = train(x, y, class_count=3)
        assert np.array_equal(a.weights, b.weights)

    def test_progress_callback_reports_every_iteration(self):
        x, y = blobs()
        seen: list[tuple[int, int]] = []
        train(x, y, class_count=3, iterations=10, progress=lambda d, t: seen.append((d, t)))
        assert seen[0] == (1, 10) and seen[-1] == (10, 10)

    @pytest.mark.parametrize("bad", [
        dict(x=np.zeros((4, 3), dtype=np.float32), labels=np.zeros(3), class_count=2),
        dict(x=np.zeros((4, 3), dtype=np.float32), labels=np.zeros(4), class_count=1),
        dict(x=np.zeros((0, 3), dtype=np.float32), labels=np.zeros(0), class_count=2),
    ])
    def test_invalid_input_raises(self, bad):
        with pytest.raises(ValueError):
            train(bad["x"], bad["labels"], bad["class_count"])

    def test_out_of_range_label_raises(self):
        with pytest.raises(ValueError):
            train(np.zeros((4, 3), dtype=np.float32), np.array([0, 1, 2, 5]), class_count=3)


class TestStratifiedSplit:
    def test_every_class_appears_in_both_sides(self):
        labels = np.repeat([0, 1, 2], 50)
        train_idx, val_idx = stratified_split(labels, val_fraction=0.2)
        assert set(labels[train_idx]) == {0, 1, 2}
        assert set(labels[val_idx]) == {0, 1, 2}

    def test_proportions_are_respected(self):
        labels = np.repeat([0, 1], 100)
        train_idx, val_idx = stratified_split(labels, val_fraction=0.25)
        assert len(val_idx) == 50 and len(train_idx) == 150

    def test_partition_is_exact(self):
        labels = np.repeat([0, 1, 2], 33)
        train_idx, val_idx = stratified_split(labels)
        assert sorted(np.concatenate([train_idx, val_idx])) == list(range(99))

    def test_singleton_class_stays_in_train(self):
        labels = np.array([0] * 20 + [1])
        train_idx, val_idx = stratified_split(labels, val_fraction=0.5)
        assert 20 in train_idx and 20 not in val_idx

    def test_is_seeded(self):
        labels = np.repeat([0, 1], 50)
        a, _ = stratified_split(labels, seed=7)
        b, _ = stratified_split(labels, seed=7)
        assert np.array_equal(a, b)


class TestZScore:
    def test_round_trip_gives_unit_variance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 3, size=(200, 6)).astype(np.float32)
        mean, std = zscore_fit(x)
        z = zscore_apply(x, mean, std)
        assert np.allclose(z.mean(axis=0), 0, atol=1e-4)
        assert np.allclose(z.std(axis=0), 1, atol=1e-4)

    def test_constant_column_does_not_divide_by_zero(self):
        x = np.ones((10, 2), dtype=np.float32)
        mean, std = zscore_fit(x)
        assert np.all(std == 1.0)
        assert np.all(np.isfinite(zscore_apply(x, mean, std)))


class TestSplitMix64:
    def test_is_deterministic_for_a_seed(self):
        assert [SplitMix64(42).next_uint64() for _ in range(3)] == \
               [SplitMix64(42).next_uint64() for _ in range(3)]

    def test_stays_in_64_bit_range(self):
        rng = SplitMix64(1)
        assert all(0 <= rng.next_uint64() < 2 ** 64 for _ in range(100))

    def test_gaussian_is_roughly_standard(self):
        rng = SplitMix64(42)
        values = np.array([rng.next_gaussian() for _ in range(20_000)])
        assert abs(values.mean()) < 0.05
        assert abs(values.std() - 1.0) < 0.05


class TestTSNE:
    def test_shape_and_finiteness(self):
        x, _ = blobs(n_per_class=40, dim=10)
        y = embed(x, TSNEConfig(iterations=120, perplexity=10))
        assert y.shape == (120, 2)
        assert np.all(np.isfinite(y))

    def test_is_seeded(self):
        x, _ = blobs(n_per_class=20, dim=6)
        config = TSNEConfig(iterations=60, perplexity=5)
        assert np.allclose(embed(x, config), embed(x, config))

    def test_different_seeds_differ(self):
        x, _ = blobs(n_per_class=20, dim=6)
        a = embed(x, TSNEConfig(iterations=60, perplexity=5, seed=1))
        b = embed(x, TSNEConfig(iterations=60, perplexity=5, seed=2))
        assert not np.allclose(a, b)

    def test_separates_blobs(self):
        """Within-class spread must be smaller than between-class spread."""
        x, labels = blobs(n_per_class=40, dim=10, classes=3, seed=3)
        y = embed(x, TSNEConfig(iterations=350, perplexity=10))

        centres = np.stack([y[labels == k].mean(axis=0) for k in range(3)])
        within = np.mean([np.linalg.norm(y[labels == k] - centres[k], axis=1).mean()
                          for k in range(3)])
        between = np.mean([np.linalg.norm(centres[i] - centres[j])
                           for i in range(3) for j in range(i + 1, 3)])
        assert between > 3 * within

    def test_output_is_centred(self):
        x, _ = blobs(n_per_class=20, dim=6)
        y = embed(x, TSNEConfig(iterations=80, perplexity=5))
        assert np.allclose(y.mean(axis=0), 0, atol=1e-4)

    def test_too_few_points_raises(self):
        with pytest.raises(TSNEError):
            embed(np.zeros((5, 4), dtype=np.float32))

    def test_perplexity_too_large_raises(self):
        with pytest.raises(TSNEError, match="too large"):
            embed(np.zeros((20, 4), dtype=np.float32), TSNEConfig(perplexity=30))

    def test_progress_is_reported(self):
        x, _ = blobs(n_per_class=20, dim=6)
        seen: list[str] = []
        embed(x, TSNEConfig(iterations=40, perplexity=5),
              progress=lambda d, t, m: seen.append(m))
        assert any("distance" in m for m in seen)
        assert any("Iteration" in m for m in seen)


class TestStratifiedSubsample:
    def test_returns_everything_when_under_the_cap(self):
        labels = ["a"] * 10 + ["b"] * 10
        assert len(stratified_subsample(labels, max_points=100)) == 20

    def test_respects_the_cap(self):
        labels = ["a"] * 5000 + ["b"] * 5000
        assert len(stratified_subsample(labels, max_points=1000)) <= 1000

    def test_keeps_class_proportions(self):
        labels = np.array(["a"] * 9000 + ["b"] * 1000)
        idx = stratified_subsample(list(labels), max_points=1000)
        share = np.mean(labels[idx] == "b")
        assert 0.05 < share < 0.15    # ~10%, as in the source distribution

    def test_rare_class_survives(self):
        labels = ["a"] * 9990 + ["b"] * 10
        idx = stratified_subsample(labels, max_points=500)
        assert any(labels[i] == "b" for i in idx)


class TestCentroids:
    def test_means_are_per_class(self):
        x = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        centroids = CentroidSet.compute(x, ["a", "a", "b"])
        assert centroids.labels == ["a", "b"]
        assert np.allclose(centroids.centroids[0].feature_mean, [1.0, 0.0])
        assert centroids.centroids[0].count == 2

    def test_classify_picks_the_nearest(self):
        x = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        centroids = CentroidSet.compute(x, ["near", "far"])
        label, distances = centroids.classify(np.array([0.5, 0.5], dtype=np.float32))
        assert label == "near"
        assert len(distances) == 2

    def test_plot_positions_when_embedding_given(self):
        x = np.array([[0.0], [1.0]], dtype=np.float32)
        points = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        centroids = CentroidSet.compute(x, ["a", "b"], points=points)
        assert centroids.centroids[0].plot_x == pytest.approx(1.0)
        assert centroids.centroids[1].plot_y == pytest.approx(4.0)

    def test_promotion_preserves_the_decision(self):
        """Softmax argmax over the promoted model must equal nearest-centroid."""
        rng = np.random.default_rng(0)
        x = rng.normal(0, 3, size=(60, 5)).astype(np.float32)
        labels = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
        centroids = CentroidSet.compute(x, labels)
        weights, biases, _ = centroids.to_linear_model()

        for feature in rng.normal(0, 3, size=(50, 5)).astype(np.float32):
            expected = centroids.classify(feature)[0]
            got = centroids.labels[int(np.argmax(feature @ weights + biases))]
            assert got == expected

    def test_dimension_mismatch_raises(self):
        centroids = CentroidSet.compute(np.zeros((2, 4), dtype=np.float32), ["a", "b"])
        with pytest.raises(ValueError):
            centroids.classify(np.zeros(3, dtype=np.float32))

    def test_empty_set_raises(self):
        with pytest.raises(ValueError):
            CentroidSet([])
