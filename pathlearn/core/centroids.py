"""Class centroids and nearest-centroid classification — ``Models/Centroids.swift``.

Per-class mean vector in feature space plus a mean position in the t-SNE plot,
classified by minimum L2 distance.  A centroid set can be *promoted* to a
softmax classifier so it saves and predicts through the same path as a trained
logistic model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Centroid:
    label: str
    #: Mean vector in feature space, ``(d,)``.
    feature_mean: np.ndarray
    #: Mean position in the 2-D plot, or ``None`` when no embedding is attached.
    plot_x: float | None
    plot_y: float | None
    count: int


class CentroidSet:
    """Centroids for every class present in a bank."""

    def __init__(self, centroids: list[Centroid]) -> None:
        if not centroids:
            raise ValueError("A centroid set needs at least one class")
        dims = {c.feature_mean.shape[0] for c in centroids}
        if len(dims) != 1:
            raise ValueError(f"Centroids disagree on feature dim: {sorted(dims)}")
        self.centroids = centroids
        self.feature_dim = dims.pop()

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.centroids]

    @property
    def matrix(self) -> np.ndarray:
        """Stacked centroids, ``(k, d)``."""
        return np.stack([c.feature_mean for c in self.centroids])

    def classify(self, feature: np.ndarray) -> tuple[str, np.ndarray]:
        """Nearest centroid by L2, returning the label and all distances."""
        feature = np.asarray(feature, dtype=np.float32).ravel()
        if feature.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature dim {feature.shape[0]} does not match centroids ({self.feature_dim})"
            )
        distances = np.linalg.norm(self.matrix - feature, axis=1)
        return self.centroids[int(np.argmin(distances))].label, distances

    @classmethod
    def compute(cls, features: np.ndarray, labels: list[str],
                points: np.ndarray | None = None) -> "CentroidSet":
        """Per-class means over ``(n, d)`` *features*.

        *points* is the optional ``(n, 2)`` t-SNE embedding; when given, each
        centroid also carries its mean plot position so the plot can draw them.
        """
        features = np.asarray(features, dtype=np.float32)
        labels_array = np.asarray(labels)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got {features.shape}")
        if labels_array.shape[0] != features.shape[0]:
            raise ValueError("features and labels must have the same length")

        centroids: list[Centroid] = []
        for label in sorted(set(labels_array.tolist())):
            mask = labels_array == label
            subset = features[mask]
            plot_x = plot_y = None
            if points is not None and len(points):
                plot = np.asarray(points, dtype=np.float32)[mask]
                if plot.size:
                    plot_x, plot_y = float(plot[:, 0].mean()), float(plot[:, 1].mean())
            centroids.append(Centroid(
                label=str(label),
                feature_mean=subset.mean(axis=0).astype(np.float32),
                plot_x=plot_x,
                plot_y=plot_y,
                count=int(mask.sum()),
            ))
        return cls(centroids)

    def to_linear_model(self, temperature: float | None = None
                        ) -> tuple[np.ndarray, np.ndarray, float]:
        """Weights/biases whose softmax argmax equals nearest-centroid.

        Expanding ``-||f - c_k||^2 = -||f||^2 + 2 f.c_k - ||c_k||^2`` and
        dropping the ``-||f||^2`` term (constant across classes, so it cannot
        change the argmax) gives ``W_k = 2*c_k/T`` and ``b_k = -||c_k||^2/T``.

        *temperature* only scales confidence, never the decision: larger T gives
        softer probabilities.  The default ``max(1, feature_dim)`` keeps logits
        in a sane range for high-dimensional embeddings.
        """
        t = float(temperature) if temperature is not None else max(1.0, float(self.feature_dim))
        matrix = self.matrix
        weights = (2.0 * matrix / t).T.astype(np.float32)          # (d, k)
        biases = (-np.einsum("kd,kd->k", matrix, matrix) / t).astype(np.float32)
        return weights, biases, t
