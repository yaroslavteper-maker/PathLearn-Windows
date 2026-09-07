"""Multinomial logistic regression — from ``Models/LogisticRegression.swift``.

Softmax classifier trained by full-batch gradient descent.  The Swift used
Accelerate's ``cblas_sgemm``; NumPy's ``@`` reaches the same BLAS, so this is a
direct translation rather than a reimplementation.

Two places where the shipped Swift differs from ``03-MODELS-AND-ML.md`` §B1.
The code is ground truth (see the repo README's precedence rule), so this
follows the code:

* The doc writes the L2 gradient as ``+ l2*W``; the Swift adds ``2*l2*W``.
* The doc includes ``(l2/2)*||W||^2`` in the reported loss; the Swift reports
  plain cross-entropy.  Only the reported number differs, not the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

DEFAULT_ITERATIONS = 200
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_L2 = 1e-3
DEFAULT_VAL_FRACTION = 0.2
#: The Swift recomputes the loss only every N iterations, and on the last one.
LOSS_EVERY = 25


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """Weights ``D x K`` (row-major) and biases ``K``."""

    weights: np.ndarray
    biases: np.ndarray
    feature_dim: int
    class_count: int
    final_loss: float

    def decision_scores(self, features: np.ndarray) -> np.ndarray:
        return np.atleast_2d(features) @ self.weights + self.biases

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return softmax(self.decision_scores(features))

    def predict(self, features: np.ndarray) -> tuple[int, np.ndarray]:
        """Argmax index and probability vector for a single feature vector."""
        probs = self.predict_proba(np.asarray(features, dtype=np.float32).ravel())[0]
        return int(np.argmax(probs)), probs


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    z = np.asarray(z, dtype=np.float32)
    shifted = z - z.max(axis=-1, keepdims=True)
    np.exp(shifted, out=shifted)
    return shifted / shifted.sum(axis=-1, keepdims=True)


def train(x: np.ndarray,
          labels: np.ndarray,
          class_count: int,
          iterations: int = DEFAULT_ITERATIONS,
          learning_rate: float = DEFAULT_LEARNING_RATE,
          l2: float = DEFAULT_L2,
          progress: Callable[[int, int], None] | None = None,
          should_cancel: Callable[[], bool] | None = None) -> TrainedModel:
    """Train on ``x`` (N x D float32) with integer *labels* in ``[0, K)``.

    *should_cancel* is checked inside the descent loop, not only between
    phases. Without that a Stop button appears to do nothing for as long as
    the fit runs, which on a large bank is the part worth interrupting.
    Cancelling returns the weights as they stand — partially trained, and
    the caller is expected to discard them.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.intp)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (N x D), got shape {x.shape}")
    n, d = x.shape
    if labels.shape != (n,):
        raise ValueError(f"labels must have length {n}, got {labels.shape}")
    if class_count < 2:
        raise ValueError("Need at least 2 classes to train")
    if n == 0:
        raise ValueError("No training samples")
    if labels.min() < 0 or labels.max() >= class_count:
        raise ValueError("labels out of range for class_count")

    k = class_count
    weights = np.zeros((d, k), dtype=np.float32)
    biases = np.zeros(k, dtype=np.float32)
    rows = np.arange(n)
    final_loss = 0.0

    for iteration in range(iterations):
        probs = softmax(x @ weights + biases)

        if iteration % LOSS_EVERY == 0 or iteration == iterations - 1:
            picked = np.maximum(probs[rows, labels], 1e-9)
            final_loss = float(-np.log(picked).mean())

        # gradient = P - one_hot(y), then averaged over the batch.
        grad = probs
        grad[rows, labels] -= 1.0
        grad /= np.float32(n)

        d_weights = x.T @ grad
        if l2 > 0:
            d_weights += np.float32(2 * l2) * weights
        d_biases = grad.sum(axis=0)

        weights -= np.float32(learning_rate) * d_weights
        biases -= np.float32(learning_rate) * d_biases

        if progress is not None:
            progress(iteration + 1, iterations)
        if should_cancel is not None and should_cancel():
            break

    return TrainedModel(weights, biases, d, k, final_loss)


def predict(features: np.ndarray, weights: np.ndarray, biases: np.ndarray,
            class_count: int) -> tuple[int, np.ndarray]:
    """Single-vector prediction, mirroring the Swift free function."""
    features = np.asarray(features, dtype=np.float32).ravel()
    probs = softmax(features @ weights + biases)
    return int(np.argmax(probs)), probs


def stratified_split(labels: np.ndarray, val_fraction: float = DEFAULT_VAL_FRACTION,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Per-class train/validation index split.

    Every class contributes the same *fraction* to validation, so a rare class
    cannot vanish from the validation set and silently inflate accuracy.  A
    class with a single sample stays wholly in train.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []

    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        # Never empty the training side of a class.
        n_val = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])

    return (np.concatenate(train_idx) if train_idx else np.zeros(0, dtype=np.intp),
            np.concatenate(val_idx) if val_idx else np.zeros(0, dtype=np.intp))


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std, with zero-variance columns forced to std 1."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)
