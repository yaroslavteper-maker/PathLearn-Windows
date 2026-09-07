"""t-SNE — from ``Models/TSNE.swift`` (van der Maaten 2008, symmetric).

A direct translation of the shipped Swift, including its simplifications: no
adaptive gains, no Barnes-Hut, plain momentum.  At the bank sizes this app uses
(stratified-subsampled to <= 5000 points) the exact O(N^2) gradient is fine and
keeps the embedding reproducible against the macOS build.

Determinism: the Swift seeded a splitmix64 PRNG and drew the initial embedding
from ``N(0, 1e-4)`` via Box-Muller.  :class:`SplitMix64` reproduces that bit for
bit, so the same seed gives the same layout as the Swift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

ProgressFn = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class TSNEConfig:
    perplexity: float = 30.0
    iterations: int = 1000
    early_exaggeration: float = 12.0
    early_exaggeration_iters: int = 250
    learning_rate: float = 200.0
    initial_momentum: float = 0.5
    final_momentum: float = 0.8
    momentum_switch_iter: int = 250
    seed: int = 42


class TSNEError(ValueError):
    """Raised when the input cannot support a meaningful embedding."""


class SplitMix64:
    """The Swift's seeded PRNG, reproduced exactly (64-bit wrapping)."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int = 42) -> None:
        self._state = seed & self._MASK

    def next_uint64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & self._MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (z ^ (z >> 31)) & self._MASK

    def next_double(self) -> float:
        """Uniform in [0, 1), from the top 53 bits."""
        return (self.next_uint64() >> 11) * (1.0 / (1 << 53))

    def next_gaussian(self) -> float:
        """Standard normal via Box-Muller, matching the Swift."""
        u1 = max(self.next_double(), 1e-300)
        u2 = self.next_double()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def embed(features: np.ndarray,
          config: TSNEConfig | None = None,
          progress: ProgressFn | None = None) -> np.ndarray:
    """Embed ``(N, D)`` *features* into ``(N, 2)``.

    Raises :class:`TSNEError` for fewer than 10 points, or when perplexity is
    too large for the sample count (``perplexity * 3 >= N``) — the same guards
    the Swift applies, since beyond them the binary search cannot converge.
    """
    config = config or TSNEConfig()
    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise TSNEError(f"features must be 2-D (N x D), got {features.shape}")
    n = features.shape[0]
    if n < 10:
        raise TSNEError(f"Need at least 10 points to embed, got {n}")
    if config.perplexity * 3 >= n:
        raise TSNEError(
            f"Perplexity {config.perplexity:g} is too large for {n} points "
            f"(needs perplexity * 3 < N); try {max(2, (n // 3) - 1)} or fewer."
        )

    _report(progress, 0, config.iterations, "Computing pairwise distances…")
    distances = _pairwise_squared_distances(features)

    _report(progress, 0, config.iterations, f"Solving {n} perplexities…")
    p = _symmetric_affinities(distances, config.perplexity)

    rng = SplitMix64(config.seed)
    y = np.array([rng.next_gaussian() * 1e-4 for _ in range(n * 2)],
                 dtype=np.float32).reshape(n, 2)

    velocity = np.zeros_like(y)
    diag = np.arange(n)

    for iteration in range(config.iterations):
        # Student-t numerator: 1 / (1 + ||y_i - y_j||^2), zero on the diagonal.
        y_norm2 = np.einsum("ij,ij->i", y, y)
        num = y_norm2[:, None] + y_norm2[None, :] - 2.0 * (y @ y.T)
        num += 1.0
        np.reciprocal(num, out=num)
        num[diag, diag] = 0.0

        z = max(float(num.sum()), 1e-12)
        exaggeration = (config.early_exaggeration
                        if iteration < config.early_exaggeration_iters else 1.0)

        # A = num * (P*exaggeration - num/Z)
        a = (p * np.float32(exaggeration)) - (num / z)
        a *= num

        grad = 4.0 * (y * a.sum(axis=1)[:, None] - (a @ y))

        momentum = (config.initial_momentum
                    if iteration < config.momentum_switch_iter else config.final_momentum)
        velocity *= np.float32(momentum)
        velocity -= np.float32(config.learning_rate) * grad
        y += velocity

        # Recentre for numerical stability.
        y -= y.mean(axis=0)

        if progress is not None and (iteration % 20 == 0 or iteration == config.iterations - 1):
            _report(progress, iteration + 1, config.iterations,
                    f"Iteration {iteration + 1} / {config.iterations}")

    return y


def _pairwise_squared_distances(x: np.ndarray) -> np.ndarray:
    norm2 = np.einsum("ij,ij->i", x, x)
    d = norm2[:, None] + norm2[None, :] - 2.0 * (x @ x.T)
    np.maximum(d, 0.0, out=d)   # kill negative round-off before the exp
    np.fill_diagonal(d, 0.0)
    return d


def _symmetric_affinities(distances: np.ndarray, perplexity: float,
                          tolerance: float = 1e-5, max_iters: int = 50) -> np.ndarray:
    """Binary-search a per-point sigma to match *perplexity*, then symmetrise.

    Vectorised over points: every row runs its own bisection, but all rows step
    together, which is what makes this tractable at N=5000.
    """
    n = distances.shape[0]
    log_perplexity = math.log(perplexity)
    d = distances.astype(np.float64)

    beta = np.ones(n, dtype=np.float64)
    beta_low = np.full(n, -np.inf)
    beta_high = np.full(n, np.inf)
    off_diagonal = ~np.eye(n, dtype=bool)

    p = np.zeros((n, n), dtype=np.float64)
    for _ in range(max_iters):
        np.exp(-d * beta[:, None], out=p)
        p[~off_diagonal] = 0.0
        sum_p = np.maximum(p.sum(axis=1), 1e-12)
        normalised = p / sum_p[:, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(normalised > 1e-12, np.log(normalised), 0.0)
        entropy = -(normalised * log_p).sum(axis=1)

        diff = entropy - log_perplexity
        if np.all(np.abs(diff) < tolerance):
            break

        # Entropy too high -> raise beta (narrower kernel), and vice versa.
        too_high = diff > 0
        beta_low = np.where(too_high, beta, beta_low)
        beta_high = np.where(too_high, beta_high, beta)
        beta = np.where(
            too_high,
            np.where(np.isfinite(beta_high), (beta + beta_high) / 2.0, beta * 2.0),
            np.where(np.isfinite(beta_low), (beta + beta_low) / 2.0, beta / 2.0),
        )

    # Final rows using the converged betas.
    np.exp(-d * beta[:, None], out=p)
    p[~off_diagonal] = 0.0
    p /= np.maximum(p.sum(axis=1), 1e-12)[:, None]

    symmetric = (p + p.T) / (2.0 * n)
    np.maximum(symmetric, 1e-12, out=symmetric)
    return symmetric.astype(np.float32)


def stratified_subsample(labels: list[str], max_points: int = 5000,
                         seed: int = 42) -> np.ndarray:
    """Indices of a class-proportional subsample of at most *max_points*.

    Keeping the class balance matters: an unbalanced subsample would make the
    embedding's visual density misleading about the bank's actual composition.
    """
    labels_array = np.asarray(labels)
    n = labels_array.shape[0]
    if n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    unique = np.unique(labels_array)
    selected: list[np.ndarray] = []
    for label in unique:
        idx = np.flatnonzero(labels_array == label)
        quota = max(1, int(round(len(idx) * max_points / n)))
        rng.shuffle(idx)
        selected.append(idx[:quota])

    out = np.concatenate(selected)
    if out.size > max_points:
        rng.shuffle(out)
        out = out[:max_points]
    return np.sort(out)


def _report(progress: ProgressFn | None, done: int, total: int, message: str) -> None:
    if progress is not None:
        progress(done, total, message)
