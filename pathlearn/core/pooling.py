"""Feature pooling — from ``Models/FeaturePooling.swift``.

Aggregates a *bag* of per-patch vectors into one fixed-length descriptor for a
whole annotation, as ``mean || max || std`` (so the output is ``3 * D``).

Rationale, carried over from the Swift: PaNIN grade is a per-lesion label, not a
per-cell one, and a pathologist grades the whole duct — taking the worst focus.
The mean captures the dominant appearance, the max the most atypical focus, and
the std the heterogeneity that grade-3 lesions show more of.

``04-DESIGN-DECISIONS.md`` §2 records that pooling alone was data-starved at
~136 annotations and collapsed to the majority class; it is kept because it
becomes viable with more annotations, not because it currently wins.
"""

from __future__ import annotations

import numpy as np

#: Stored on the classifier so prediction knows to expect pooled features.
MEAN_MAX_STD_ID = "meanmaxstd"
#: Output-dimension multiplier for the mean||max||std scheme.
MEAN_MAX_STD_MULTIPLIER = 3


def mean_max_std(vectors: np.ndarray) -> np.ndarray | None:
    """Pool ``(n, d)`` vectors into one ``(3*d,)`` descriptor.

    Returns ``None`` for an empty bag.  The std is the **population** std
    (divisor ``n``), matching the Swift.
    """
    if vectors is None:
        return None
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        return None

    return np.concatenate([
        array.mean(axis=0),
        array.max(axis=0),
        array.std(axis=0),
    ]).astype(np.float32)


def pooled_dimension(feature_dim: int) -> int:
    return feature_dim * MEAN_MAX_STD_MULTIPLIER
