"""Classifier training — from ``Models/ClassifierTrainer.swift``.

Loads patches from the bank, applies the filters, optionally pools per
annotation, holds out a stratified validation split, trains, and reports
metrics.

THREE THINGS THAT ARE NOT OPTIONAL
==================================
1. **One extractor per model.**  Feature spaces cannot be mixed, so training
   selects exactly one ``extractor_identity`` and refuses ambiguity.
2. **Null classes never become real classes.**  A class marked ``isNull``
   (lumen-like) contributes a *reference set*, not a label
   (``04-DESIGN-DECISIONS.md`` §4).
3. **The split is stratified.**  With ~136 annotations over 10 classes, a random
   split can empty a rare class out of validation and make accuracy meaningless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.logistic import (DEFAULT_ITERATIONS, DEFAULT_L2, DEFAULT_LEARNING_RATE,
                             DEFAULT_VAL_FRACTION, stratified_split, train as train_logistic,
                             zscore_apply, zscore_fit)
from ..core.pooling import mean_max_std
from ..data.bank import Patch, PatchBank
from ..models.classifier import (AGGREGATION_MEAN_MAX_STD, DEFAULT_NULL_THRESHOLD,
                                 KIND_LOGISTIC, MAX_NULL_REFERENCE, MLClassifier)
from ..models.metrics import TrainingMetrics, evaluate

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]
CancelFn = Callable[[], bool]

#: Below this many patches in a class, training is refused rather than misleading.
MIN_PATCHES_PER_CLASS = 2


class TrainingError(ValueError):
    """Raised when the bank cannot support the requested training."""


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    extractor_identity: str
    #: Classes to train as real labels.  Empty means every non-null class present.
    class_labels: Sequence[str] = ()
    #: Classes whose patches form the null reference instead of a label.
    null_labels: Sequence[str] = ()
    null_threshold: float | None = None
    pooled: bool = False
    max_white_fraction: float | None = None
    min_nuclei: int | None = None
    iterations: int = DEFAULT_ITERATIONS
    learning_rate: float = DEFAULT_LEARNING_RATE
    l2: float = DEFAULT_L2
    val_fraction: float = DEFAULT_VAL_FRACTION
    seed: int = 42


def train_classifier(bank: PatchBank, settings: TrainingSettings, *,
                     progress: ProgressFn | None = None,
                     should_cancel: CancelFn | None = None) -> MLClassifier:
    """Train a classifier over *bank*.  Raises :class:`TrainingError` if it cannot."""

    def report(done: int, total: int, message: str) -> None:
        if progress:
            progress(done, total, message)

    report(0, 100, "Loading patches…")
    patches = bank.fetch(extractor_identity=settings.extractor_identity,
                         max_white_fraction=settings.max_white_fraction,
                         min_nuclei=settings.min_nuclei)
    if not patches:
        raise TrainingError(
            f"No patches in the bank for {settings.extractor_identity} "
            f"matching those filters.")

    null_set = set(settings.null_labels)
    wanted = set(settings.class_labels) if settings.class_labels else None

    real = [p for p in patches
            if p.classification not in null_set
            and (wanted is None or p.classification in wanted)]
    nulls = [p for p in patches if p.classification in null_set]

    if not real:
        raise TrainingError("Every selected class is marked null — nothing to train on.")

    report(15, 100, "Building the null reference…" if nulls else "Preparing features…")
    null_reference = _build_null_reference(nulls, settings.seed) if nulls else None
    null_threshold = (settings.null_threshold
                      if settings.null_threshold is not None
                      else (DEFAULT_NULL_THRESHOLD if null_reference is not None else None))

    if null_reference is not None and null_threshold is not None:
        before = len(real)
        real = _drop_null_like(real, null_reference, null_threshold)
        dropped = before - len(real)
        if dropped:
            log.info("Null filter removed %d of %d patches", dropped, before)
        if not real:
            raise TrainingError(
                f"The null filter removed all {before} training patches. "
                f"Raise the threshold above {null_threshold:g}.")

    if should_cancel and should_cancel():
        raise TrainingError("Cancelled.")

    report(35, 100, "Assembling the feature matrix…")
    if settings.pooled:
        features, labels = _pool_by_annotation(real)
        if features.shape[0] < 4:
            raise TrainingError(
                f"Pooling gives one vector per annotation, so this bank yields only "
                f"{features.shape[0]} training examples. Pooled training needs far "
                f"more annotations; use per-patch training instead.")
    else:
        features = np.stack([p.features for p in real]).astype(np.float32)
        labels = [p.classification for p in real]

    class_labels = sorted(set(labels))
    if len(class_labels) < 2:
        raise TrainingError(
            f"Training needs at least 2 classes, found {len(class_labels)} "
            f"({', '.join(class_labels) or 'none'}).")

    counts = {name: labels.count(name) for name in class_labels}
    thin = {n: c for n, c in counts.items() if c < MIN_PATCHES_PER_CLASS}
    if thin:
        raise TrainingError(
            "These classes have too few examples to train or validate: "
            + ", ".join(f"{n} ({c})" for n, c in sorted(thin.items())))

    label_index = {name: i for i, name in enumerate(class_labels)}
    y = np.array([label_index[name] for name in labels], dtype=np.intp)

    # Pooled features are z-scored; per-patch embeddings are left alone, which
    # is what the Swift did — the pooled vector concatenates statistics on very
    # different scales, so it needs it and raw embeddings do not.
    feature_mean = feature_std = None
    if settings.pooled:
        feature_mean, feature_std = zscore_fit(features)
        features = zscore_apply(features, feature_mean, feature_std)

    report(50, 100, "Splitting train and validation…")
    train_idx, val_idx = stratified_split(y, settings.val_fraction, settings.seed)
    if train_idx.size == 0:
        raise TrainingError("The training split came out empty.")

    if should_cancel and should_cancel():
        raise TrainingError("Cancelled.")

    report(60, 100, f"Training on {train_idx.size} examples…")
    model = train_logistic(
        features[train_idx], y[train_idx], class_count=len(class_labels),
        should_cancel=should_cancel,
        iterations=settings.iterations, learning_rate=settings.learning_rate,
        l2=settings.l2,
        progress=lambda done, total: report(60 + int(35 * done / max(total, 1)), 100,
                                            f"Training… iteration {done}/{total}"),
    )

    report(96, 100, "Scoring…")
    train_pred = np.argmax(model.predict_proba(features[train_idx]), axis=1)
    train_accuracy, _, _ = evaluate(y[train_idx], train_pred, class_labels)

    if val_idx.size:
        val_pred = np.argmax(model.predict_proba(features[val_idx]), axis=1)
        val_accuracy, per_class, confusion = evaluate(y[val_idx], val_pred, class_labels)
    else:
        val_accuracy, per_class, confusion = 0.0, [], []

    metrics = TrainingMetrics(
        class_labels=class_labels,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        per_class=per_class,
        confusion=confusion,
        final_loss=model.final_loss,
        train_count=int(train_idx.size),
        val_count=int(val_idx.size),
        extractor_revision=_revision(settings.extractor_identity),
    )

    report(100, 100, metrics.summary().splitlines()[0])
    return MLClassifier(
        class_labels=class_labels,
        weights=model.weights,
        biases=model.biases,
        extractor_identity=settings.extractor_identity,
        kind=KIND_LOGISTIC,
        aggregation=AGGREGATION_MEAN_MAX_STD if settings.pooled else None,
        feature_mean=feature_mean,
        feature_std=feature_std,
        null_reference=null_reference,
        null_threshold=null_threshold,
        metrics=metrics,
    )


def _build_null_reference(nulls: Sequence[Patch], seed: int) -> np.ndarray:
    """A capped, subsampled set of raw null vectors.

    Capped at :data:`MAX_NULL_REFERENCE` because the reference is stored inside
    every `.cl` file and compared against every candidate at predict time; an
    uncapped set would bloat the model and slow prediction for no gain.
    """
    vectors = np.stack([p.features for p in nulls]).astype(np.float32)
    if vectors.shape[0] > MAX_NULL_REFERENCE:
        rng = np.random.default_rng(seed)
        keep = rng.choice(vectors.shape[0], MAX_NULL_REFERENCE, replace=False)
        vectors = vectors[np.sort(keep)]
    return vectors


def _drop_null_like(patches: Sequence[Patch], reference: np.ndarray,
                    threshold: float) -> list[Patch]:
    """Remove patches whose cosine to the nearest null vector reaches *threshold*."""
    features = np.stack([p.features for p in patches]).astype(np.float32)
    a = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    b = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12)
    similarity = (a @ b.T).max(axis=1)
    return [p for p, s in zip(patches, similarity) if s < threshold]


def _pool_by_annotation(patches: Sequence[Patch]) -> tuple[np.ndarray, list[str]]:
    """One ``mean || max || std`` vector per annotation."""
    groups: dict[object, list[Patch]] = {}
    for patch in patches:
        groups.setdefault(patch.annotation_id, []).append(patch)

    vectors, labels = [], []
    for members in groups.values():
        pooled = mean_max_std(np.stack([p.features for p in members]))
        if pooled is None:
            continue
        vectors.append(pooled)
        labels.append(members[0].classification)

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(vectors).astype(np.float32), labels


def _revision(identity: str) -> int:
    from ..extractors.identity import ExtractorIdentity
    try:
        return ExtractorIdentity.parse(identity).revision
    except ValueError:
        return 1
