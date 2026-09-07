"""Geometry descriptor computation and training — ``Models/GeometryTrainer.swift``.

Two jobs: turn annotations into 14-D descriptors, and train a grade classifier
over them with stratified k-fold cross-validation.

WHY CROSS-VALIDATION RATHER THAN A HOLD-OUT SPLIT
=================================================
There is one descriptor per *annotation*, so a bank is tens to low hundreds of
examples, not thousands.  A single 20% hold-out would put a handful of examples
in validation and produce a number that moves several points depending on the
seed.  k-fold uses every example for validation exactly once, which is the only
way to get a stable estimate at this scale — and it is what the Python
validation of this approach (~62% 5-fold) reported, so the numbers stay
comparable.

``k`` is capped by the smallest class, since a fold that cannot contain one
example of a class cannot validate it.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..core.geometry import DESCRIPTOR_NAMES, DIMENSION, VERSION, GeometryConfig, describe
from ..core.logistic import train as train_logistic, zscore_apply, zscore_fit
from ..core.panin import PANIN_VERSION, describe_panin
from ..core.shape import SHAPE_VERSION, describe_shape
from ..data.geometry_bank import FeatureSource, GeometryBank, GeometryRecord
from ..io.slide import SlideImage
from ..models.annotation import Annotation
from ..models.classifier import SOURCE_GEOMETRY, KIND_LOGISTIC, MLClassifier
from ..models.metrics import TrainingMetrics, evaluate

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]
CancelFn = Callable[[], bool]

DEFAULT_FOLDS = 5


class Validation(enum.Enum):
    """How held-out data is chosen.

    ``STRATIFIED`` splits annotations at random, keeping class balance. It is
    the standard choice and what the macOS trainer did — but every slide then
    appears in both train and test, so the model may be rewarded for
    memorising a slide's staining, scanner and the annotator's habits on that
    day rather than for learning grade.

    ``BY_SLIDE`` holds out one whole slide per fold. Nothing from a validation
    slide is ever seen in training, so the score answers the question that
    actually matters: does this generalise to a slide it has never seen?
    It is always the lower and more honest number.
    """

    STRATIFIED = "stratified"
    BY_SLIDE = "bySlide"

    @property
    def label(self) -> str:
        return {Validation.STRATIFIED: "Stratified k-fold (optimistic)",
                Validation.BY_SLIDE: "Leave-one-slide-out (honest)"}[self]
#: Geometry descriptors have no extractor; this marks that on the saved model.
GEOMETRY_IDENTITY = "geometry:handcrafted:r2"


class GeometryError(ValueError):
    """Raised when descriptors cannot be computed or trained."""


@dataclass
class DescribeReport:
    computed: int = 0
    skipped: int = 0
    with_shape: int = 0
    with_texture: int = 0
    with_panin: int = 0
    texture_skipped: int = 0
    cancelled: bool = False
    reasons: dict[str, int] = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = {}

    def summary(self) -> str:
        if self.computed == 0 and self.skipped == 0:
            return "No annotations to describe."
        text = (f"Described {self.computed} annotation(s): "
                f"{self.with_shape} with shape, {self.with_texture} with texture, "
                f"{self.with_panin} with PanIN architecture")
        if self.texture_skipped:
            text += f"; texture unavailable for {self.texture_skipped}"
            if self.reasons:
                text += " (" + ", ".join(f"{k}" for k in sorted(self.reasons)) + ")"
        if self.skipped:
            text += f"; {self.skipped} unusable"
        return text + ("  (cancelled early)" if self.cancelled else ".")


def describe_annotations(slide: SlideImage, annotations: Sequence[Annotation],
                         bank: GeometryBank,
                         config: GeometryConfig | None = None,
                         *, progress: ProgressFn | None = None,
                         should_cancel: CancelFn | None = None) -> DescribeReport:
    """Compute and store a descriptor for each annotation.

    Annotations that cannot support a descriptor — degenerate polygons, or too
    few nuclei at the analysis resolution — are skipped with a reason rather
    than stored as a zero vector, which would be indistinguishable from a real
    measurement of an empty region.
    """
    config = config or GeometryConfig()
    report = DescribeReport()
    usable = [a for a in annotations if not a.is_subtractive]
    if not usable:
        return report

    fresh: list[GeometryRecord] = []
    for index, annotation in enumerate(usable):
        if should_cancel and should_cancel():
            report.cancelled = True
            break
        if progress:
            progress(index, len(usable),
                     f"{annotation.classification}: {report.computed} described")

        # Shape first: it reads no pixels and almost never fails, so an
        # annotation too small for interior analysis still yields a usable
        # record instead of being dropped entirely.
        shape = describe_shape(annotation, mpp=slide.mpp_x or config.fallback_mpp)
        shape_features = shape.features if shape is not None else None

        try:
            features = describe(slide, annotation, config)
        except Exception as exc:
            log.debug("texture descriptor failed for %s: %s", annotation.id, exc)
            features = None

        # Independent of the texture block: PanIN architecture needs only a
        # lumen and some nuclei, so it survives annotations too small for
        # the interior descriptor.
        try:
            panin = describe_panin(slide, annotation, config)
        except Exception as exc:
            log.debug("panin descriptor failed for %s: %s", annotation.id, exc)
            panin = None

        if features is None:
            reason = (f"too few nuclei at {config.target_mpp:g} µm/px "
                      f"(needs {config.min_nuclei})")
            report.reasons[reason] = report.reasons.get(reason, 0) + 1
            report.texture_skipped += 1

        if features is None and shape_features is None and panin is None:
            report.skipped += 1
            continue

        fresh.append(GeometryRecord(
            slide_path=str(slide.path), slide_name=slide.name,
            annotation_id=annotation.id, classification=annotation.classification,
            features=features, shape_features=shape_features,
            panin_features=panin,
            version=VERSION, shape_version=SHAPE_VERSION,
            panin_version=PANIN_VERSION))
        report.computed += 1
        if shape_features is not None:
            report.with_shape += 1
        if features is not None:
            report.with_texture += 1
        if panin is not None:
            report.with_panin += 1

    if fresh:
        bank.add(fresh)
    if progress:
        progress(len(usable), len(usable), report.summary())
    return report


@dataclass(frozen=True, slots=True)
class GeometryTrainingSettings:
    source: FeatureSource = FeatureSource.SHAPE
    class_labels: Sequence[str] = ()
    validation: Validation = Validation.BY_SLIDE
    folds: int = DEFAULT_FOLDS
    iterations: int = 400
    learning_rate: float = 0.1
    l2: float = 0.01
    seed: int = 42


def train_geometry_classifier(bank: GeometryBank,
                              settings: GeometryTrainingSettings | None = None,
                              *, progress: ProgressFn | None = None,
                              should_cancel: CancelFn | None = None) -> MLClassifier:
    """Train a grade classifier on the 14-D descriptors, scored by k-fold CV."""
    settings = settings or GeometryTrainingSettings()

    def report(done: int, total: int, message: str) -> None:
        if progress:
            progress(done, total, message)

    report(0, 100, "Loading descriptors…")
    if not bank.records:
        raise GeometryError(
            "The geometry bank is empty. Compute descriptors for some "
            "annotations first.")

    records = bank.usable_for(settings.source)
    if not records:
        available = bank.stats()
        raise GeometryError(
            f"No records carry the {settings.source.label.lower()} block at the "
            f"current descriptor version. The bank holds {available.with_shape} "
            f"with shape and {available.with_texture} with texture. "
            f"Recompute descriptors, or pick a source the bank actually has.")

    wanted = set(settings.class_labels) if settings.class_labels else None
    if wanted:
        records = [r for r in records if r.classification in wanted]

    features, labels = bank.matrix(records, settings.source)
    class_labels = sorted(set(labels))
    if len(class_labels) < 2:
        raise GeometryError(
            f"Training needs at least 2 classes, found {len(class_labels)}.")

    counts = {name: labels.count(name) for name in class_labels}
    smallest = min(counts.values())
    if smallest < 2:
        thin = ", ".join(f"{n} ({c})" for n, c in sorted(counts.items()) if c < 2)
        raise GeometryError(
            f"These classes have too few annotations to cross-validate: {thin}. "
            f"Each class needs at least 2.")

    index_of = {name: i for i, name in enumerate(class_labels)}
    y = np.array([index_of[name] for name in labels], dtype=np.intp)

    # Descriptors mix counts per mm², µm² areas and unit fractions, so they are
    # on wildly different scales — z-scoring is mandatory here, not optional.
    mean, std = zscore_fit(features)
    scaled = zscore_apply(features, mean, std)

    if settings.validation is Validation.BY_SLIDE:
        folds = _slide_folds(records)
        if len(folds) < 2:
            raise GeometryError(
                f"Leave-one-slide-out needs annotations from at least 2 slides; "
                f"this selection covers {len(folds)}. Describe annotations on "
                f"more slides, or switch to stratified k-fold — but that score "
                f"cannot tell you whether the model generalises to a new slide.")
        unit = "slide"
    else:
        folds = _stratified_folds(y, max(2, min(settings.folds, smallest)), settings.seed)
        unit = "fold"
    report(20, 100, f"Validating over {len(folds)} held-out {unit}(s)…")

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    fold_accuracies: list[float] = []
    for number, validation in enumerate(folds):
        if should_cancel and should_cancel():
            raise GeometryError("Cancelled.")
        training = np.array([i for i in range(len(y)) if i not in set(validation)],
                            dtype=np.intp)
        if training.size == 0 or len(validation) == 0:
            continue
        # A held-out slide may leave the training set with only one class — the
        # trainer cannot fit that, and the fold is simply unusable rather than
        # an error. It is dropped from the pooled estimate.
        if len(np.unique(y[training])) < 2:
            log.info("Skipping held-out %s %d: training set has one class", unit, number)
            continue
        model = train_logistic(scaled[training], y[training], len(class_labels),
                               iterations=settings.iterations,
                               learning_rate=settings.learning_rate, l2=settings.l2,
                               should_cancel=should_cancel)
        predicted = np.argmax(model.predict_proba(scaled[validation]), axis=1)
        accuracy, _, fold_confusion = evaluate(y[validation], predicted, class_labels)
        confusion += np.asarray(fold_confusion, dtype=int)
        fold_accuracies.append(accuracy)
        report(20 + int(60 * (number + 1) / len(folds)), 100,
               f"Fold {number + 1}/{len(folds)}: {accuracy:.1%}")

    # Per-class scores come from the pooled out-of-fold confusion matrix, so
    # every example contributes to validation exactly once.
    per_class = _per_class_from_confusion(confusion, class_labels)
    cv_accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))

    report(85, 100, "Fitting the final model on all data…")
    if should_cancel and should_cancel():
        raise GeometryError("Cancelled.")
    final = train_logistic(scaled, y, len(class_labels),
                           iterations=settings.iterations,
                           learning_rate=settings.learning_rate, l2=settings.l2,
                           should_cancel=should_cancel)
    train_predicted = np.argmax(final.predict_proba(scaled), axis=1)
    train_accuracy, _, _ = evaluate(y, train_predicted, class_labels)

    metrics = TrainingMetrics(
        class_labels=class_labels,
        train_accuracy=train_accuracy,
        val_accuracy=cv_accuracy,
        per_class=per_class,
        confusion=confusion.tolist(),
        final_loss=final.final_loss,
        train_count=len(y),
        val_count=int(confusion.sum()),
        extractor_revision=VERSION,
    )

    report(100, 100, f"{len(folds)}-fold CV: {cv_accuracy:.1%}")
    return MLClassifier(
        class_labels=class_labels,
        weights=final.weights,
        biases=final.biases,
        extractor_identity=GEOMETRY_IDENTITY,
        kind=KIND_LOGISTIC,
        feature_source=SOURCE_GEOMETRY,
        aggregation=settings.source.value,
        feature_mean=mean,
        feature_std=std,
        metrics=metrics,
    )


def _slide_folds(records: Sequence[GeometryRecord]) -> list[list[int]]:
    """One fold per source slide — leave-one-slide-out.

    This is the honest protocol for this data. Annotations from one slide share
    its staining, scanner, section thickness and whatever tracing habits the
    annotator had that session; splitting them at random lets a model score well
    by recognising the slide rather than the grade. Holding out whole slides
    removes that shortcut.

    Expect a materially lower number than stratified k-fold. The gap between
    them *is* the measurement of how much the model was leaning on slide
    identity.
    """
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        key = record.slide_path or record.slide_name or f"unknown-{index}"
        groups.setdefault(key, []).append(index)
    return [sorted(indices) for _, indices in sorted(groups.items())]


def _stratified_folds(y: np.ndarray, k: int, seed: int) -> list[list[int]]:
    """Shuffle each class's indices and deal them round-robin into *k* folds.

    Dealing round-robin (rather than slicing) keeps class proportions even when
    a class size is not a multiple of k — with 7 examples over 5 folds, slicing
    would leave two folds empty of that class.
    """
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % k].append(int(index))
    return [sorted(f) for f in folds if f]


def _per_class_from_confusion(confusion: np.ndarray, class_labels: Sequence[str]):
    from ..models.metrics import ClassMetrics

    out = []
    for i, label in enumerate(class_labels):
        true_positive = int(confusion[i, i])
        predicted = int(confusion[:, i].sum())
        actual = int(confusion[i, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        out.append(ClassMetrics(label, precision, recall, f1, actual))
    return out


def descriptor_table(record_features: np.ndarray) -> list[tuple[str, float]]:
    """Name/value pairs for one descriptor, for display."""
    return [(name, float(value))
            for name, value in zip(DESCRIPTOR_NAMES, np.asarray(record_features).ravel())]
