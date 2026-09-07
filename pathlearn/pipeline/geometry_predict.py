"""Grade traced annotations with a geometry model.

A geometry model classifies a *polygon*, not a tile. Its input is the 12-D
outline descriptor (and/or the 14-D interior block) computed from one traced
lesion, so there is nothing to feed it per-224px-patch — the tile prediction
pipeline in :mod:`pathlearn.pipeline.predict` structurally cannot carry it.
This module is the inference path that fits: one verdict per annotation.

The descriptors are computed exactly as :func:`describe_annotations` computes
them for training, and by the same functions, so a lesion scores the same
whether it went into the bank or through here.

Nothing is written back to the store: grading produces a report, and applying
those labels to the annotations is a separate, explicit act by the user.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.geometry import GeometryConfig, describe
from ..core.panin import describe_panin
from ..core.shape import describe_shape
from ..data.geometry_bank import FeatureSource
from ..io.slide import SlideImage
from ..models.annotation import Annotation
from ..models.classifier import MLClassifier
from .geometry_train import GeometryError, ProgressFn, CancelFn

log = logging.getLogger(__name__)


def source_of(classifier: MLClassifier) -> FeatureSource:
    """Which descriptor block(s) *classifier* was trained on.

    The trainer stores the source in ``aggregation`` — the field is otherwise
    unused for geometry models, and it round-trips through ``.cl`` — so a model
    loaded from disk still knows whether its 12 inputs are shape or its 26 are
    shape+texture. Older files that predate the shape block have no value
    there, so fall back to matching on width.
    """
    if not classifier.is_geometry:
        raise GeometryError(
            f"{classifier.extractor_identity} is not a geometry model. Grading "
            f"reads a traced outline; this model reads extractor features, so "
            f"it belongs in Predict rather than here.")
    if classifier.aggregation:
        for source in FeatureSource:
            if source.value == classifier.aggregation:
                return source
    for source in FeatureSource:
        if source.dimension == classifier.feature_dim:
            return source
    raise GeometryError(
        f"This model takes {classifier.feature_dim}-d input, which matches no "
        f"descriptor block ("
        + ", ".join(f"{s.label.lower()} {s.dimension}-d" for s in FeatureSource)
        + "). It was probably trained against a different descriptor version.")


@dataclass(frozen=True, slots=True)
class Grade:
    """One annotation's verdict."""

    annotation_id: uuid.UUID
    display_name: str
    annotated_as: str
    predicted: str | None = None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    #: Why no verdict was produced, when ``predicted`` is None.
    skipped: str | None = None

    @property
    def agrees(self) -> bool:
        return self.predicted is not None and self.predicted == self.annotated_as

    @property
    def changed(self) -> bool:
        """A verdict that would relabel the annotation if applied."""
        return self.predicted is not None and self.predicted != self.annotated_as


@dataclass
class GradeReport:
    source: FeatureSource
    class_labels: list[str]
    grades: list[Grade] = field(default_factory=list)
    cancelled: bool = False

    @property
    def graded(self) -> list[Grade]:
        return [g for g in self.grades if g.predicted is not None]

    @property
    def skipped(self) -> list[Grade]:
        return [g for g in self.grades if g.predicted is None]

    @property
    def agreement(self) -> float | None:
        """Fraction of graded annotations whose verdict matches their label.

        This is **not** an accuracy estimate. Most of these annotations were
        almost certainly in the model's own training set, so agreement here is
        closer to a memorisation check than a measurement. The honest number is
        the leave-one-slide-out score from training.
        """
        graded = self.graded
        if not graded:
            return None
        return sum(1 for g in graded if g.agrees) / len(graded)

    def summary(self) -> str:
        if not self.grades:
            return "Nothing to grade."
        parts = [f"Graded {len(self.graded)} of {len(self.grades)} annotation(s) "
                 f"on {self.source.label.lower()}"]
        agreement = self.agreement
        if agreement is not None:
            changed = sum(1 for g in self.graded if g.changed)
            parts.append(f"{agreement:.0%} agree with their current label "
                         f"({changed} would change)")
        if self.skipped:
            parts.append(f"{len(self.skipped)} could not be described")
        text = " · ".join(parts)
        return text + ("  (cancelled early)" if self.cancelled else ".")


def grade_annotations(slide: SlideImage, annotations: Sequence[Annotation],
                      classifier: MLClassifier,
                      config: GeometryConfig | None = None,
                      *, progress: ProgressFn | None = None,
                      should_cancel: CancelFn | None = None) -> GradeReport:
    """Predict a class for each of *annotations* using a geometry model.

    Annotations whose required descriptor block cannot be computed are reported
    with a reason rather than given a verdict — a zero vector would produce a
    confident-looking prediction from no measurement at all.
    """
    source = source_of(classifier)
    if classifier.feature_dim != source.dimension:
        raise GeometryError(
            f"Model expects {classifier.feature_dim}-d input but "
            f"{source.label.lower()} is {source.dimension}-d.")

    config = config or GeometryConfig()
    report = GradeReport(source=source, class_labels=list(classifier.class_labels))
    usable = [a for a in annotations if not a.is_subtractive]
    if not usable:
        return report

    mpp = slide.mpp_x or config.fallback_mpp
    for index, annotation in enumerate(usable):
        if should_cancel and should_cancel():
            report.cancelled = True
            break
        if progress:
            progress(index, len(usable), f"Grading {index + 1}/{len(usable)}…")

        vector, reason = _vector_for(slide, annotation, source, config, mpp)
        if vector is None:
            report.grades.append(Grade(annotation_id=annotation.id,
                                       display_name=_label_of(annotation),
                                       annotated_as=annotation.classification,
                                       skipped=reason))
            continue

        label, probabilities = classifier.predict(vector)
        report.grades.append(Grade(
            annotation_id=annotation.id,
            display_name=_label_of(annotation),
            annotated_as=annotation.classification,
            predicted=label,
            confidence=float(np.max(probabilities)),
            probabilities={name: float(p) for name, p
                           in zip(classifier.class_labels, probabilities)},
        ))

    if progress:
        progress(len(usable), len(usable), report.summary())
    return report


def _vector_for(slide: SlideImage, annotation: Annotation, source: FeatureSource,
                config: GeometryConfig, mpp: float
                ) -> tuple[np.ndarray | None, str | None]:
    """The model's input vector for one annotation, or a reason there is none.

    Blocks are concatenated shape-then-texture, matching ``GeometryBank.matrix``.
    That order is load-bearing: swapping it silently feeds each weight the wrong
    descriptor.

    Every ``FeatureSource`` must be handled explicitly. An unhandled one used
    to fall through to the concatenate below with both blocks still None,
    which numpy reports as "zero-dimensional arrays cannot be concatenated" —
    an error that says nothing about the actual cause.
    """
    if source is FeatureSource.PANIN:
        try:
            panin = describe_panin(slide, annotation, config)
        except Exception as exc:                    # noqa: BLE001 - reported
            log.debug("panin descriptor failed for %s: %s", annotation.id, exc)
            panin = None
        if panin is None:
            return None, ("no measurable lumen or nuclei at "
                          f"{config.target_mpp:g} µm/px")
        return np.asarray(panin, dtype=np.float32), None

    shape = None
    if source in (FeatureSource.SHAPE, FeatureSource.COMBINED):
        result = describe_shape(annotation, mpp=mpp)
        if result is None:
            return None, "outline is degenerate (fewer than 3 distinct points, or zero area)"
        shape = result.features

    texture = None
    if source in (FeatureSource.TEXTURE, FeatureSource.COMBINED):
        try:
            texture = describe(slide, annotation, config)
        except Exception as exc:                        # noqa: BLE001 - reported, not raised
            log.debug("texture descriptor failed for %s: %s", annotation.id, exc)
            texture = None
        if texture is None:
            return None, (f"too few nuclei at {config.target_mpp:g} µm/px "
                          f"(needs {config.min_nuclei})")

    if source is FeatureSource.SHAPE:
        return np.asarray(shape, dtype=np.float32), None
    if source is FeatureSource.TEXTURE:
        return np.asarray(texture, dtype=np.float32), None
    if source is FeatureSource.COMBINED:
        return np.concatenate([shape, texture]).astype(np.float32), None
    raise GeometryError(f"{source} has no grading path. This is a bug: every "
                        f"feature source must be handled here.")


def _label_of(annotation: Annotation) -> str:
    return annotation.display_name or f"{annotation.classification} {annotation.id.hex[:6]}"
