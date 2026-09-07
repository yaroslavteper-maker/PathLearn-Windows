"""Prediction — from ``Models/PredictionPipeline.swift``.

Samples tiles across a region, embeds them, classifies, and returns one
:class:`PatchPrediction` per surviving tile.

AGGREGATION MODES
=================
* ``PER_PATCH`` — every tile keeps its own verdict.  This is what the heatmap
  and the region proposer want.
* ``MEAN_PROBABILITY`` — average the probability vectors over the whole region
  and give every tile that single verdict.
* ``MAX_PROBABILITY`` — take the region's most confident tile and apply its
  verdict everywhere ("worst focus", how a pathologist grades a duct).

A **pooled** classifier is different again: it expects one ``mean‖max‖std``
vector per region, so the tiles are pooled once, classified once, and every tile
is painted with that verdict.  The model says which it needs, not the caller.

GATES
=====
Whiteness, optional nucleus count, and — when the model carries one — the null
filter, applied to raw embeddings before classification exactly as at training
time (``04-DESIGN-DECISIONS.md`` §4).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..core.nucleus import ClassicalNucleusSegmenter, NucleusSegmenter
from ..core.pooling import mean_max_std
from ..core.sampler import cancel_origins, sample_patches
from ..io.slide import SlideImage
from ..models.annotation import Annotation
from ..models.classifier import MLClassifier
from ..models.prediction import PatchPrediction, PredictionSet
from .extract import ExtractionSettings, count_nuclei, white_fraction, _resize

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]
CancelFn = Callable[[], bool]


class Aggregation(enum.Enum):
    PER_PATCH = "perPatch"
    MEAN_PROBABILITY = "meanProbability"
    MAX_PROBABILITY = "maxProbability"


class PredictionError(ValueError):
    """Raised when a prediction run cannot proceed."""


@dataclass(frozen=True, slots=True)
class PredictionSettings:
    patch_size_level: int = 224
    stride_level: int = 224
    level: int = 0
    max_white_fraction: float = 0.75
    min_nuclei: int = 0
    min_confidence: float = 0.0
    aggregation: Aggregation = Aggregation.PER_PATCH
    batch_size: int = 8

    def validate(self) -> None:
        if self.patch_size_level <= 0 or self.stride_level <= 0:
            raise PredictionError("Patch size and stride must be positive.")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise PredictionError("Minimum confidence must be in [0, 1].")


@dataclass
class PredictionReport:
    considered: int = 0
    predicted: int = 0
    skipped_white: int = 0
    skipped_nuclei: int = 0
    skipped_null: int = 0
    below_confidence: int = 0
    cancelled: bool = False

    def summary(self) -> str:
        if self.considered == 0:
            return "No tiles were sampled — check the region and patch size."
        parts = [f"{self.predicted} of {self.considered} tiles predicted"]
        for count, why in ((self.skipped_white, "too white"),
                           (self.skipped_nuclei, "too few nuclei"),
                           (self.skipped_null, "null-like"),
                           (self.below_confidence, "below confidence")):
            if count:
                parts.append(f"{count} {why}")
        text = ", ".join(parts) + "."
        if self.cancelled:
            text += "  (cancelled early)"
        return text


def predict_regions(
    slide: SlideImage,
    regions: Sequence[Annotation],
    extractor,
    classifier: MLClassifier,
    settings: PredictionSettings,
    *,
    subtractive: Sequence[Annotation] | None = None,
    segmenter: NucleusSegmenter | None = None,
    colors: dict | None = None,
    accept_identities: set[str] | None = None,
    progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> tuple[PredictionSet, PredictionReport]:
    """Classify tiles across *regions*.

    The classifier's own extractor identity is checked against *extractor*:
    applying a model to a different feature space produces confident nonsense,
    which is precisely the failure the identity stamp exists to prevent.

    *accept_identities* widens that check to a caller-supplied set — used only
    for equivalences the user has explicitly approved (see
    :mod:`pathlearn.extractors.equivalence`).  Passing it is a deliberate act;
    the default remains strict.
    """
    settings.validate()
    report = PredictionReport()
    result = PredictionSet(
        class_labels=list(classifier.class_labels),
        colors=dict(colors or {}),
        extractor_identity=str(extractor.identity),
        slide_path=str(slide.path),
        min_confidence=settings.min_confidence,
    )

    identity = str(extractor.identity)
    permitted = set(accept_identities or ()) | {classifier.extractor_identity}
    if not classifier.accepts(identity) and identity not in permitted:
        raise PredictionError(
            f"This model was trained on {classifier.extractor_identity} but the "
            f"selected extractor is {identity}. Their feature spaces "
            f"are not comparable.")

    regions = [r for r in regions if not r.is_subtractive]
    if not regions:
        return result, report

    segmenter = segmenter or ClassicalNucleusSegmenter()
    downsample = slide.level_downsamples[settings.level]
    size_level0 = int(round(settings.patch_size_level * downsample))
    subtractive_polys = [a.points for a in (subtractive or []) if a.is_subtractive]

    plan: list[tuple[Annotation, list[tuple[int, int]]]] = []
    for region in regions:
        origins = sample_patches(region.points, slide.dimensions.width,
                                 slide.dimensions.height, settings.patch_size_level,
                                 settings.stride_level, downsample)
        if subtractive_polys:
            origins = cancel_origins(origins, size_level0, subtractive_polys)
        plan.append((region, origins))

    total = sum(len(o) for _, o in plan)
    if total == 0:
        return result, report

    done = 0
    for pass_index, (region, origins) in enumerate(plan):
        if should_cancel and should_cancel():
            report.cancelled = True
            break

        kept_origins: list[tuple[int, int]] = []
        kept_features: list[np.ndarray] = []

        for start in range(0, len(origins), settings.batch_size):
            if should_cancel and should_cancel():
                report.cancelled = True
                break

            chunk = origins[start:start + settings.batch_size]
            images, usable = [], []
            for origin in chunk:
                image = _read_and_gate(slide, origin, settings, extractor,
                                       segmenter, report)
                if image is not None:
                    images.append(image)
                    usable.append(origin)

            done += len(chunk)
            report.considered += len(chunk)
            if progress:
                progress(done, total, f"{region.classification or 'region'}: "
                                      f"{report.predicted} predicted / {done} of {total}")
            if not images:
                continue

            features = extractor.extract(np.stack(images))

            # Null gating uses the raw embedding, as at training time.
            if classifier.uses_null_filter:
                null_like = classifier.is_null_like(features)
                if null_like.any():
                    report.skipped_null += int(null_like.sum())
                    keep = ~null_like
                    features = features[keep]
                    usable = [o for o, k in zip(usable, keep) if k]
                if not len(usable):
                    continue

            kept_origins.extend(usable)
            kept_features.append(features)

        if report.cancelled:
            break
        if not kept_origins:
            continue

        matrix = np.concatenate(kept_features, axis=0)
        result.predictions.extend(
            _classify(matrix, kept_origins, classifier, settings, size_level0,
                      pass_index, report))

    return result, report


def _classify(features: np.ndarray, origins: list[tuple[int, int]],
              classifier: MLClassifier, settings: PredictionSettings,
              size_level0: int, pass_index: int,
              report: PredictionReport) -> list[PatchPrediction]:
    """Turn one region's features into predictions, honouring the mode."""
    if classifier.is_pooled:
        # The model wants one vector per region, so pool and classify once; the
        # verdict then applies to every tile in the region.
        pooled = mean_max_std(features)
        if pooled is None:
            return []
        label, probs = classifier.predict(pooled)
        return _fill(origins, label, probs, settings, size_level0, pass_index, report)

    labels, probabilities = classifier.predict_batch(features)

    if settings.aggregation is Aggregation.MEAN_PROBABILITY:
        mean = probabilities.mean(axis=0)
        label = classifier.class_labels[int(np.argmax(mean))]
        return _fill(origins, label, mean, settings, size_level0, pass_index, report)

    if settings.aggregation is Aggregation.MAX_PROBABILITY:
        best = int(np.argmax(probabilities.max(axis=1)))
        probs = probabilities[best]
        label = classifier.class_labels[int(np.argmax(probs))]
        return _fill(origins, label, probs, settings, size_level0, pass_index, report)

    out: list[PatchPrediction] = []
    for origin, label, probs in zip(origins, labels, probabilities):
        if float(probs.max()) < settings.min_confidence:
            report.below_confidence += 1
            continue
        out.append(PatchPrediction(
            x=origin[0], y=origin[1], size_level0=size_level0, label=label,
            probabilities=probs.astype(np.float32), pass_index=pass_index,
            patch_level=settings.level, patch_size_level=settings.patch_size_level))
        report.predicted += 1
    return out


def _fill(origins: list[tuple[int, int]], label: str, probs: np.ndarray,
          settings: PredictionSettings, size_level0: int, pass_index: int,
          report: PredictionReport) -> list[PatchPrediction]:
    """Give every tile in a region the same verdict."""
    if float(np.max(probs)) < settings.min_confidence:
        report.below_confidence += len(origins)
        return []
    report.predicted += len(origins)
    return [PatchPrediction(x=x, y=y, size_level0=size_level0, label=label,
                            probabilities=np.asarray(probs, dtype=np.float32),
                            pass_index=pass_index, patch_level=settings.level,
                            patch_size_level=settings.patch_size_level)
            for x, y in origins]


def _read_and_gate(slide: SlideImage, origin: tuple[int, int],
                   settings: PredictionSettings, extractor,
                   segmenter: NucleusSegmenter,
                   report: PredictionReport) -> np.ndarray | None:
    """Read one tile and apply the cheap gates.  ``None`` means rejected."""
    try:
        rgb = slide.read_region(origin[0], origin[1], settings.level,
                                settings.patch_size_level, settings.patch_size_level)
    except Exception:
        return None
    if rgb.size == 0:
        return None

    if white_fraction(rgb) > settings.max_white_fraction:
        report.skipped_white += 1
        return None

    if settings.min_nuclei > 0 and count_nuclei(rgb, segmenter) < settings.min_nuclei:
        report.skipped_nuclei += 1
        return None

    target_w, target_h = extractor.input_size
    if rgb.shape[:2] != (target_h, target_w):
        rgb = _resize(rgb, target_w, target_h)
    return rgb
