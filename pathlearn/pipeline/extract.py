"""Patch extraction — from ``Models/PatchExtractionPipeline.swift``.

Per annotation: sample a grid, read each patch, apply the whiteness and
nucleus filters, embed the survivors, and write them to the bank.

Order matters and is not arbitrary.  The cheap rejections run first — whiteness
is a mean over pixels, nucleus counting is a stain deconvolution plus connected
components, and embedding is a ViT forward pass.  Filtering before embedding is
what makes a large ROI affordable, because most of a slide is background.

Patches are embedded in **batches** (default 8): the GPU is idle between small
calls, and batching is the difference between ~29 ms and ~450 ms per patch on
UNI2-h.  Larger batches did not measurably help and cost memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from ..core.nucleus import ClassicalNucleusSegmenter, NucleusSegmenter
from ..core.sampler import cancel_origins, sample_patches
from ..core.stain import deconvolve
from ..data.bank import ELEMENT_FLOAT32, Patch, PatchBank
from ..extractors.onnx_extractor import OnnxFeatureExtractor
from ..io.slide import SlideImage
from ..models.annotation import Annotation

log = logging.getLogger(__name__)

#: Pixels with R, G and B all at or above this count as white background.
WHITE_LEVEL = 220

#: Rows buffered before a bank write, matching the Swift's batched inserts.
DB_FLUSH_EVERY = 50


@dataclass(frozen=True, slots=True)
class ExtractionSettings:
    """Sampling geometry and filters.

    ``patch_size_level`` and ``stride_level`` are in the sampled level's pixels;
    the sampler scales them to level-0 by the level's downsample.
    """

    patch_size_level: int = 224
    stride_level: int = 224
    level: int = 0
    max_white_fraction: float = 0.75
    min_nuclei: int = 0
    #: Resize patches to the extractor's input size when the sampled size
    #: differs.  Off by default: a silent resize changes the feature space.
    allow_resize: bool = True
    batch_size: int = 8

    def validate(self) -> None:
        if self.patch_size_level <= 0:
            raise ValueError("patch size must be positive")
        if self.stride_level <= 0:
            raise ValueError("stride must be positive")
        if not 0.0 <= self.max_white_fraction <= 1.0:
            raise ValueError("max white fraction must be in [0, 1]")


@dataclass
class ExtractionReport:
    """What happened, per class and in total."""

    saved: int = 0
    skipped_white: int = 0
    skipped_nuclei: int = 0
    skipped_bounds: int = 0
    failed_reads: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def considered(self) -> int:
        return (self.saved + self.skipped_white + self.skipped_nuclei
                + self.skipped_bounds + self.failed_reads)

    def summary(self) -> str:
        if self.considered == 0:
            return "No patches were sampled — check the annotation and patch size."
        parts = [f"Saved {self.saved} of {self.considered} sampled"]
        if self.skipped_white:
            parts.append(f"{self.skipped_white} too white")
        if self.skipped_nuclei:
            parts.append(f"{self.skipped_nuclei} too few nuclei")
        if self.skipped_bounds:
            parts.append(f"{self.skipped_bounds} out of bounds")
        if self.failed_reads:
            parts.append(f"{self.failed_reads} unreadable")
        text = ", ".join(parts) + "."
        if self.by_class:
            text += "  " + ", ".join(f"{k}: {v}" for k, v in sorted(self.by_class.items()))
        if self.cancelled:
            text += "  (cancelled early)"
        return text


ProgressFn = Callable[[int, int, str], None]
CancelFn = Callable[[], bool]


def white_fraction(rgb: np.ndarray) -> float:
    """Share of pixels with R, G and B all >= :data:`WHITE_LEVEL`."""
    if rgb.size == 0:
        return 1.0
    return float(np.all(rgb >= WHITE_LEVEL, axis=-1).mean())


def count_nuclei(rgb: np.ndarray, segmenter: NucleusSegmenter) -> int:
    """Nucleus instances in a patch, via stain deconvolution."""
    stain = deconvolve(rgb)
    tissue = ~stain.is_background
    if not tissue.any():
        return 0
    return len(segmenter.segment(stain, tissue).nuclei)


def extract_annotations(
    slide: SlideImage,
    annotations: Sequence[Annotation],
    extractor: OnnxFeatureExtractor,
    bank: PatchBank,
    settings: ExtractionSettings,
    *,
    subtractive: Sequence[Annotation] | None = None,
    segmenter: NucleusSegmenter | None = None,
    progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> ExtractionReport:
    """Extract every annotation into *bank*.  Returns a per-class report.

    *subtractive* polygons cancel patches whose centre falls inside them; they
    are never themselves extracted.
    """
    settings.validate()
    report = ExtractionReport()
    if not annotations:
        return report

    segmenter = segmenter or ClassicalNucleusSegmenter()
    downsample = slide.level_downsamples[settings.level]
    subtractive_polys = [a.points for a in (subtractive or []) if a.is_subtractive]

    # Plan all origins first so progress is a real fraction, not a guess.
    plan: list[tuple[Annotation, list[tuple[int, int]]]] = []
    for annotation in annotations:
        origins = sample_patches(
            annotation.points, slide.dimensions.width, slide.dimensions.height,
            settings.patch_size_level, settings.stride_level, downsample)
        if subtractive_polys:
            size0 = int(round(settings.patch_size_level * downsample))
            origins = cancel_origins(origins, size0, subtractive_polys)
        plan.append((annotation, origins))

    total = sum(len(o) for _, o in plan)
    if total == 0:
        return report

    pending: list[Patch] = []
    done = 0

    for annotation, origins in plan:
        if should_cancel and should_cancel():
            report.cancelled = True
            break

        for start in range(0, len(origins), settings.batch_size):
            if should_cancel and should_cancel():
                report.cancelled = True
                break

            chunk = origins[start:start + settings.batch_size]
            kept: list[tuple[tuple[int, int], np.ndarray, float, int]] = []

            for origin in chunk:
                outcome = _prepare_patch(slide, origin, settings, extractor,
                                         segmenter, report)
                if outcome is not None:
                    kept.append((origin, *outcome))

            done += len(chunk)
            if progress:
                progress(done, total, f"{annotation.classification}: "
                                      f"{report.saved} saved / {done} of {total}")

            if not kept:
                continue

            batch = np.stack([item[1] for item in kept])
            features = extractor.extract(batch)

            for (origin, _, white, nuclei), vector in zip(kept, features):
                pending.append(Patch(
                    slide_path=str(slide.path),
                    slide_name=slide.name,
                    annotation_id=annotation.id,
                    classification=annotation.classification,
                    patch_x=origin[0],
                    patch_y=origin[1],
                    patch_level=settings.level,
                    patch_size_level=settings.patch_size_level,
                    features=vector,
                    extractor_identity=str(extractor.identity),
                    white_fraction=white,
                    nucleus_count=nuclei,
                    element_type=ELEMENT_FLOAT32,
                ))
                report.saved += 1
                report.by_class[annotation.classification] = \
                    report.by_class.get(annotation.classification, 0) + 1

            if len(pending) >= DB_FLUSH_EVERY:
                bank.add_many(pending)
                pending.clear()

        if report.cancelled:
            break

    if pending:
        bank.add_many(pending)
    return report


def _prepare_patch(slide: SlideImage, origin: tuple[int, int],
                   settings: ExtractionSettings, extractor: OnnxFeatureExtractor,
                   segmenter: NucleusSegmenter,
                   report: ExtractionReport) -> tuple[np.ndarray, float, int] | None:
    """Read, filter and size one patch.  ``None`` means it was rejected.

    Filters run cheapest-first: whiteness is a mean, nucleus counting is a
    deconvolution plus connected components, and embedding (which happens later,
    batched) is a ViT forward pass.
    """
    x, y = origin
    try:
        rgb = slide.read_region(x, y, settings.level,
                                settings.patch_size_level, settings.patch_size_level)
    except Exception as exc:
        log.debug("read failed at (%d, %d): %s", x, y, exc)
        report.failed_reads += 1
        return None

    if rgb.size == 0:
        report.skipped_bounds += 1
        return None

    white = white_fraction(rgb)
    if white > settings.max_white_fraction:
        report.skipped_white += 1
        return None

    nuclei = -1
    if settings.min_nuclei > 0:
        nuclei = count_nuclei(rgb, segmenter)
        if nuclei < settings.min_nuclei:
            report.skipped_nuclei += 1
            return None

    target_w, target_h = extractor.input_size
    if rgb.shape[:2] != (target_h, target_w):
        if not settings.allow_resize:
            report.skipped_bounds += 1
            return None
        rgb = _resize(rgb, target_w, target_h)

    return rgb, white, nuclei


def _resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize to the extractor's input size.

    Only reached when the sampled patch size differs from the model's input.
    Sampling at the model's native size avoids it entirely, which is why the
    extraction sheet defaults patch size to the extractor's input.
    """
    from PIL import Image
    return np.asarray(
        Image.fromarray(rgb).resize((width, height), Image.BILINEAR),
        dtype=np.uint8)


def estimate_patch_count(slide: SlideImage, annotations: Iterable[Annotation],
                         settings: ExtractionSettings) -> int:
    """Exact in-polygon count, for the sheet's live preview.

    Runs the real sampler rather than the bbox estimate, because the difference
    between "12,000 patches" and "2,000 patches" changes what the user chooses.
    """
    downsample = slide.level_downsamples[settings.level]
    return sum(
        len(sample_patches(a.points, slide.dimensions.width, slide.dimensions.height,
                           settings.patch_size_level, settings.stride_level, downsample))
        for a in annotations if not a.is_subtractive
    )
