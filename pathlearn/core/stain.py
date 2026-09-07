"""H&E colour deconvolution — from ``Models/StainDeconvolution.swift``.

Ruifrok & Johnston (2001): separates an RGB image into per-pixel haematoxylin
(nuclei) and eosin (cytoplasm/stroma) concentrations in optical-density space.
Pure CPU, no models.  This is the classical backend behind the geometry
descriptors; a learned stain normaliser can replace it without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Near-white cutoff: a pixel with R, G and B all >= this is background.
BACKGROUND_THRESHOLD = 220

#: Normalised H&E stain vectors in RGB optical-density space.  The third row is
#: a residual stain that exists only to keep the 3x3 matrix invertible.
STAIN_MATRIX = np.array([
    [0.650, 0.704, 0.286],   # haematoxylin (nuclei, blue)
    [0.072, 0.990, 0.105],   # eosin (cytoplasm, pink)
    [0.268, 0.570, 0.776],   # residual
], dtype=np.float64)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.where(norms > 0, matrix / norms, matrix)


_NORMALISED = _normalise_rows(STAIN_MATRIX)
try:
    _INVERSE = np.linalg.inv(_NORMALISED)
except np.linalg.LinAlgError:  # pragma: no cover - the constant matrix is invertible
    _INVERSE = np.eye(3)


@dataclass(frozen=True, slots=True)
class StainResult:
    """Decoded stain concentrations for a region, one scalar per pixel."""

    #: Haematoxylin concentration, ``(h, w)`` float32.  Higher = more nuclei.
    hematoxylin: np.ndarray
    #: Eosin concentration, ``(h, w)`` float32.  Higher = more cytoplasm.
    eosin: np.ndarray
    #: True where the pixel is near-white background rather than tissue.
    is_background: np.ndarray

    @property
    def height(self) -> int:
        return self.hematoxylin.shape[0]

    @property
    def width(self) -> int:
        return self.hematoxylin.shape[1]


def deconvolve(rgb: np.ndarray,
               background_threshold: int = BACKGROUND_THRESHOLD) -> StainResult:
    """Deconvolve an ``(h, w, 3)`` uint8 RGB image into H and E concentrations.

    Optical density is ``OD = -log10((I + 1) / 256)`` per channel — the ``+1``
    keeps pure black finite — then concentrations are ``inverse_stain @ OD``.
    Negative concentrations are clamped to zero, as in the Swift.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected an (h, w, 3) RGB image, got shape {rgb.shape}")

    channels = rgb[..., :3].astype(np.float64)
    is_background = np.all(channels >= background_threshold, axis=2)

    optical_density = -np.log10((channels + 1.0) / 256.0)
    # Only the first two stain rows are of interest.
    hematoxylin = np.tensordot(optical_density, _INVERSE[0], axes=([2], [0]))
    eosin = np.tensordot(optical_density, _INVERSE[1], axes=([2], [0]))

    return StainResult(
        hematoxylin=np.maximum(hematoxylin, 0.0).astype(np.float32),
        eosin=np.maximum(eosin, 0.0).astype(np.float32),
        is_background=is_background,
    )
