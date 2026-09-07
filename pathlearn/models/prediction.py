"""Prediction records and their store — ``Models/PatchPrediction.swift``.

A prediction is one classified tile: where it sits on the slide, what the model
called it, and how confident it was.  Coordinates use the same space as
everything else (level-0, top-left origin, Y down), which is why the heatmap can
draw them with the canvas transform and no correction.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .annotation import AnnotationColor


#: Decimal places kept for a stored probability.  float32 repr writes 17
#: significant digits, which is three megabytes of noise on a large slide:
#: the confidence slider moves in hundredths and nothing downstream reads
#: below 1e-4, so anything past four places is storage spent on nothing.
#: The winning class is stored as a label, not recomputed, so rounding can
#: never flip a call.
PROBABILITY_PLACES = 4


@dataclass(slots=True)
class PatchPrediction:
    """One classified tile."""

    #: Level-0 origin, same space as ``Patch``.
    x: int
    y: int
    #: Edge length in level-0 pixels, so the heatmap needs no level lookup.
    size_level0: int
    label: str
    probabilities: np.ndarray
    #: Which multi-pass geometry produced this tile.
    pass_index: int = 0
    patch_level: int = 0
    patch_size_level: int = 0

    @property
    def confidence(self) -> float:
        return float(np.max(self.probabilities)) if self.probabilities.size else 0.0

    @property
    def centre(self) -> tuple[float, float]:
        half = self.size_level0 / 2.0
        return (self.x + half, self.y + half)

    def to_dict(self, class_labels: Sequence[str]) -> dict:
        return {
            "x": self.x, "y": self.y, "sizeLevel0": self.size_level0,
            "label": self.label,
            "probabilities": [round(float(v), PROBABILITY_PLACES)
                              for v in self.probabilities],
            "maxProbability": round(self.confidence, PROBABILITY_PLACES),
            "passIndex": self.pass_index,
            "patchLevel": self.patch_level,
            "patchSizeLevel": self.patch_size_level,
            "classLabels": list(class_labels),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatchPrediction":
        return cls(
            x=int(data["x"]), y=int(data["y"]),
            size_level0=int(data.get("sizeLevel0", 0)),
            label=str(data.get("label", "?")),
            probabilities=np.asarray(data.get("probabilities") or [], dtype=np.float32),
            pass_index=int(data.get("passIndex", 0)),
            patch_level=int(data.get("patchLevel", 0)),
            patch_size_level=int(data.get("patchSizeLevel", 0)),
        )


@dataclass
class PredictionSet:
    """Everything one prediction run produced, plus how to draw it."""

    predictions: list[PatchPrediction] = field(default_factory=list)
    class_labels: list[str] = field(default_factory=list)
    colors: dict[str, AnnotationColor] = field(default_factory=dict)
    extractor_identity: str = ""
    slide_path: str = ""
    #: Classes the user has hidden in the overlay.
    hidden: set[str] = field(default_factory=set)
    #: Tiles below this confidence are not drawn.
    min_confidence: float = 0.0

    def __len__(self) -> int:
        return len(self.predictions)

    @property
    def is_empty(self) -> bool:
        return not self.predictions

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(p.label for p in self.predictions))

    def visible(self) -> list[PatchPrediction]:
        """Predictions passing the current visibility and confidence filters."""
        return [p for p in self.predictions
                if p.label not in self.hidden and p.confidence >= self.min_confidence]

    def color_for(self, label: str) -> AnnotationColor:
        return self.colors.get(label, AnnotationColor.default())

    def summary(self) -> str:
        if self.is_empty:
            return "No predictions."
        counts = self.counts
        total = len(self.predictions)
        shown = len(self.visible())
        parts = ", ".join(f"{name}: {n}" for name, n in sorted(counts.items()))
        text = f"{total} tiles — {parts}"
        if shown != total:
            text += f"  ({shown} shown at the current threshold)"
        return text

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "slidePath": self.slide_path,
            # The view state is part of the result, not a UI detail: a run
            # left at 0.60 confidence with one class hidden should come back
            # that way, or the numbers a user quoted will not reproduce.
            "minConfidence": float(self.min_confidence),
            "hidden": sorted(self.hidden),
            "extractorIdentity": self.extractor_identity,
            "classLabels": list(self.class_labels),
            "colors": {k: list(v.as_tuple()) for k, v in self.colors.items()},
            "predictions": [p.to_dict(self.class_labels) for p in self.predictions],
        }

    def save(self, path: str | Path) -> None:
        """Write the set, gzipped when the name says ``.gz``.

        Compression is not cosmetic here: 50,000 tiles is ~15 MB of JSON and
        ~1 MB gzipped, and the auto-saved sidecar is rewritten whenever a run
        finishes.  Writing fifteen megabytes to the slide's drive on every
        prediction is the difference between a sidecar you keep and one you
        turn off.
        """
        target = Path(path)
        payload = json.dumps(self.to_dict())
        # Write beside the target and swap: an interrupted write must not
        # leave a truncated sidecar where a whole one used to be.
        temporary = target.with_suffix(target.suffix + ".partial")
        if target.suffix == ".gz":
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                handle.write(payload)
        else:
            temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "PredictionSet":
        """Read a set, gzipped or not.

        Detection is by the file's own magic number rather than its name, so
        a sidecar still loads after someone renames it — and a plain ``.json``
        written by an older build, or by the macOS app, still loads here.
        """
        source = Path(path)
        raw = source.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        data = json.loads(raw.decode("utf-8"))
        colors = {k: AnnotationColor.from_sequence(v)
                  for k, v in (data.get("colors") or {}).items()}
        return cls(
            predictions=[PatchPrediction.from_dict(d)
                         for d in (data.get("predictions") or [])],
            class_labels=[str(x) for x in (data.get("classLabels") or [])],
            colors=colors,
            extractor_identity=str(data.get("extractorIdentity") or ""),
            slide_path=str(data.get("slidePath") or ""),
            # Absent in version 1, and the defaults are the old behaviour:
            # show everything at no threshold.
            hidden={str(x) for x in (data.get("hidden") or [])},
            min_confidence=float(data.get("minConfidence", 0.0) or 0.0),
        )


def sidecar_path(slide_path: str | Path) -> Path:
    """Where a slide's predictions live — matching the macOS companion file."""
    return Path(slide_path).with_suffix(".predictions.json")


def auto_sidecar_path(slide_path: str | Path) -> Path:
    """Where PathLearn saves the heatmap by itself, beside the slide.

    Separate from ``sidecar_path`` and gzipped.  The plain ``.json`` name is
    the macOS companion format and stays readable for anything that reads it;
    this one is written unattended after every run, so it is compressed.
    """
    return Path(slide_path).with_suffix(".predictions.json.gz")


def find_sidecar(slide_path: str | Path) -> Path | None:
    """The heatmap saved beside *slide_path*, compressed or plain.

    Prefers the auto-saved one, then a hand-saved ``.json`` — so a sidecar
    exported by an older build, or by the macOS app, is still picked up.
    """
    for candidate in (auto_sidecar_path(slide_path), sidecar_path(slide_path)):
        if candidate.exists():
            return candidate
    return None
