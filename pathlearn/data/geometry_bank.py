"""Geometry descriptor bank — ``geometry_bank.json`` (``02-DATA-FORMATS.md`` §3).

A flat JSON array that accumulates across slides, one record per annotation.
Unlike the patch bank this is small — 14 floats per annotation — so a plain file
is the right shape, matching the macOS store.

VERSIONING IS LOAD-BEARING
==========================
Each record carries the descriptor ``version`` that produced it. Version 1 read
each annotation at "whatever pyramid level fits", so larger lesions were
measured coarser and their absolute descriptors inflated — the classifier
learned "big nuclei ⇒ PaNIN-3" and collapsed (``04-DESIGN-DECISIONS.md`` §2).
Version 2 analyses at a fixed ~1 µm/px.

Mixing the two would silently reintroduce that bug, so stale records are
**rejected by the trainer, not migrated** — the fix is to recompute, which is
cheap, rather than to rescale, which cannot recover the lost resolution.
"""

from __future__ import annotations

import enum
import json
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..core.geometry import DIMENSION, VERSION
from ..core.panin import PANIN_DIMENSION, PANIN_VERSION
from ..core.shape import SHAPE_DIMENSION, SHAPE_VERSION

log = logging.getLogger(__name__)


class FeatureSource(enum.Enum):
    """Which descriptor block(s) a model is trained on."""

    #: Outline of the traced lesion. Computable for every valid polygon and
    #: independent of staining, resolution and nucleus segmentation.
    SHAPE = "shape"
    #: Interior architecture — nuclei and lumina at fixed physical resolution.
    #: Requires enough nuclei, so it is unavailable for small annotations.
    TEXTURE = "texture"
    #: Shape concatenated with texture, in that fixed order.
    COMBINED = "combined"
    #: Polarity and inner-lumen shape — where nuclei sit relative to the
    #: lumen, and the shape of the space inside the duct. Measured on 62
    #: annotated regions as the strongest scale-free signal available
    #: (measured in an internal study); nothing else here looks inside.
    PANIN = "panin"

    @property
    def dimension(self) -> int:
        if self is FeatureSource.SHAPE:
            return SHAPE_DIMENSION
        if self is FeatureSource.TEXTURE:
            return DIMENSION
        if self is FeatureSource.PANIN:
            return PANIN_DIMENSION
        return SHAPE_DIMENSION + DIMENSION

    @property
    def label(self) -> str:
        return {
            FeatureSource.SHAPE: "Outline shape",
            FeatureSource.TEXTURE: "Interior texture",
            FeatureSource.COMBINED: "Shape + texture",
            FeatureSource.PANIN: "PanIN architecture",
        }[self]


GEOMETRY_BANK_FILENAME = "geometry_bank.json"


def default_geometry_bank_path() -> Path:
    directory = Path.home() / "AppData" / "Local" / "PathLearn"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / GEOMETRY_BANK_FILENAME


@dataclass(slots=True)
class GeometryRecord:
    """One annotation's architectural descriptor."""

    slide_path: str
    slide_name: str
    annotation_id: uuid.UUID
    classification: str
    #: Interior texture descriptor (14-D), or None when it could not be computed
    #: — typically too few nuclei in a small annotation.
    features: np.ndarray | None = None
    #: Outline shape descriptor (12-D). Computable for any valid polygon, so
    #: this is present far more often than the texture block.
    shape_features: np.ndarray | None = None
    #: PanIN architecture descriptor (17-D): polarity + inner-lumen shape.
    panin_features: np.ndarray | None = None
    version: int = VERSION
    shape_version: int = SHAPE_VERSION
    panin_version: int = PANIN_VERSION

    @property
    def has_texture(self) -> bool:
        return (self.features is not None and self.version == VERSION
                and self.features.size == DIMENSION)

    @property
    def has_shape(self) -> bool:
        return (self.shape_features is not None
                and self.shape_version == SHAPE_VERSION
                and self.shape_features.size == SHAPE_DIMENSION)

    @property
    def has_panin(self) -> bool:
        return (self.panin_features is not None
                and self.panin_version == PANIN_VERSION
                and self.panin_features.size == PANIN_DIMENSION)

    @property
    def is_current(self) -> bool:
        """Usable for something — any block at the current version."""
        return self.has_texture or self.has_shape or self.has_panin

    def to_dict(self) -> dict:
        # `features` and `version` keep the macOS names and meaning so the file
        # still round-trips with the Swift store; the shape block is additive.
        out = {
            "slidePath": self.slide_path,
            "slideName": self.slide_name,
            "annotationID": str(self.annotation_id).upper(),
            "classification": self.classification,
            "features": ([float(v) for v in self.features]
                         if self.features is not None else None),
            "version": self.version,
        }
        if self.shape_features is not None:
            out["shapeFeatures"] = [float(v) for v in self.shape_features]
            out["shapeVersion"] = self.shape_version
        if self.panin_features is not None:
            out["paninFeatures"] = [float(v) for v in self.panin_features]
            out["paninVersion"] = self.panin_version
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "GeometryRecord | None":
        def block(key: str) -> np.ndarray | None:
            raw = data.get(key)
            if raw is None:
                return None
            try:
                array = np.asarray(raw, dtype=np.float32)
            except (TypeError, ValueError):
                return None
            return array if array.size else None

        features = block("features")
        shape_features = block("shapeFeatures")
        panin_features = block("paninFeatures")
        if features is None and shape_features is None and panin_features is None:
            return None

        try:
            annotation_id = uuid.UUID(str(data.get("annotationID")))
        except (ValueError, TypeError):
            annotation_id = uuid.uuid4()

        return cls(
            slide_path=str(data.get("slidePath", "")),
            slide_name=str(data.get("slideName", "")),
            annotation_id=annotation_id,
            classification=str(data.get("classification", "Unlabeled")),
            features=features,
            shape_features=shape_features,
            panin_features=panin_features,
            version=int(data.get("version", 1)),
            # A record written before a block existed has no version for it;
            # mark it 0 so it is never mistaken for a current block.
            shape_version=int(data.get("shapeVersion", 0)),
            panin_version=int(data.get("paninVersion", 0)),
        )


@dataclass(frozen=True, slots=True)
class GeometryStats:
    total: int
    current: int
    stale: int
    by_class: dict[str, int]
    by_slide: dict[str, int]
    with_shape: int = 0
    with_texture: int = 0
    with_panin: int = 0

    @property
    def is_empty(self) -> bool:
        return self.total == 0

    def summary(self) -> str:
        if self.is_empty:
            return "Geometry bank is empty."
        lines = [f"{self.total} record(s) across {len(self.by_slide)} slide(s)",
                 f"  {self.with_shape} with outline shape, "
                 f"{self.with_texture} with interior texture, "
                 f"{self.with_panin} with PanIN architecture"]
        for name, count in sorted(self.by_class.items()):
            lines.append(f"  {name}: {count}")
        if self.stale:
            lines.append(
                f"  {self.stale} record(s) use an older descriptor version and "
                f"will be ignored — recompute them.")
        return "\n".join(lines)


class GeometryBank:
    """The persisted set of geometry descriptors."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_geometry_bank_path()
        self.records: list[GeometryRecord] = []
        self.load()

    # -- persistence ------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            self.records = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s: %s", self.path.name, exc)
            self.records = []
            return
        entries = data if isinstance(data, list) else (data.get("records") or [])
        self.records = [r for r in (GeometryRecord.from_dict(d) for d in entries
                                    if isinstance(d, dict)) if r is not None]

    def save(self) -> None:
        # A flat array, matching the macOS file exactly.
        self.path.write_text(
            json.dumps([r.to_dict() for r in self.records], indent=2), encoding="utf-8")

    # -- mutation ---------------------------------------------------------

    def add(self, incoming: Iterable[GeometryRecord]) -> int:
        """Add records, replacing any existing entry for the same annotation."""
        items = list(incoming)
        if not items:
            return 0
        replacing = {r.annotation_id for r in items}
        self.records = [r for r in self.records if r.annotation_id not in replacing]
        self.records.extend(items)
        self.save()
        return len(items)

    def rename_class(self, old: str, new: str) -> int:
        """Relabel every record of class *old* as *new*.  Returns the count."""
        if not old or not new or old == new:
            return 0
        changed = 0
        for index, record in enumerate(self.records):
            if record.classification == old:
                self.records[index] = replace(record, classification=new)
                changed += 1
        if changed:
            self.save()
        return changed

    def remove_class(self, label: str) -> int:
        before = len(self.records)
        self.records = [r for r in self.records if r.classification != label]
        removed = before - len(self.records)
        if removed:
            self.save()
        return removed

    def remove_stale(self) -> int:
        before = len(self.records)
        self.records = [r for r in self.records if r.is_current]
        removed = before - len(self.records)
        if removed:
            self.save()
        return removed

    def clear(self) -> int:
        removed = len(self.records)
        self.records = []
        self.save()
        return removed

    # -- queries ----------------------------------------------------------

    @property
    def current_records(self) -> list[GeometryRecord]:
        return [r for r in self.records if r.is_current]

    @property
    def stale_count(self) -> int:
        return sum(1 for r in self.records if not r.is_current)

    def has_annotation(self, annotation_id: uuid.UUID) -> bool:
        return any(r.annotation_id == annotation_id and r.is_current
                   for r in self.records)

    def stats(self) -> GeometryStats:
        current = self.current_records
        return GeometryStats(
            total=len(self.records),
            current=len(current),
            stale=self.stale_count,
            by_class=dict(Counter(r.classification for r in current)),
            by_slide=dict(Counter(r.slide_name for r in current)),
            with_shape=sum(1 for r in self.records if r.has_shape),
            with_texture=sum(1 for r in self.records if r.has_texture),
            with_panin=sum(1 for r in self.records if r.has_panin),
        )

    def usable_for(self, source: "FeatureSource") -> list[GeometryRecord]:
        """Records carrying every block *source* needs."""
        if source is FeatureSource.SHAPE:
            return [r for r in self.records if r.has_shape]
        if source is FeatureSource.TEXTURE:
            return [r for r in self.records if r.has_texture]
        if source is FeatureSource.PANIN:
            return [r for r in self.records if r.has_panin]
        return [r for r in self.records if r.has_shape and r.has_texture]

    def matrix(self, records: Sequence[GeometryRecord] | None = None,
               source: "FeatureSource" = None) -> tuple[np.ndarray, list[str]]:
        """Feature matrix for *source*, defaulting to interior texture.

        Combined concatenates shape then texture, in that order — the order is
        fixed because a trained model's weights are meaningless if the blocks
        are ever swapped.
        """
        source = source or FeatureSource.TEXTURE
        rows = list(records if records is not None else self.usable_for(source))
        width = source.dimension
        if not rows:
            return np.zeros((0, width), dtype=np.float32), []

        if source is FeatureSource.SHAPE:
            stacked = np.stack([r.shape_features for r in rows])
        elif source is FeatureSource.PANIN:
            stacked = np.stack([r.panin_features for r in rows])
        elif source is FeatureSource.TEXTURE:
            stacked = np.stack([r.features for r in rows])
        else:
            stacked = np.stack([np.concatenate([r.shape_features, r.features])
                                for r in rows])
        return stacked.astype(np.float32), [r.classification for r in rows]

    def __len__(self) -> int:
        return len(self.records)
