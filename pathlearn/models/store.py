"""Annotation store — ported from ``Models/AnnotationStore.swift``.

Owns the in-memory annotation list for the open slide and keeps a ``.geojson``
sidecar next to the slide in sync, written atomically on every mutation.

The one behavioural change from the Swift is coordinate handling.  The macOS
store wrote mirrored view-Y coordinates into the sidecar; this store writes the
single upright space (see :mod:`pathlearn.coords`), which makes the sidecar
directly QuPath-compatible.  Legacy mirrored sidecars are detected on load and
migrated once, after backing up the original.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..coords import (MARKER_KEY, CoordinateSpace, Provenance,
                      infer_space)
from ..io import geojson
from .annotation import Annotation, AnnotationColor, mirror_all_y

#: Suffix appended to a legacy sidecar before it is rewritten upright.
LEGACY_BACKUP_SUFFIX = ".macos-mirrored.bak"


class SidecarError(Exception):
    """Raised when a sidecar cannot be read or safely migrated."""


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What happened when a sidecar was loaded."""

    path: Path | None
    count: int
    space: CoordinateSpace
    migrated: bool
    backup_path: Path | None = None

    @property
    def message(self) -> str:
        if self.path is None:
            return "No slide bound."
        if self.count == 0:
            return "No existing annotations."
        if self.migrated:
            where = f" Original backed up to {self.backup_path.name}." if self.backup_path else ""
            return (f"Loaded {self.count} annotation(s) and migrated them from the legacy "
                    f"mirrored macOS layout.{where}")
        return f"Loaded {self.count} annotation(s)."


class AnnotationStore:
    """The annotation list for one slide, persisted to a GeoJSON sidecar.

    ``on_change`` is invoked after every mutation so the UI can repaint without
    the store knowing anything about Qt.
    """

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self.annotations: list[Annotation] = []
        self.selected_id: uuid.UUID | None = None
        self.current_label: str = "PaNIN-1"
        self.current_color: AnnotationColor = AnnotationColor.default()
        self.sidecar_path: Path | None = None
        self._slide_height: float = 0.0
        self._on_change = on_change

    # -- binding ----------------------------------------------------------

    def bind(self, slide_path: str | Path, slide_height: float,
             sidecar_suffix: str = "geojson") -> LoadReport:
        """Bind to *slide_path* and load its sidecar.

        *slide_height* is required to migrate legacy mirrored files, so the
        caller must have the slide open already.  ``sidecar_suffix`` lets a
        second store (e.g. predicted regions) persist to a companion file
        without colliding with the manual ``.geojson``.
        """
        slide_path = Path(slide_path)
        self.selected_id = None
        self._slide_height = float(slide_height)
        self.sidecar_path = slide_path.with_suffix("." + sidecar_suffix.lstrip("."))
        return self.load_if_present()

    def unbind(self) -> None:
        """Flush a final save, then clear state so the next slide starts clean."""
        self.save()
        self.annotations = []
        self.selected_id = None
        self.sidecar_path = None
        self._slide_height = 0.0
        self._notify()

    # -- persistence ------------------------------------------------------

    def load_if_present(self) -> LoadReport:
        path = self.sidecar_path
        if path is None or not path.exists():
            self.annotations = []
            self._notify()
            return LoadReport(path, 0, CoordinateSpace.TOP_LEFT, migrated=False)

        try:
            document = geojson.decode_document(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.annotations = []
            self._notify()
            raise SidecarError(f"Could not read {path.name}: {exc}") from exc

        space = infer_space(document, Provenance.SIDECAR)
        annotations = geojson.decode_features(document)

        migrated = False
        backup: Path | None = None
        if space is CoordinateSpace.LEGACY_MIRRORED and annotations:
            # A macOS-era sidecar: stored in mirrored view-Y.  Flip once, back
            # up the original, then rewrite upright and marked so this is a
            # one-time cost.  See pathlearn.coords for why this is necessary.
            annotations = mirror_all_y(annotations, self._slide_height)
            backup = _unique_backup_path(path)
            try:
                shutil.copy2(path, backup)
            except OSError as exc:
                raise SidecarError(
                    f"Refusing to migrate {path.name}: could not write a backup ({exc})."
                ) from exc
            migrated = True

        self.annotations = annotations
        if migrated:
            self.save()
        self._notify()
        return LoadReport(path, len(annotations), space, migrated, backup)

    def save(self) -> None:
        """Write the sidecar atomically.  Silent no-op when unbound."""
        path = self.sidecar_path
        if path is None:
            return
        _atomic_write(path, geojson.encode(self.annotations))

    # -- mutations --------------------------------------------------------

    def add(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)
        self.selected_id = annotation.id
        self.save()
        self._notify()

    def add_batch(self, incoming: Iterable[Annotation]) -> None:
        """Append many at once, saving the sidecar a single time."""
        items = list(incoming)
        if not items:
            return
        self.annotations.extend(items)
        self.save()
        self._notify()

    def update(self, annotation: Annotation) -> None:
        for i, existing in enumerate(self.annotations):
            if existing.id == annotation.id:
                self.annotations[i] = annotation
                self.save()
                self._notify()
                return

    def remove(self, annotation_id: uuid.UUID) -> None:
        before = len(self.annotations)
        self.annotations = [a for a in self.annotations if a.id != annotation_id]
        if len(self.annotations) != before:
            if self.selected_id == annotation_id:
                self.selected_id = None
            self.save()
            self._notify()

    def remove_many(self, ids: Iterable[uuid.UUID]) -> int:
        """Delete a batch, saving the sidecar once rather than per annotation."""
        doomed = set(ids)
        if not doomed:
            return 0
        before = len(self.annotations)
        self.annotations = [a for a in self.annotations if a.id not in doomed]
        removed = before - len(self.annotations)
        if removed:
            if self.selected_id in doomed:
                self.selected_id = None
            self.save()
            self._notify()
        return removed

    def remove_all(self) -> int:
        removed = len(self.annotations)
        self.annotations = []
        self.selected_id = None
        self.save()
        self._notify()
        return removed

    def rename(self, annotation_id: uuid.UUID, name: str | None) -> None:
        ann = self.by_id(annotation_id)
        if ann is None:
            return
        trimmed = (name or "").strip()
        ann.name = trimmed or None
        self.save()
        self._notify()

    def set_classification(self, annotation_id: uuid.UUID, name: str,
                           color: AnnotationColor) -> None:
        ann = self.by_id(annotation_id)
        if ann is None:
            return
        ann.classification = name
        ann.color = color
        self.save()
        self._notify()

    def rename_class(self, old: str, new: str) -> int:
        """Point every annotation on this slide at a renamed class.

        Returns how many were changed.  The drawing class follows the rename
        too, so the next region you trace does not resurrect the old name.
        """
        changed = 0
        for annotation in self.annotations:
            if annotation.classification == old:
                annotation.classification = new
                changed += 1
        if self.current_label == old:
            self.current_label = new
        if changed:
            self.save()
            self._notify()
        return changed

    def set_subtractive(self, annotation_id: uuid.UUID, subtractive: bool) -> None:
        ann = self.by_id(annotation_id)
        if ann is None:
            return
        ann.is_subtractive = subtractive
        self.save()
        self._notify()

    def import_merge(self, incoming: Sequence[Annotation], replace: bool) -> None:
        """Merge annotations decoded from another GeoJSON source.

        In *replace* mode the current set is discarded; otherwise the incoming
        polygons get fresh UUIDs so they cannot collide with existing rows.
        """
        if replace:
            self.annotations = list(incoming)
            self.selected_id = None
        else:
            self.annotations.extend(a.copy_with_new_id() for a in incoming)
        self.save()
        self._notify()

    # -- queries ----------------------------------------------------------

    def by_id(self, annotation_id: uuid.UUID) -> Annotation | None:
        for a in self.annotations:
            if a.id == annotation_id:
                return a
        return None

    @property
    def selected(self) -> Annotation | None:
        return self.by_id(self.selected_id) if self.selected_id else None

    @property
    def existing_labels(self) -> list[str]:
        return sorted({a.classification for a in self.annotations})

    def in_use(self, include_subtractive: bool = False) -> list[Annotation]:
        """Annotations the user has ticked, i.e. those pipelines act on.

        Keyed on ``is_selected``, deliberately NOT on ``is_visible``. Hiding an
        annotation to look at the tissue under it must not remove it from
        analysis — and macOS-era sidecars are full of isVisible=false for
        exactly that reason.
        """
        return [a for a in self.annotations
                if a.is_selected and (include_subtractive or not a.is_subtractive)]

    @property
    def subtractive_in_use(self) -> list[Annotation]:
        return [a for a in self.annotations if a.is_selected and a.is_subtractive]

    @property
    def selected_count(self) -> int:
        return sum(1 for a in self.annotations if a.is_selected)

    def set_selected(self, annotation_id: uuid.UUID, selected: bool) -> None:
        ann = self.by_id(annotation_id)
        if ann is None:
            return
        ann.is_selected = selected
        self.save()
        self._notify()

    def select_all(self, selected: bool = True) -> int:
        changed = sum(1 for a in self.annotations if a.is_selected != selected)
        for a in self.annotations:
            a.is_selected = selected
        if changed:
            self.save()
            self._notify()
        return changed

    def hit_test(self, x: float, y: float) -> Annotation | None:
        """Topmost annotation containing the point, or ``None``.

        Iterates in reverse so the most recently drawn polygon wins, matching
        what the user sees on the canvas — and since every annotation is
        now drawn, every annotation is clickable.
        """
        for ann in reversed(self.annotations):
            if ann.contains(x, y):
                return ann
        return None

    # -- internals --------------------------------------------------------

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()


def count_class_in_sidecar(path: Path, label: str) -> int:
    """How many annotations in *path* carry *label*.  0 if it is unreadable."""
    try:
        document = geojson.decode_document(Path(path).read_text(encoding="utf-8"))
        return sum(1 for a in geojson.decode_features(document)
                   if a.classification == label)
    except (OSError, ValueError):
        return 0


def rename_class_in_sidecar(path: Path, old: str, new: str) -> int:
    """Rename a class inside a sidecar for a slide that is not open.

    **The mirroring marker is preserved rather than stamped.**  A sidecar
    written on macOS carries no marker, and that absence is what tells
    PathLearn to flip it the first time the slide is opened.  Re-encoding with
    the default ``stamp_marker=True`` would claim we wrote it, the flip would
    never happen, and every annotation on that slide would stay mirrored for
    good.  So whatever the file said about itself, it still says afterwards.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    document = geojson.decode_document(text)
    annotations = geojson.decode_features(document)
    changed = 0
    for annotation in annotations:
        if annotation.classification == old:
            annotation.classification = new
            changed += 1
    if changed:
        _atomic_write(source, geojson.encode(
            annotations, stamp_marker=MARKER_KEY in document))
    return changed


def _unique_backup_path(path: Path) -> Path:
    """A non-colliding ``<name>.macos-mirrored.bak`` next to *path*."""
    candidate = path.with_name(path.name + LEGACY_BACKUP_SUFFIX)
    n = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}{LEGACY_BACKUP_SUFFIX}.{n}")
        n += 1
    return candidate


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then replace.

    Same directory matters: ``os.replace`` is only atomic within a volume, and
    slides commonly live on a different drive from the system temp folder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
