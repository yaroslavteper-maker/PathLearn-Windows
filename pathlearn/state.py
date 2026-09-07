"""Saving and restoring a whole working state as one file.

A state is everything the app is holding: the class palette, both banks, the
trained models, the annotations for every slide you have touched, and the
settings you were working with. One ``.pathlearn`` file (a zip) replaces the
old profile file, which only ever held the class list.

WHAT IS NOT IN A STATE, AND WHY
===============================
**Slides.** They are gigabytes each and usually live on a separate drive.
A state records their *paths* so it can put annotations back beside them, and
says plainly when a slide is missing rather than pretending it restored.

**Extractors.** They are installed models, not working state — 4 GB of them,
shared by every project on the machine, and separately licensed. Use
``tools/transfer.py`` to move those.

ANNOTATIONS ARE COPIES, NOT MOVES
=================================
Annotations live in a ``.geojson`` beside each slide and are the authority.
A state carries a copy of each. Restoring writes them back, which **overwrites
the sidecar** — so it is confirmed, and the previous file is backed up first.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

STATE_SUFFIX = ".pathlearn"
STATE_VERSION = 1

MANIFEST = "manifest.json"
PROFILE = "profile.json"
PATCH_BANK = "bank.db"
GEOMETRY_BANK = "geometry_bank.json"
EQUIVALENCES = "extractor-equivalences.json"
ANNOTATION_DIR = "annotations"
MODEL_DIR = "models"

#: Appended to a sidecar before a restore overwrites it.
RESTORE_BACKUP_SUFFIX = ".before-restore.bak"

ProgressFn = Callable[[str], None]


class StateError(RuntimeError):
    """Raised when a state file cannot be written or read."""


@dataclass
class SlideEntry:
    """One slide's annotations, and where they came from."""

    slide_path: str
    entry: str
    annotation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"slidePath": self.slide_path, "entry": self.entry,
                "annotationCount": self.annotation_count}


@dataclass
class StateManifest:
    """What a state file contains."""

    version: int = STATE_VERSION
    created_at: str = ""
    profile_name: str = ""
    current_slide: str = ""
    slides: list[SlideEntry] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    patch_count: int = 0
    geometry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "createdAt": self.created_at,
                "profileName": self.profile_name,
                "currentSlide": self.current_slide,
                "slides": [s.to_dict() for s in self.slides],
                "models": dict(self.models), "settings": dict(self.settings),
                "patchCount": self.patch_count,
                "geometryCount": self.geometry_count}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateManifest":
        return cls(
            version=int(data.get("version", 0)),
            created_at=str(data.get("createdAt", "")),
            profile_name=str(data.get("profileName", "")),
            current_slide=str(data.get("currentSlide", "")),
            slides=[SlideEntry(str(s.get("slidePath", "")), str(s.get("entry", "")),
                               int(s.get("annotationCount", 0)))
                    for s in data.get("slides") or []],
            models=dict(data.get("models") or {}),
            settings=dict(data.get("settings") or {}),
            patch_count=int(data.get("patchCount", 0)),
            geometry_count=int(data.get("geometryCount", 0)),
        )

    def summary(self) -> str:
        when = self.created_at.split("T")[0] if self.created_at else "unknown date"
        parts = [f"Saved {when}"]
        if self.slides:
            annotations = sum(s.annotation_count for s in self.slides)
            parts.append(f"{len(self.slides)} slide(s), {annotations} annotation(s)")
        if self.patch_count:
            parts.append(f"{self.patch_count:,} patches")
        if self.geometry_count:
            parts.append(f"{self.geometry_count} descriptor(s)")
        if self.models:
            parts.append(f"{len(self.models)} model(s)")
        return " · ".join(parts) + "."


def _entry_name(slide_path: str) -> str:
    """A stable, filesystem-safe entry name for one slide's annotations.

    Hashed rather than sanitised: two slides in different folders can share a
    filename, and a collision would silently overwrite one set of annotations.
    """
    digest = hashlib.sha256(str(slide_path).encode("utf-8")).hexdigest()[:16]
    stem = Path(slide_path).stem[:60] or "slide"
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in stem)
    return f"{ANNOTATION_DIR}/{safe}-{digest}.geojson"


def sidecar_for(slide_path: str | Path) -> Path:
    return Path(slide_path).with_suffix(".geojson")


# -- saving ------------------------------------------------------------------

def save_state(path: str | Path, *, profile_json: str,
               patch_bank_path: Path | None = None,
               geometry_bank_path: Path | None = None,
               equivalences_path: Path | None = None,
               slide_paths: Iterable[str | Path] = (),
               models: dict[str, str] | None = None,
               settings: dict[str, Any] | None = None,
               current_slide: str = "",
               profile_name: str = "",
               patch_count: int = 0,
               geometry_count: int = 0,
               progress: ProgressFn | None = None) -> StateManifest:
    """Write everything into one ``.pathlearn`` file.

    *models* maps a role (``"patch"``, ``"geometry"``) to a serialised ``.cl``
    document, so a state never depends on a model file staying where it was.
    """
    path = Path(path)
    manifest = StateManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        profile_name=profile_name, current_slide=str(current_slide),
        settings=dict(settings or {}), patch_count=patch_count,
        geometry_count=geometry_count)

    if patch_bank_path:
        _checkpoint(patch_bank_path)

    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            if progress:
                progress("Writing the class palette…")
            archive.writestr(PROFILE, profile_json)

            for name, source in ((PATCH_BANK, patch_bank_path),
                                 (GEOMETRY_BANK, geometry_bank_path),
                                 (EQUIVALENCES, equivalences_path)):
                if source and Path(source).is_file():
                    if progress:
                        progress(f"Writing {name}…")
                    archive.write(Path(source), name)

            for role, document in (models or {}).items():
                entry = f"{MODEL_DIR}/{role}{'.cl'}"
                archive.writestr(entry, document)
                manifest.models[role] = entry

            seen: set[str] = set()
            for slide in slide_paths:
                key = str(slide)
                if key in seen:
                    continue
                seen.add(key)
                sidecar = sidecar_for(key)
                if not sidecar.is_file():
                    continue
                if progress:
                    progress(f"Writing annotations for {Path(key).name}…")
                text = sidecar.read_text(encoding="utf-8")
                entry = _entry_name(key)
                archive.writestr(entry, text)
                manifest.slides.append(
                    SlideEntry(key, entry, _count_features(text)))

            archive.writestr(MANIFEST,
                             json.dumps(manifest.to_dict(), indent=2))
    except OSError as exc:
        raise StateError(f"Could not write {path.name}: {exc}") from exc
    return manifest


def _count_features(text: str) -> int:
    try:
        return len(json.loads(text).get("features") or [])
    except (ValueError, AttributeError):
        return 0


def _checkpoint(path: Path) -> None:
    """Fold the write-ahead log in, so the copied .db is complete."""
    if not Path(path).is_file():
        return
    try:
        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
    except sqlite3.Error:
        pass


# -- reading -----------------------------------------------------------------

def read_manifest(path: str | Path) -> StateManifest:
    """The manifest alone, for showing what a state holds before restoring."""
    try:
        with zipfile.ZipFile(Path(path)) as archive:
            data = json.loads(archive.read(MANIFEST).decode("utf-8"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise StateError(f"{Path(path).name} is not a PathLearn state: {exc}") from exc
    manifest = StateManifest.from_dict(data)
    if manifest.version > STATE_VERSION:
        raise StateError(
            f"{Path(path).name} was written by a newer PathLearn "
            f"(state version {manifest.version}, this build reads "
            f"{STATE_VERSION}).")
    return manifest


@dataclass
class RestoreReport:
    manifest: StateManifest
    restored_slides: list[str] = field(default_factory=list)
    missing_slides: list[str] = field(default_factory=list)
    backed_up: list[str] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    profile_json: str = ""

    def summary(self) -> str:
        parts = []
        if self.restored_slides:
            parts.append(f"{len(self.restored_slides)} slide(s) restored")
        if self.missing_slides:
            parts.append(f"{len(self.missing_slides)} slide(s) not found")
        if self.models:
            parts.append(f"{len(self.models)} model(s)")
        return ("Restored. " + " · ".join(parts) + "."
                if parts else "Restored, but the state held nothing to apply.")


def load_state(path: str | Path, *,
               patch_bank_path: Path | None = None,
               geometry_bank_path: Path | None = None,
               equivalences_path: Path | None = None,
               restore_annotations: bool = True,
               progress: ProgressFn | None = None) -> RestoreReport:
    """Unpack a state over the live data.

    Destructive by nature: the banks are replaced and each slide's sidecar is
    overwritten. Every sidecar is backed up first, and a slide that is not
    where the state remembers it is reported rather than skipped silently.
    """
    path = Path(path)
    manifest = read_manifest(path)
    report = RestoreReport(manifest=manifest)

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())

            if PROFILE in names:
                report.profile_json = archive.read(PROFILE).decode("utf-8")

            for entry, target in ((PATCH_BANK, patch_bank_path),
                                  (GEOMETRY_BANK, geometry_bank_path),
                                  (EQUIVALENCES, equivalences_path)):
                if entry in names and target:
                    if progress:
                        progress(f"Restoring {entry}…")
                    _extract_to(archive, entry, Path(target))

            for role, entry in manifest.models.items():
                if entry in names:
                    report.models[role] = archive.read(entry).decode("utf-8")

            if restore_annotations:
                for slide in manifest.slides:
                    target = Path(slide.slide_path)
                    if not target.is_file():
                        report.missing_slides.append(slide.slide_path)
                        continue
                    if slide.entry not in names:
                        continue
                    if progress:
                        progress(f"Restoring annotations for {target.name}…")
                    sidecar = sidecar_for(target)
                    if sidecar.is_file():
                        backup = _unique(sidecar.with_name(
                            sidecar.name + RESTORE_BACKUP_SUFFIX))
                        shutil.copy2(sidecar, backup)
                        report.backed_up.append(str(backup))
                    sidecar.write_bytes(archive.read(slide.entry))
                    report.restored_slides.append(slide.slide_path)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise StateError(f"Could not read {path.name}: {exc}") from exc
    return report


def _extract_to(archive: zipfile.ZipFile, entry: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archive.read(entry))
    # SQLite sidecars describe the *old* database; leaving them beside a
    # replaced .db makes SQLite read a mixture of the two.
    for suffix in ("-wal", "-shm"):
        stale = target.with_name(target.name + suffix)
        if stale.exists():
            stale.unlink()


def _unique(path: Path) -> Path:
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{n}")
        n += 1
    return candidate
