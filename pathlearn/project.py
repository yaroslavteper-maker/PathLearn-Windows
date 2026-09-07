"""A project: one composition result per slide, accumulated across sessions.

The unit of work here is a **cohort**, not a slide.  You open a slide, predict,
add the breakdown to the project, close it, open the next one.  The project
file holds one entry per slide and exports the lot as a single table.

WHY THIS IS NOT JUST A CSV ON DISK
==================================
The stored form is JSON and the table is generated from it, because a CSV
cannot hold the two things that keep the table honest:

* **The model's full class list per entry.**  Two slides graded by different
  models have different columns, and a wide table has to reconcile them.  A
  blank and a zero mean genuinely different things — see below — and once a
  row is flattened into CSV that distinction is gone for good.
* **Per-entry provenance.**  The confidence threshold, pixel size, grid size
  and extractor identity belong to the run, not to the file.  Appending rows
  to a flat CSV silently mixes runs made under different settings.

Export is therefore one-way and repeatable: edit nothing, re-export anything.

BLANK IS NOT ZERO
=================
In the wide table a cell is:

* a **number** when the entry's model had that class — including ``0.0``,
  meaning the model looked for it and found none;
* **blank** when the entry's model had no such class at all, so the question
  was never asked.

Reading a blank as zero would turn "not measured" into "measured as absent",
which is exactly the error that makes a cohort table lie.  ``percent_for``
returns ``None`` for the second case and a float for the first.

WHAT THE PERCENTAGES ARE OF
===========================
Every entry inherits the denominator described in ``pipeline/composition.py``:
the tissue that was predicted over on that slide, not the slide's area.  Two
entries are comparable only if the regions were drawn comparably — which is
why the mm² columns are exported alongside the percentages, and why the run
settings travel with each row.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .models.prediction import PredictionSet
from .pipeline.annotation_stats import AnnotationBreakdown
from .pipeline.composition import CompositionReport, format_area

log = logging.getLogger(__name__)

PROJECT_SUFFIX = ".pathlearn-project.json"

PROJECT_VERSION = 2

#: What produced an entry.  A slide can carry one of each: what the model
#: called it, and what you drew.  They are separate rows precisely so they can
#: be compared rather than silently conflated.
PREDICTED = "predicted"
ANNOTATED = "annotated"

#: What one "unit" is for each source — tiles are sampled, regions are drawn.
UNIT_NAMES = {PREDICTED: "tiles", ANNOTATED: "regions"}


class ProjectError(ValueError):
    """Raised when a project file cannot be read or written."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class EntryShare:
    """One class's numbers within one slide's entry."""

    label: str
    tiles: int
    cells: int
    area_px: float
    area_mm2: float | None
    tile_percent: float
    area_percent: float
    mean_confidence: float

    def to_dict(self) -> dict:
        return {"tiles": self.tiles, "cells": self.cells,
                "areaPx": self.area_px, "areaMm2": self.area_mm2,
                "tilePercent": self.tile_percent,
                "areaPercent": self.area_percent,
                "meanConfidence": self.mean_confidence}

    @classmethod
    def from_dict(cls, label: str, data: dict) -> "EntryShare":
        area_mm2 = data.get("areaMm2")
        return cls(label=label, tiles=int(data.get("tiles", 0)),
                   cells=int(data.get("cells", 0)),
                   area_px=float(data.get("areaPx", 0.0)),
                   area_mm2=None if area_mm2 is None else float(area_mm2),
                   tile_percent=float(data.get("tilePercent", 0.0)),
                   area_percent=float(data.get("areaPercent", 0.0)),
                   mean_confidence=float(data.get("meanConfidence", 0.0)))


@dataclass
class ProjectEntry:
    """One slide's composition, frozen at the moment it was added."""

    slide_path: str
    slide_name: str = ""
    added: str = field(default_factory=_now)
    model_name: str = ""
    extractor_identity: str = ""
    #: Every class the model could have produced — the key to blank vs zero.
    model_classes: list[str] = field(default_factory=list)
    shares: dict[str, EntryShare] = field(default_factory=dict)
    total_tiles: int = 0
    total_cells: int = 0
    total_area_px: float = 0.0
    total_area_mm2: float | None = None
    excluded_low_confidence: int = 0
    min_confidence: float = 0.0
    cell_size: int = 0
    mpp: float | None = None
    notes: str = ""
    #: PREDICTED or ANNOTATED.  Version 1 files have no field and are all
    #: predictions, which is exactly what the default gives them.
    source: str = PREDICTED
    #: Which annotations were counted — "all" or those checked under Use.
    scope: str = ""
    # Annotation-only bookkeeping, meaningless for a prediction row.
    subtractive_count: int = 0
    carved_px: float = 0.0
    orphan_subtractive: int = 0
    overlapping_pairs: int = 0

    def __post_init__(self) -> None:
        if not self.slide_name:
            self.slide_name = Path(self.slide_path).name or self.slide_path

    # -- reading ----------------------------------------------------------

    def percent_for(self, label: str) -> float | None:
        """Area share of *label*, or None if this model had no such class.

        The None is load-bearing: a blank cell means the question was never
        asked, and a 0.0 means it was asked and the answer was none.
        """
        share = self.shares.get(label)
        if share is not None:
            return share.area_percent
        return 0.0 if label in self.model_classes else None

    def mm2_for(self, label: str) -> float | None:
        share = self.shares.get(label)
        if share is not None:
            return share.area_mm2
        if label in self.model_classes and self.total_area_mm2 is not None:
            return 0.0
        return None

    def area_text(self) -> str:
        return format_area(self.total_area_px, self.total_area_mm2)

    @property
    def is_annotated(self) -> bool:
        return self.source == ANNOTATED

    @property
    def unit_name(self) -> str:
        """"tiles" or "regions" — what ``total_tiles`` is counting here."""
        return UNIT_NAMES.get(self.source, "units")

    @property
    def dominant(self) -> str:
        """The class holding the most tissue, or "" for an empty entry."""
        if not self.shares:
            return ""
        return max(self.shares.values(), key=lambda s: s.area_percent).label

    # -- building ---------------------------------------------------------

    @classmethod
    def from_report(cls, report: CompositionReport, predictions: PredictionSet,
                    *, slide_path: str = "", model_name: str = "",
                    notes: str = "") -> "ProjectEntry":
        """Freeze a composition run into an entry.

        The model's class list comes from the ``PredictionSet`` rather than
        from whatever model happens to be loaded now — the entry has to record
        the run that produced it, not the state of the app afterwards.
        """
        path = slide_path or predictions.slide_path or report.slide_name
        return cls(
            slide_path=str(path),
            model_name=model_name,
            extractor_identity=predictions.extractor_identity,
            model_classes=list(predictions.class_labels),
            shares={s.label: EntryShare(
                label=s.label, tiles=s.tiles, cells=s.cells,
                area_px=s.area_px, area_mm2=s.area_mm2,
                tile_percent=s.tile_percent, area_percent=s.area_percent,
                mean_confidence=s.mean_confidence) for s in report.shares},
            total_tiles=report.total_tiles,
            total_cells=report.total_cells,
            total_area_px=report.total_area_px,
            total_area_mm2=report.total_area_mm2,
            excluded_low_confidence=report.excluded_low_confidence,
            min_confidence=report.min_confidence,
            cell_size=report.cell_size,
            mpp=report.mpp,
            notes=notes)

    @classmethod
    def from_breakdown(cls, report: AnnotationBreakdown, *, slide_path: str,
                       palette_classes: Sequence[str] = (),
                       notes: str = "") -> "ProjectEntry":
        """Freeze an annotation class breakdown into an entry.

        *palette_classes* is the classification profile in force — the set of
        classes you could have drawn.  It plays exactly the part a model's
        class list plays for a prediction: a palette class you drew none of is
        a real **0**, while a class that was not in your palette at all stays
        **blank**, because the question was never on the table.  Pass nothing
        and every class you did not draw is blank, which is the safe reading
        when the palette is unknown.
        """
        classes = list(palette_classes)
        for tally in report.tallies:
            if tally.label not in classes:
                classes.append(tally.label)
        return cls(
            slide_path=str(slide_path),
            source=ANNOTATED,
            model_classes=classes,
            scope=report.scope,
            shares={t.label: EntryShare(
                label=t.label, tiles=t.count, cells=t.count,
                area_px=t.area_px, area_mm2=t.area_mm2,
                tile_percent=t.count_percent, area_percent=t.area_percent,
                # Regions carry no confidence — nobody scored them, you drew
                # them. Zero here means "not applicable", and the exports
                # leave the column empty for annotated rows rather than
                # printing a 0.00 that reads as no confidence.
                mean_confidence=0.0) for t in report.tallies},
            total_tiles=report.total_count,
            total_cells=report.total_count,
            total_area_px=report.total_area_px,
            total_area_mm2=report.total_area_mm2,
            mpp=report.mpp,
            subtractive_count=report.subtractive_count,
            carved_px=report.carved_px,
            orphan_subtractive=report.orphan_subtractive,
            overlapping_pairs=report.overlapping_pairs,
            notes=notes)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "slidePath": self.slide_path, "slideName": self.slide_name,
            "source": self.source, "scope": self.scope,
            "subtractiveCount": self.subtractive_count,
            "carvedPx": self.carved_px,
            "orphanSubtractive": self.orphan_subtractive,
            "overlappingPairs": self.overlapping_pairs,
            "added": self.added, "modelName": self.model_name,
            "extractorIdentity": self.extractor_identity,
            "modelClasses": list(self.model_classes),
            "shares": {k: v.to_dict() for k, v in self.shares.items()},
            "totalTiles": self.total_tiles, "totalCells": self.total_cells,
            "totalAreaPx": self.total_area_px,
            "totalAreaMm2": self.total_area_mm2,
            "excludedLowConfidence": self.excluded_low_confidence,
            "minConfidence": self.min_confidence,
            "cellSize": self.cell_size, "mpp": self.mpp, "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectEntry":
        total_mm2 = data.get("totalAreaMm2")
        mpp = data.get("mpp")
        return cls(
            slide_path=str(data.get("slidePath") or ""),
            slide_name=str(data.get("slideName") or ""),
            added=str(data.get("added") or _now()),
            model_name=str(data.get("modelName") or ""),
            extractor_identity=str(data.get("extractorIdentity") or ""),
            model_classes=[str(x) for x in (data.get("modelClasses") or [])],
            shares={str(k): EntryShare.from_dict(str(k), v)
                    for k, v in (data.get("shares") or {}).items()},
            total_tiles=int(data.get("totalTiles", 0)),
            total_cells=int(data.get("totalCells", 0)),
            total_area_px=float(data.get("totalAreaPx", 0.0)),
            total_area_mm2=None if total_mm2 is None else float(total_mm2),
            excluded_low_confidence=int(data.get("excludedLowConfidence", 0)),
            min_confidence=float(data.get("minConfidence", 0.0)),
            cell_size=int(data.get("cellSize", 0)),
            mpp=None if mpp is None else float(mpp),
            notes=str(data.get("notes") or ""),
            source=str(data.get("source") or PREDICTED),
            scope=str(data.get("scope") or ""),
            subtractive_count=int(data.get("subtractiveCount", 0)),
            carved_px=float(data.get("carvedPx", 0.0)),
            orphan_subtractive=int(data.get("orphanSubtractive", 0)),
            overlapping_pairs=int(data.get("overlappingPairs", 0)))


@dataclass
class Project:
    """A named collection of per-slide composition entries."""

    name: str = "Untitled project"
    path: Path | None = None
    notes: str = ""
    created: str = field(default_factory=_now)
    modified: str = field(default_factory=_now)
    entries: list[ProjectEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    # -- contents ---------------------------------------------------------

    def entry_for(self, slide_path: str,
                  source: str = PREDICTED) -> ProjectEntry | None:
        """The entry for one slide **from one source**.

        Keyed on both, because a slide legitimately carries a predicted row
        and an annotated row at once — keying on the path alone would make
        filing your tracing silently destroy the model's answer.
        """
        key = _key(slide_path)
        return next((e for e in self.entries
                     if _key(e.slide_path) == key and e.source == source), None)

    def entries_for(self, slide_path: str) -> list[ProjectEntry]:
        key = _key(slide_path)
        return [e for e in self.entries if _key(e.slide_path) == key]

    def add(self, entry: ProjectEntry, *, replace: bool = True) -> bool:
        """Add *entry*; returns True if it replaced an existing slide.

        Re-running a prediction on a slide already in the project usually means
        the earlier number is superseded, so replacing is the default. Passing
        ``replace=False`` keeps both, which is what you want when deliberately
        comparing two models on one slide.
        """
        existing = self.entry_for(entry.slide_path, entry.source)
        if existing is not None and replace:
            self.entries[self.entries.index(existing)] = entry
            self.modified = _now()
            return True
        self.entries.append(entry)
        self.modified = _now()
        return False

    def remove(self, slide_path: str, source: str | None = None) -> bool:
        """Remove one row, or every row for a slide when *source* is None."""
        doomed = ([e for e in self.entries_for(slide_path)] if source is None
                  else [e for e in self.entries_for(slide_path)
                        if e.source == source])
        if not doomed:
            return False
        for entry in doomed:
            self.entries.remove(entry)
        self.modified = _now()
        return True

    def sources(self) -> list[str]:
        """Which sources appear here, in a stable order."""
        return [s for s in (PREDICTED, ANNOTATED)
                if any(e.source == s for e in self.entries)]

    def class_union(self) -> list[str]:
        """Every class any entry could report, in first-seen order.

        First-seen rather than alphabetical, so a model's own ordering —
        PanIN-1a, 1b, 2, 3 — survives into the table instead of being
        scrambled into something that reads as arbitrary.
        """
        seen: list[str] = []
        for entry in self.entries:
            for label in list(entry.model_classes) + list(entry.shares):
                if label not in seen:
                    seen.append(label)
        return seen

    def summary(self) -> str:
        if self.is_empty:
            return f"{self.name} — empty. Add a slide's composition to start."
        classes = self.class_union()
        slides = len({_key(e.slide_path) for e in self.entries})
        text = (f"{self.name} — {slides} slide(s), {len(self.entries)} row(s), "
                f"{len(classes)} class(es)")
        total = sum(e.total_area_mm2 for e in self.entries
                    if e.total_area_mm2 is not None)
        if total:
            text += f", {total:.2f} mm2 of tissue"
        mixed = {e.extractor_identity for e in self.entries if e.extractor_identity}
        if len(mixed) > 1:
            # Two feature spaces in one table is a real problem, not a detail.
            text += f"  — WARNING: {len(mixed)} different extractors"
        return text + "."

    def mixed_extractors(self) -> list[str]:
        """Distinct extractor identities across entries; >1 is a warning."""
        seen: list[str] = []
        for entry in self.entries:
            if entry.extractor_identity and entry.extractor_identity not in seen:
                seen.append(entry.extractor_identity)
        return seen

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"version": PROJECT_VERSION, "name": self.name,
                "notes": self.notes, "created": self.created,
                "modified": self.modified,
                "entries": [e.to_dict() for e in self.entries]}

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ProjectError("This project has nowhere to save to.")
        self.modified = _now()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write beside the target and swap, so an interrupted save cannot
            # leave a half-written project where a whole one used to be.
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.write_text(json.dumps(self.to_dict(), indent=2),
                                 encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            raise ProjectError(f"Could not save the project: {exc}") from exc
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ProjectError(f"Could not open the project: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProjectError(
                f"{source.name} is not a readable project file: {exc}") from exc
        if not isinstance(data, dict) or "entries" not in data:
            raise ProjectError(f"{source.name} is not a PathLearn project.")
        version = int(data.get("version", PROJECT_VERSION))
        if version > PROJECT_VERSION:
            # Refuse rather than silently drop the fields we cannot read.
            raise ProjectError(
                f"{source.name} was written by a newer PathLearn "
                f"(format {version}, this build reads {PROJECT_VERSION}).")
        project = cls(
            name=str(data.get("name") or source.stem),
            path=source,
            notes=str(data.get("notes") or ""),
            created=str(data.get("created") or _now()),
            modified=str(data.get("modified") or _now()),
            entries=[ProjectEntry.from_dict(d)
                     for d in (data.get("entries") or [])])
        return project

    @classmethod
    def create(cls, path: str | Path, name: str = "") -> "Project":
        target = Path(path)
        project = cls(name=name or target.stem, path=target)
        project.save()
        return project

    # -- export -----------------------------------------------------------

    def to_csv(self, *, layout: str = "wide") -> str:
        """The whole project as one table.

        ``wide`` is one row per slide with a percentage column per class — the
        shape you want for a cohort.  ``long`` is one row per slide and class,
        which is the shape statistics packages want.
        """
        if layout == "long":
            return "\n".join(_long_rows(self)) + "\n"
        if layout != "wide":
            raise ProjectError(f"Unknown table layout: {layout!r}")
        return "\n".join(_wide_rows(self)) + "\n"

    def to_xlsx(self, path: str | Path) -> Path:
        """Write a real workbook: Composition, Detail and About sheets.

        Values go in as numbers, not strings, so the columns can be charted and
        averaged without a re-import step.
        """
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Alignment, Font
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ProjectError(
                "Excel export needs openpyxl, which is not installed.\n\n"
                "Install it with:  pip install openpyxl\n\n"
                "Or use Export CSV, which opens in Excel directly.") from exc

        target = Path(path)
        classes = self.class_union()
        workbook = Workbook()

        sheet = workbook.active
        sheet.title = "Composition"
        header = (["Slide", "Source", "Added", "Model", "Count",
                   "Total area (mm2)"]
                  + [f"{label} %" for label in classes]
                  + [f"{label} mm2" for label in classes])
        sheet.append(header)
        for entry in self.entries:
            row: list = [entry.slide_name, entry.source, entry.added,
                         entry.model_name, entry.total_tiles,
                         entry.total_area_mm2]
            # None becomes an empty cell, which is the whole point: blank
            # means the model had no such class, zero means it found none.
            row += [entry.percent_for(label) for label in classes]
            row += [entry.mm2_for(label) for label in classes]
            sheet.append(row)
        # Format, do not round: the cell keeps full precision for charting
        # and averaging, and merely displays two decimals.
        _format_columns(sheet, {6: "0.0000"}
                        | {7 + i: "0.00" for i in range(len(classes))}
                        | {7 + len(classes) + i: "0.0000"
                           for i in range(len(classes))})
        _style(sheet, Font, Alignment, len(header))

        detail = workbook.create_sheet("Detail")
        detail.append(["Slide", "Source", "Class", "Count", "Count %",
                       "Cells", "Area (px2)", "Area (mm2)", "Area %",
                       "Mean confidence", "Confidence threshold",
                       "Excluded below threshold", "Cell size (px)",
                       "Microns per pixel", "Model", "Extractor", "Scope",
                       "Subtractive excluded", "Carved out (px2)",
                       "Orphan subtractive", "Overlapping pairs", "Added",
                       "Slide path"])
        for entry in self.entries:
            for label in classes:
                share = entry.shares.get(label)
                if share is None and label not in entry.model_classes:
                    continue  # never asked; do not manufacture a row
                annotated = entry.is_annotated
                detail.append([
                    entry.slide_name, entry.source, label,
                    share.tiles if share else 0,
                    share.tile_percent if share else 0.0,
                    share.cells if share else 0,
                    share.area_px if share else 0.0,
                    share.area_mm2 if share else entry.mm2_for(label),
                    share.area_percent if share else 0.0,
                    # Empty, not zero: a drawn region was never scored, and a
                    # 0.00 in a confidence column reads as "no confidence".
                    None if annotated else (share.mean_confidence
                                            if share else None),
                    None if annotated else entry.min_confidence,
                    None if annotated else entry.excluded_low_confidence,
                    None if annotated else entry.cell_size,
                    entry.mpp, entry.model_name, entry.extractor_identity,
                    entry.scope,
                    entry.subtractive_count if annotated else None,
                    entry.carved_px if annotated else None,
                    entry.orphan_subtractive if annotated else None,
                    entry.overlapping_pairs if annotated else None,
                    entry.added, entry.slide_path])
        _format_columns(detail, {5: "0.00", 7: "0.00", 8: "0.0000",
                                 9: "0.00", 10: "0.0000", 11: "0.00",
                                 14: "0.0000", 19: "0"})
        _style(detail, Font, Alignment, 23)

        about = workbook.create_sheet("About")
        for row in _about_rows(self):
            about.append(row)
        about.column_dimensions["A"].width = 30
        about.column_dimensions["B"].width = 80
        for cell in about["A"]:
            cell.font = Font(bold=True)
        for cell in about["B"]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

        try:
            workbook.save(target)
        except OSError as exc:
            raise ProjectError(f"Could not write the workbook: {exc}") from exc
        return target


def _format_columns(sheet, formats: dict[int, str]) -> None:
    """Apply a display format to whole columns, header row excepted."""
    for column, pattern in formats.items():
        for row in sheet.iter_rows(min_row=2, min_col=column, max_col=column):
            for cell in row:
                cell.number_format = pattern


def _style(sheet, Font, Alignment, columns: int) -> None:
    """Bold the header and freeze it, so a long cohort stays readable."""
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    sheet.freeze_panes = "B2"
    sheet.column_dimensions["A"].width = 32


def _about_rows(project: Project) -> list[list[str]]:
    slides = len({_key(e.slide_path) for e in project.entries})
    rows = [
        ["PathLearn project", project.name],
        ["Slides", str(slides)],
        ["Rows", str(len(project.entries))],
        ["Created", project.created],
        ["Last modified", project.modified],
        ["Notes", project.notes],
        ["", ""],
        ["Source column",
         "'predicted' is what a model called the tissue, counted in tiles. "
         "'annotated' is what you drew, counted in regions. A slide can have "
         "one of each. They measure different things and are separate rows "
         "so they can be compared - do not average them together."],
        ["Count column",
         "Tiles for a predicted row, traced regions for an annotated one. "
         "The Source column says which."],
        ["Annotated rows",
         "Confidence columns are empty because a drawn region was never "
         "scored. Subtractive polygons are excluded from the count and their "
         "area deducted from the region enclosing them; where regions "
         "overlap, the area figures are upper bounds."],
        ["What the percentages are of",
         "For a predicted row: the tissue predicted over on that slide - NOT "
         "the slide's area, and not counting tiles the sampler rejected as "
         "background. For an annotated row: the regions you drew, which is "
         "not the same denominator. Compare within a source, not across."],
        ["Blank versus zero",
         "A blank cell means that slide's model had no such class, so the "
         "question was never asked. A zero means the model looked and found "
         "none. Do not fill blanks with zeros."],
        ["Comparing slides",
         "Percentages compare only if the regions were drawn comparably. The "
         "mm2 columns are the safer figure to compare across slides."],
        ["Per-run settings",
         "Confidence threshold, pixel size and grid size vary per entry and "
         "are on the Detail sheet. Rows made under different settings are not "
         "directly comparable."],
    ]
    mixed = project.mixed_extractors()
    if len(mixed) > 1:
        rows.append(["WARNING - mixed extractors",
                     "These entries came from " + str(len(mixed))
                     + " different feature extractors ("
                     + ", ".join(mixed)
                     + "). Two feature spaces in one table are not "
                       "comparable; re-run the odd ones out."])
    return rows


def _csv(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(character in text for character in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _number(value: float | None, places: int = 2) -> str:
    return "" if value is None else f"{value:.{places}f}"


def _wide_rows(project: Project) -> list[str]:
    classes = project.class_union()
    header = (["Slide", "Source", "Added", "Model", "Count",
               "Total area (mm2)"]
              + [f"{label} %" for label in classes]
              + [f"{label} mm2" for label in classes])
    rows = [",".join(_csv(h) for h in header)]
    for entry in project.entries:
        values = [_csv(entry.slide_name), _csv(entry.source),
                  _csv(entry.added), _csv(entry.model_name),
                  str(entry.total_tiles), _number(entry.total_area_mm2, 4)]
        values += [_number(entry.percent_for(label)) for label in classes]
        values += [_number(entry.mm2_for(label), 4) for label in classes]
        rows.append(",".join(values))
    rows.append("")
    rows += [f"# {_csv(a)},{_csv(b)}" for a, b in _about_rows(project)]
    return rows


def _long_rows(project: Project) -> list[str]:
    classes = project.class_union()
    header = ["Slide", "Source", "Class", "Count", "Count %", "Cells",
              "Area (px2)", "Area (mm2)", "Area %", "Mean confidence",
              "Confidence threshold", "Excluded below threshold",
              "Cell size (px)", "Microns per pixel", "Model", "Extractor",
              "Scope", "Subtractive excluded", "Carved out (px2)",
              "Orphan subtractive", "Overlapping pairs", "Added",
              "Slide path"]
    rows = [",".join(_csv(h) for h in header)]
    for entry in project.entries:
        for label in classes:
            share = entry.shares.get(label)
            if share is None and label not in entry.model_classes:
                continue
            annotated = entry.is_annotated
            rows.append(",".join([
                _csv(entry.slide_name), _csv(entry.source), _csv(label),
                str(share.tiles if share else 0),
                _number(share.tile_percent if share else 0.0),
                str(share.cells if share else 0),
                _number(share.area_px if share else 0.0, 0),
                _number(share.area_mm2 if share else entry.mm2_for(label), 4),
                _number(share.area_percent if share else 0.0),
                "" if annotated else _number(
                    share.mean_confidence if share else None, 4),
                "" if annotated else _number(entry.min_confidence),
                "" if annotated else str(entry.excluded_low_confidence),
                "" if annotated else str(entry.cell_size),
                _number(entry.mpp, 4), _csv(entry.model_name),
                _csv(entry.extractor_identity), _csv(entry.scope),
                str(entry.subtractive_count) if annotated else "",
                _number(entry.carved_px, 0) if annotated else "",
                str(entry.orphan_subtractive) if annotated else "",
                str(entry.overlapping_pairs) if annotated else "",
                _csv(entry.added), _csv(entry.slide_path)]))
    return rows


def _key(slide_path: str) -> str:
    """Identity for a slide.

    Case-folded on Windows, where F:\\A.svs and f:\\a.svs are one file — two
    entries for one slide would double-count it in every cohort figure.
    """
    return str(Path(slide_path)).casefold()


def default_project_dir() -> Path:
    return Path.home() / "PathLearn Projects"
