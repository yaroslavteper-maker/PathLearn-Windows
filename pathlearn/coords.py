"""The coordinate-space policy for PathLearn on Windows.  Read this first.

THE ONE COORDINATE SPACE
========================
Every polygon, patch origin, prediction rectangle and pixel read in this
codebase uses **level-0 slide pixels, origin at top-left, Y increasing
downward**.  That is OpenSlide's native space and QuPath's native space.

There is **no mirrored "view-Y" space anywhere**.  The viewer draws the slide
un-mirrored, so screen coordinates map to slide coordinates by a pure
scale + translate.  Consequently *none* of the ``y_data = slideH - y_view``
mirrors that pepper the macOS code exist here.  Do not add one.

WHY THE macOS APP HAD A MIRROR (and why we drop it)
===================================================
The macOS tile renderer read ``y0 = slideH - r.maxY`` to work around a tile
tearing artifact, which drew the whole slide vertically mirrored.  Annotations
were captured in that mirrored space and stored that way, so every pipeline
that read pixels from polygon coordinates had to mirror Y back first
(``PatchSampler``, ``GeometryPipeline``, ``AnnotationExporter``, the heatmap
overlay...).  Windows has no such tearing bug, so we render upright and the
entire class of mirror bugs disappears.

WHAT THIS MEANS FOR EXISTING macOS DATA  (important)
====================================================
Ground truth is ``reference-swift-source``, and it disagrees with the handoff
docs.  ``ContentView.swift`` states it outright::

    /// The auto-saved sidecar (`.geojson` next to the slide) stays
    /// un-flipped to preserve compatibility with this app's internal load.

and ``AnnotationStore.save()`` calls ``GeoJSON.encode(annotations)`` with no
transform.  So:

* ``<slide>.geojson`` auto-saved **sidecars are MIRRORED** (view-Y space).
  They are *not* QuPath-compatible, despite what ``02-DATA-FORMATS.md`` §1 and
  ``04-DESIGN-DECISIONS.md`` §1 claim.
* Only the explicit *File -> Export GeoJSON* command mirrored Y on write, so
  files the user deliberately exported (typically ``*-qupath.geojson``) **are**
  in QuPath top-left space.

Loading a legacy sidecar without migrating would place every annotation at the
opposite end of the slide.  So sidecars carry a provenance marker:

MARKER
------
Files written by this app include a GeoJSON foreign member::

    "pathlearn": {"version": 1, "coordinateSpace": "level0-topleft"}

Foreign members are legal GeoJSON and QuPath ignores them.  Its presence means
"already in the one true space".  Its absence in a *sidecar* means "written by
macOS PathLearn, therefore mirrored, therefore migrate".  Absence in a file the
user explicitly picks via *Import* means "third-party/QuPath, therefore already
top-left, therefore do not touch" — see :class:`CoordinateSpace`.
"""

from __future__ import annotations

import enum

#: Key of the GeoJSON foreign member that stamps our coordinate space.
MARKER_KEY = "pathlearn"
#: Current value of the marker's ``version`` field.
MARKER_VERSION = 1
#: The only coordinate space this app writes.
COORDINATE_SPACE = "level0-topleft"


def marker() -> dict:
    """The foreign member stamped into every GeoJSON file we write."""
    return {"version": MARKER_VERSION, "coordinateSpace": COORDINATE_SPACE}


def has_marker(document: dict) -> bool:
    """True if *document* was written by the Windows build (already upright)."""
    m = document.get(MARKER_KEY)
    return isinstance(m, dict) and m.get("coordinateSpace") == COORDINATE_SPACE


class CoordinateSpace(enum.Enum):
    """How to interpret the Y axis of a GeoJSON file being read."""

    #: Level-0, top-left origin, Y down.  QuPath's space and ours.  No change.
    TOP_LEFT = "top-left"
    #: Legacy macOS PathLearn sidecar: Y is mirrored, needs ``y -> slideH - y``.
    LEGACY_MIRRORED = "legacy-mirrored"


class Provenance(enum.Enum):
    """Where a GeoJSON file came from, which decides the default space."""

    #: ``<slide>.geojson`` sitting next to the slide, auto-managed by the app.
    SIDECAR = "sidecar"
    #: A file the user explicitly picked via File -> Import.
    IMPORTED = "imported"


def infer_space(document: dict, provenance: Provenance) -> CoordinateSpace:
    """Decide how to interpret *document*'s Y axis.

    A marker always wins.  Without one, an auto-saved sidecar must have come
    from the macOS build and is therefore mirrored, while a deliberately
    imported file is assumed to be QuPath-style top-left.
    """
    if has_marker(document):
        return CoordinateSpace.TOP_LEFT
    if provenance is Provenance.SIDECAR:
        return CoordinateSpace.LEGACY_MIRRORED
    return CoordinateSpace.TOP_LEFT
