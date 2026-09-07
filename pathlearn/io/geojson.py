"""QuPath-dialect GeoJSON codec — ported from ``Models/GeoJSON.swift``.

Field names and semantics are reproduced verbatim so the user's existing files
keep loading and QuPath interop keeps working (``02-DATA-FORMATS.md`` §1).

Two deliberate differences from the Swift:

1. We stamp a :data:`pathlearn.coords.MARKER_KEY` foreign member so a file's
   coordinate space is self-describing (the macOS build had no way to tell a
   mirrored sidecar from an upright QuPath export).
2. The decoder accepts the annotation ``id`` at either the Feature level (where
   the Swift encoder puts it) or inside ``properties`` (where
   ``02-DATA-FORMATS.md`` §1 documents it).  Both appear in the wild.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterable, Sequence

from ..coords import MARKER_KEY, marker
from ..models.annotation import Annotation, AnnotationColor, Point

DEFAULT_CLASSIFICATION = "Unlabeled"


def encode(annotations: Iterable[Annotation], *, stamp_marker: bool = True) -> str:
    """Serialise *annotations* to a QuPath-dialect ``FeatureCollection``.

    Rings are closed (first point repeated last), matching the Swift encoder.
    Keys are sorted and the output indented, so sidecars diff cleanly.
    """
    features: list[dict[str, Any]] = []
    for ann in annotations:
        ring: list[list[float]] = [[p.x, p.y] for p in ann.points]
        if len(ring) >= 2 and ring[0] != ring[-1]:
            ring.append(list(ring[0]))

        properties: dict[str, Any] = {
            "objectType": "annotation",
            "classification": {
                "name": ann.classification,
                "color": list(ann.color.as_tuple()),
            },
        }
        # Match the Swift encoder: only emit these when they differ from the
        # default, so files stay small and QuPath-clean.
        if ann.name:
            properties["name"] = ann.name
        if not ann.is_visible:
            properties["isVisible"] = False
        # Only written when False, so a legacy file without the key reads as
        # selected — which is what keeps existing sidecars usable.
        if not ann.is_selected:
            properties["selected"] = False
        if ann.is_subtractive:
            properties["subtractive"] = True

        features.append({
            "type": "Feature",
            "id": str(ann.id),
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": properties,
        })

    document: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if stamp_marker:
        document[MARKER_KEY] = marker()
    return json.dumps(document, indent=2, sort_keys=True)


def decode_document(text: str) -> dict[str, Any]:
    """Parse GeoJSON text into a raw dict (so callers can inspect the marker)."""
    obj = json.loads(text)
    return obj if isinstance(obj, dict) else {}


def decode_features(document: dict[str, Any]) -> list[Annotation]:
    """Extract annotations from an already-parsed ``FeatureCollection``.

    Coordinates are returned exactly as written; applying any coordinate-space
    transform is the caller's job (see :mod:`pathlearn.coords`).
    """
    result: list[Annotation] = []
    for feature in document.get("features") or []:
        if not isinstance(feature, dict):
            continue
        ring = _first_ring(feature.get("geometry"))
        if ring is None:
            continue

        points = [Point(float(pair[0]), float(pair[1]))
                  for pair in ring
                  if isinstance(pair, Sequence) and len(pair) >= 2]
        # Drop the closing point; we store open rings internally.
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) < 3:
            continue

        props = feature.get("properties") or {}
        if not isinstance(props, dict):
            props = {}

        classification = DEFAULT_CLASSIFICATION
        color = AnnotationColor.default()
        cls = props.get("classification")
        if isinstance(cls, dict):
            if isinstance(cls.get("name"), str):
                classification = cls["name"]
            raw_color = cls.get("color")
            if isinstance(raw_color, Sequence) and not isinstance(raw_color, (str, bytes)):
                if len(raw_color) >= 3:
                    color = AnnotationColor.from_sequence(raw_color)

        name = props.get("name") if isinstance(props.get("name"), str) else None
        if not name:
            name = None
        is_visible = props.get("isVisible")
        is_visible = True if not isinstance(is_visible, bool) else is_visible
        is_selected = props.get("selected")
        is_selected = True if not isinstance(is_selected, bool) else is_selected
        is_subtractive = props.get("subtractive")
        is_subtractive = False if not isinstance(is_subtractive, bool) else is_subtractive

        result.append(Annotation(
            id=_parse_id(feature, props),
            points=points,
            classification=classification,
            color=color,
            name=name,
            is_visible=is_visible,
            is_selected=is_selected,
            is_subtractive=is_subtractive,
        ))
    return result


def decode(text: str) -> list[Annotation]:
    """Convenience: parse *text* and return its annotations, untransformed."""
    return decode_features(decode_document(text))


def _first_ring(geometry: Any) -> list[Any] | None:
    """Outer ring of a Polygon, or of the first polygon of a MultiPolygon."""
    if not isinstance(geometry, dict):
        return None
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if not isinstance(coords, list) or not coords:
        return None
    if kind == "Polygon":
        ring = coords[0]
    elif kind == "MultiPolygon":
        # Swift flat-maps every polygon then takes the first ring.
        rings = [r for poly in coords if isinstance(poly, list) for r in poly]
        ring = rings[0] if rings else None
    else:
        return None
    return ring if isinstance(ring, list) else None


def _parse_id(feature: dict[str, Any], props: dict[str, Any]) -> uuid.UUID:
    """Feature-level ``id`` (Swift) or ``properties.id`` (docs); else fresh."""
    for candidate in (feature.get("id"), props.get("id")):
        if isinstance(candidate, str):
            try:
                return uuid.UUID(candidate)
            except ValueError:
                continue
    return uuid.uuid4()
