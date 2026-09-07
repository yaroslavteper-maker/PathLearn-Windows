"""GeoJSON codec + coordinate-space tests.

These guard the two things that would silently corrupt the user's data:
QuPath round-tripping, and the legacy mirrored-sidecar migration.
"""

from __future__ import annotations

import json
import uuid

import pytest

from pathlearn.coords import (MARKER_KEY, CoordinateSpace, Provenance, has_marker,
                              infer_space)
from pathlearn.io import geojson
from pathlearn.models.annotation import Annotation, AnnotationColor, Point


def make_annotation(**kwargs) -> Annotation:
    defaults = dict(
        points=[Point(10, 20), Point(110, 20), Point(110, 120), Point(10, 120)],
        classification="PaNIN-3",
        color=AnnotationColor(230, 210, 30),
    )
    defaults.update(kwargs)
    return Annotation(**defaults)


class TestRoundTrip:
    def test_preserves_all_fields(self):
        original = make_annotation(name="lesion A", is_visible=False, is_subtractive=True)
        [restored] = geojson.decode(geojson.encode([original]))

        assert restored.id == original.id
        assert restored.points == original.points
        assert restored.classification == "PaNIN-3"
        assert restored.color == AnnotationColor(230, 210, 30)
        assert restored.name == "lesion A"
        assert restored.is_visible is False
        assert restored.is_subtractive is True

    def test_defaults_are_omitted_from_output(self):
        """Match the Swift encoder: only non-default flags are written."""
        props = json.loads(geojson.encode([make_annotation()]))["features"][0]["properties"]
        assert "isVisible" not in props
        assert "subtractive" not in props
        assert "name" not in props

    def test_ring_is_closed_on_write_and_opened_on_read(self):
        ann = make_annotation()
        ring = json.loads(geojson.encode([ann]))["features"][0]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "ring must be closed on disk"
        assert len(ring) == len(ann.points) + 1
        # ...and we store it open again internally.
        assert len(geojson.decode(geojson.encode([ann]))[0].points) == len(ann.points)

    def test_qupath_shape(self):
        doc = json.loads(geojson.encode([make_annotation()]))
        assert doc["type"] == "FeatureCollection"
        feature = doc["features"][0]
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"
        assert feature["properties"]["objectType"] == "annotation"
        assert feature["properties"]["classification"] == {
            "name": "PaNIN-3", "color": [230, 210, 30],
        }


class TestDecoderTolerance:
    def test_reads_id_from_properties(self):
        """02-DATA-FORMATS.md documents properties.id; Swift writes feature.id."""
        ident = uuid.uuid4()
        doc = json.loads(geojson.encode([make_annotation()]))
        del doc["features"][0]["id"]
        doc["features"][0]["properties"]["id"] = str(ident)
        assert geojson.decode(json.dumps(doc))[0].id == ident

    def test_generates_id_when_missing_or_invalid(self):
        doc = json.loads(geojson.encode([make_annotation()]))
        doc["features"][0]["id"] = "not-a-uuid"
        assert isinstance(geojson.decode(json.dumps(doc))[0].id, uuid.UUID)

    def test_multipolygon_takes_first_ring(self):
        doc = {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "MultiPolygon", "coordinates": [
                [[[0, 0], [10, 0], [10, 10], [0, 0]]],
                [[[50, 50], [60, 50], [60, 60], [50, 50]]],
            ]},
            "properties": {},
        }]}
        [ann] = geojson.decode(json.dumps(doc))
        assert ann.points == [Point(0, 0), Point(10, 0), Point(10, 10)]

    def test_missing_classification_falls_back(self):
        doc = {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 0]]]},
            "properties": {},
        }]}
        [ann] = geojson.decode(json.dumps(doc))
        assert ann.classification == "Unlabeled"
        assert ann.color == AnnotationColor.default()

    def test_degenerate_polygons_are_dropped(self):
        doc = {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0]]]},
            "properties": {},
        }]}
        assert geojson.decode(json.dumps(doc)) == []

    def test_empty_and_malformed_documents(self):
        assert geojson.decode('{"type":"FeatureCollection","features":[]}') == []
        assert geojson.decode("[]") == []


class TestCoordinateSpace:
    def test_our_files_are_marked(self):
        doc = json.loads(geojson.encode([make_annotation()]))
        assert has_marker(doc)
        assert infer_space(doc, Provenance.SIDECAR) is CoordinateSpace.TOP_LEFT

    def test_unmarked_sidecar_is_legacy_mirrored(self):
        """A sidecar with no marker came from macOS PathLearn, so it is mirrored."""
        doc = json.loads(geojson.encode([make_annotation()]))
        del doc[MARKER_KEY]
        assert infer_space(doc, Provenance.SIDECAR) is CoordinateSpace.LEGACY_MIRRORED

    def test_unmarked_import_is_assumed_qupath(self):
        """An explicitly imported file is third-party, therefore already upright."""
        doc = json.loads(geojson.encode([make_annotation()]))
        del doc[MARKER_KEY]
        assert infer_space(doc, Provenance.IMPORTED) is CoordinateSpace.TOP_LEFT

    def test_marker_does_not_disturb_qupath_parsing(self):
        """Foreign members are legal GeoJSON; the feature list is untouched."""
        doc = json.loads(geojson.encode([make_annotation()]))
        assert set(doc) == {"type", "features", MARKER_KEY}


class TestMirroring:
    def test_mirror_is_an_involution(self):
        ann = make_annotation()
        assert ann.mirrored_y(1000).mirrored_y(1000).points == ann.points

    def test_mirror_preserves_subtractive_flag(self):
        """The Swift mirrorYForImport dropped this; we must not."""
        ann = make_annotation(is_subtractive=True, name="carve")
        flipped = ann.mirrored_y(1000)
        assert flipped.is_subtractive is True
        assert flipped.name == "carve"
        assert flipped.id == ann.id

    def test_mirror_maps_y_correctly(self):
        ann = make_annotation(points=[Point(5, 100), Point(15, 200), Point(25, 300)])
        assert ann.mirrored_y(1000).points == [Point(5, 900), Point(15, 800), Point(25, 700)]


class TestGeometry:
    def test_shoelace_area_of_a_square(self):
        ann = make_annotation(points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)])
        assert ann.area_in_slide_pixels == pytest.approx(10_000.0)

    def test_area_is_orientation_independent(self):
        square = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        assert (make_annotation(points=square).area_in_slide_pixels
                == make_annotation(points=list(reversed(square))).area_in_slide_pixels)

    def test_bounding_box(self):
        ann = make_annotation(points=[Point(10, 20), Point(110, 20), Point(60, 120)])
        assert ann.bounding_box == (10, 20, 100, 100)

    @pytest.mark.parametrize("x,y,expected", [
        (50, 50, True), (1, 1, True), (-1, 50, False), (150, 50, False), (50, 150, False),
    ])
    def test_point_in_polygon(self, x, y, expected):
        ann = make_annotation(points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)])
        assert ann.contains(x, y) is expected

    def test_concave_polygon(self):
        """An L-shape: the notch must read as outside."""
        ann = make_annotation(points=[
            Point(0, 0), Point(100, 0), Point(100, 40),
            Point(40, 40), Point(40, 100), Point(0, 100),
        ])
        assert ann.contains(20, 20) is True
        assert ann.contains(70, 70) is False
