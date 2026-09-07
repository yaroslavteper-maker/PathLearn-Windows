"""Annotation store: sidecar persistence and the one-time legacy migration."""

from __future__ import annotations

import json

import pytest

from pathlearn.coords import MARKER_KEY, CoordinateSpace, has_marker
from pathlearn.io import geojson
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.store import AnnotationStore, SidecarError

SLIDE_HEIGHT = 1000.0


@pytest.fixture
def slide_path(tmp_path):
    p = tmp_path / "specimen.svs"
    p.write_bytes(b"not a real slide")  # the store never opens it
    return p


def make_annotation(**kwargs) -> Annotation:
    defaults = dict(
        points=[Point(10, 20), Point(110, 20), Point(110, 120)],
        classification="PaNIN-2",
        color=AnnotationColor(240, 180, 30),
    )
    defaults.update(kwargs)
    return Annotation(**defaults)


def write_legacy_sidecar(slide_path, annotations) -> None:
    """Write an unmarked sidecar, exactly as the macOS build would."""
    doc = json.loads(geojson.encode(annotations))
    del doc[MARKER_KEY]
    slide_path.with_suffix(".geojson").write_text(json.dumps(doc), encoding="utf-8")


class TestBinding:
    def test_sidecar_path_replaces_slide_extension(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        assert store.sidecar_path == slide_path.with_suffix(".geojson")

    def test_missing_sidecar_is_not_an_error(self, slide_path):
        report = AnnotationStore().bind(slide_path, SLIDE_HEIGHT)
        assert report.count == 0 and report.migrated is False

    def test_custom_suffix_avoids_collision(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT, sidecar_suffix="predicted.geojson")
        assert store.sidecar_path.name == "specimen.predicted.geojson"


class TestPersistence:
    def test_add_writes_sidecar_immediately(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        store.add(make_annotation())

        assert store.sidecar_path.exists()
        reloaded = AnnotationStore()
        reloaded.bind(slide_path, SLIDE_HEIGHT)
        assert len(reloaded.annotations) == 1

    def test_written_sidecar_is_marked_upright(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        store.add(make_annotation())
        assert has_marker(json.loads(store.sidecar_path.read_text(encoding="utf-8")))

    def test_reload_is_not_migrated_again(self, slide_path):
        """Our own marked file must survive reload with coordinates unchanged."""
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        store.add(make_annotation())
        points = list(store.annotations[0].points)

        reloaded = AnnotationStore()
        report = reloaded.bind(slide_path, SLIDE_HEIGHT)
        assert report.migrated is False
        assert reloaded.annotations[0].points == points

    def test_remove_and_clear_persist(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        ann = make_annotation()
        store.add(ann)
        store.remove(ann.id)

        reloaded = AnnotationStore()
        reloaded.bind(slide_path, SLIDE_HEIGHT)
        assert reloaded.annotations == []

    def test_no_temp_files_are_left_behind(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        for _ in range(5):
            store.add(make_annotation())
        assert [p.name for p in slide_path.parent.glob("*.tmp")] == []


class TestLegacyMigration:
    def test_legacy_sidecar_is_mirrored_on_load(self, slide_path):
        """y=20 in a macOS sidecar must land at y=980 with slideH=1000."""
        write_legacy_sidecar(slide_path, [make_annotation()])

        store = AnnotationStore()
        report = store.bind(slide_path, SLIDE_HEIGHT)

        assert report.space is CoordinateSpace.LEGACY_MIRRORED
        assert report.migrated is True
        assert store.annotations[0].points == [
            Point(10, 980), Point(110, 980), Point(110, 880),
        ]

    def test_original_is_backed_up(self, slide_path):
        write_legacy_sidecar(slide_path, [make_annotation()])
        report = AnnotationStore().bind(slide_path, SLIDE_HEIGHT)

        assert report.backup_path is not None and report.backup_path.exists()
        backup = json.loads(report.backup_path.read_text(encoding="utf-8"))
        # The backup keeps the original, un-mirrored-on-disk coordinates.
        assert backup["features"][0]["geometry"]["coordinates"][0][0] == [10, 20]

    def test_migration_happens_exactly_once(self, slide_path):
        write_legacy_sidecar(slide_path, [make_annotation()])
        first = AnnotationStore()
        first.bind(slide_path, SLIDE_HEIGHT)
        migrated_points = list(first.annotations[0].points)

        second = AnnotationStore()
        report = second.bind(slide_path, SLIDE_HEIGHT)
        assert report.migrated is False
        assert second.annotations[0].points == migrated_points

    def test_migration_preserves_metadata(self, slide_path):
        write_legacy_sidecar(slide_path, [
            make_annotation(name="lesion", is_subtractive=True, is_visible=False),
        ])
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        ann = store.annotations[0]
        assert (ann.name, ann.is_subtractive, ann.is_visible) == ("lesion", True, False)

    def test_empty_legacy_sidecar_is_not_migrated(self, slide_path):
        write_legacy_sidecar(slide_path, [])
        report = AnnotationStore().bind(slide_path, SLIDE_HEIGHT)
        assert report.migrated is False and report.backup_path is None

    def test_corrupt_sidecar_raises(self, slide_path):
        slide_path.with_suffix(".geojson").write_text("{ not json", encoding="utf-8")
        with pytest.raises(SidecarError):
            AnnotationStore().bind(slide_path, SLIDE_HEIGHT)


class TestQueries:
    def test_hit_test_prefers_topmost(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        square = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        store.add(make_annotation(points=square, classification="under"))
        store.add(make_annotation(points=square, classification="over"))
        assert store.hit_test(50, 50).classification == "over"

    def test_hit_test_does_not_consult_the_legacy_visible_flag(self, slide_path):
        """``isVisible`` is kept for file fidelity but no longer does anything.

        Every annotation is painted, so every annotation must be clickable —
        gating hit-testing on a flag with no UI behind it would make regions
        from a macOS sidecar silently unselectable.
        """
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        square = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        store.add(make_annotation(points=square, is_visible=False))
        assert store.hit_test(50, 50) is not None

    def test_import_merge_append_regenerates_ids(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        ann = make_annotation()
        store.add(ann)
        store.import_merge([ann], replace=False)
        assert len({a.id for a in store.annotations}) == 2

    def test_import_merge_replace(self, slide_path):
        store = AnnotationStore()
        store.bind(slide_path, SLIDE_HEIGHT)
        store.add(make_annotation(classification="old"))
        store.import_merge([make_annotation(classification="new")], replace=True)
        assert [a.classification for a in store.annotations] == ["new"]

    def test_on_change_fires(self, slide_path):
        calls = []
        store = AnnotationStore(on_change=lambda: calls.append(1))
        store.bind(slide_path, SLIDE_HEIGHT)
        store.add(make_annotation())
        assert calls
