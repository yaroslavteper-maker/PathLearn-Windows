"""Renaming a class everywhere it is stamped, not just in the palette."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from pathlearn.coords import MARKER_KEY
from pathlearn.data.bank import Patch, PatchBank
from pathlearn.data.geometry_bank import GeometryBank, GeometryRecord
from pathlearn.io import geojson
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.store import (AnnotationStore, count_class_in_sidecar,
                                    rename_class_in_sidecar)


def box(x=0, label="PanINN"):
    return Annotation(points=[Point(x, 0), Point(x + 10, 0), Point(x + 10, 10)],
                      classification=label, color=AnnotationColor(1, 2, 3))


def a_patch(label="PanINN", slide="a.svs"):
    return Patch(slide_path=f"E:/{slide}", slide_name=slide,
                 annotation_id=uuid.uuid4(), classification=label,
                 patch_x=0, patch_y=0, patch_level=0, patch_size_level=224,
                 features=np.zeros(8, dtype=np.float32),
                 extractor_identity="onnx:test:r1", white_fraction=0.1,
                 nucleus_count=12)


def a_record(label="PanINN"):
    # Needs a descriptor block: GeometryRecord.from_dict drops a record that
    # carries no features, so a bare one would vanish on reload and the
    # persistence test would pass for the wrong reason.
    return GeometryRecord(slide_path="E:/a.svs", slide_name="a.svs",
                          annotation_id=uuid.uuid4(), classification=label,
                          shape_features=np.arange(12, dtype=np.float32))


class TestTheStore:
    def test_it_renames_every_matching_annotation(self, tmp_path):
        store = AnnotationStore()
        store.bind(tmp_path / "a.svs", 1000)
        store.add_batch([box(0), box(50), box(100, label="Acinar")])
        assert store.rename_class("PanINN", "PanIN") == 2
        assert [a.classification for a in store.annotations] == \
            ["PanIN", "PanIN", "Acinar"]

    def test_the_drawing_class_follows(self, tmp_path):
        """Otherwise the next region you trace resurrects the old name."""
        store = AnnotationStore()
        store.bind(tmp_path / "a.svs", 1000)
        store.current_label = "PanINN"
        store.rename_class("PanINN", "PanIN")
        assert store.current_label == "PanIN"

    def test_it_saves_the_sidecar(self, tmp_path):
        store = AnnotationStore()
        store.bind(tmp_path / "a.svs", 1000)
        store.add(box(0))
        store.rename_class("PanINN", "PanIN")
        text = (tmp_path / "a.geojson").read_text(encoding="utf-8")
        assert "PanIN" in text and "PanINN" not in text

    def test_renaming_something_absent_changes_nothing(self, tmp_path):
        store = AnnotationStore()
        store.bind(tmp_path / "a.svs", 1000)
        store.add(box(0))
        assert store.rename_class("Nope", "Other") == 0
        assert store.annotations[0].classification == "PanINN"


class TestTheBanks:
    def test_the_patch_bank_relabels_rather_than_deletes(self, tmp_path):
        """Those rows carry features that cost real time to produce."""
        with PatchBank(tmp_path / "bank.db") as bank:
            bank.add(a_patch())
            bank.add(a_patch())
            bank.add(a_patch(label="Acinar"))
            assert bank.rename_class("PanINN", "PanIN") == 2
            assert bank.count(classifications=["PanIN"]) == 2
            assert bank.count(classifications=["PanINN"]) == 0
            assert bank.count(classifications=["Acinar"]) == 1

    def test_renaming_into_an_existing_class_merges(self, tmp_path):
        with PatchBank(tmp_path / "bank.db") as bank:
            bank.add(a_patch(label="PanINN"))
            bank.add(a_patch(label="PanIN"))
            bank.rename_class("PanINN", "PanIN")
            assert bank.count(classifications=["PanIN"]) == 2

    def test_the_patch_bank_ignores_a_no_op(self, tmp_path):
        with PatchBank(tmp_path / "bank.db") as bank:
            bank.add(a_patch())
            assert bank.rename_class("PanINN", "PanINN") == 0
            assert bank.rename_class("", "PanIN") == 0

    def test_the_geometry_bank_relabels(self, tmp_path):
        bank = GeometryBank(tmp_path / "geometry.json")
        bank.add([a_record(), a_record(), a_record(label="Acinar")])
        assert bank.rename_class("PanINN", "PanIN") == 2
        assert {r.classification for r in bank.records} == {"PanIN", "Acinar"}

    def test_the_geometry_bank_persists_the_rename(self, tmp_path):
        path = tmp_path / "geometry.json"
        bank = GeometryBank(path)
        bank.add([a_record()])
        bank.rename_class("PanINN", "PanIN")
        assert GeometryBank(path).records[0].classification == "PanIN"

    def test_the_geometry_bank_keeps_the_features(self, tmp_path):
        """A rename is a relabel, not a re-describe."""
        bank = GeometryBank(tmp_path / "geometry.json")
        record = GeometryRecord(
            slide_path="E:/a.svs", slide_name="a.svs",
            annotation_id=uuid.uuid4(), classification="PanINN",
            shape_features=np.arange(12, dtype=np.float32))
        bank.add([record])
        bank.rename_class("PanINN", "PanIN")
        assert np.array_equal(bank.records[0].shape_features,
                              np.arange(12, dtype=np.float32))


class TestOtherSlidesSidecars:
    def a_sidecar(self, tmp_path, stamped=True, labels=("PanINN", "Acinar")):
        path = tmp_path / "other.geojson"
        text = geojson.encode([box(i * 20, label=l)
                               for i, l in enumerate(labels)],
                              stamp_marker=stamped)
        path.write_text(text, encoding="utf-8")
        return path

    def test_it_counts_before_it_changes_anything(self, tmp_path):
        path = self.a_sidecar(tmp_path, labels=("PanINN", "PanINN", "Acinar"))
        assert count_class_in_sidecar(path, "PanINN") == 2
        assert count_class_in_sidecar(path, "Nope") == 0

    def test_counting_an_unreadable_file_is_zero_not_a_crash(self, tmp_path):
        path = tmp_path / "junk.geojson"
        path.write_text("not json", encoding="utf-8")
        assert count_class_in_sidecar(path, "PanINN") == 0

    def test_it_renames_in_place(self, tmp_path):
        path = self.a_sidecar(tmp_path)
        assert rename_class_in_sidecar(path, "PanINN", "PanIN") == 1
        labels = [a.classification
                  for a in geojson.decode(path.read_text(encoding="utf-8"))]
        assert labels == ["PanIN", "Acinar"]

    def test_a_stamped_file_stays_stamped(self, tmp_path):
        path = self.a_sidecar(tmp_path, stamped=True)
        rename_class_in_sidecar(path, "PanINN", "PanIN")
        document = json.loads(path.read_text(encoding="utf-8"))
        assert MARKER_KEY in document

    def test_an_unstamped_file_stays_unstamped(self, tmp_path):
        """The absence of the marker is what triggers the macOS Y-flip.

        Stamping it here would claim PathLearn wrote the file, the flip would
        never run, and every annotation on that slide would stay mirrored.
        """
        path = self.a_sidecar(tmp_path, stamped=False)
        rename_class_in_sidecar(path, "PanINN", "PanIN")
        document = json.loads(path.read_text(encoding="utf-8"))
        assert MARKER_KEY not in document

    def test_geometry_is_not_disturbed_by_the_round_trip(self, tmp_path):
        path = self.a_sidecar(tmp_path)
        before = [[(p.x, p.y) for p in a.points]
                  for a in geojson.decode(path.read_text(encoding="utf-8"))]
        rename_class_in_sidecar(path, "PanINN", "PanIN")
        after = [[(p.x, p.y) for p in a.points]
                 for a in geojson.decode(path.read_text(encoding="utf-8"))]
        assert before == after

    def test_nothing_is_written_when_nothing_matches(self, tmp_path):
        path = self.a_sidecar(tmp_path)
        stamp = path.stat().st_mtime_ns
        assert rename_class_in_sidecar(path, "Nope", "Other") == 0
        assert path.stat().st_mtime_ns == stamp
