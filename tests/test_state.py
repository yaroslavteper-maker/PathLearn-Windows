"""Saving and restoring a whole working state."""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pathlearn.core.panin import PANIN_DIMENSION, PANIN_VERSION
from pathlearn.data.bank import Patch, PatchBank
from pathlearn.data.geometry_bank import GeometryBank, GeometryRecord
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.classification import Classification, ClassificationProfile
from pathlearn.models.classifier import MLClassifier
from pathlearn.models.store import AnnotationStore
from pathlearn.state import (RESTORE_BACKUP_SUFFIX, STATE_SUFFIX, StateError,
                             _entry_name, load_state, read_manifest, save_state)


class _Settings:
    """Stand-in for QSettings so a test never touches the user's registry."""

    def __init__(self, *_args) -> None:
        self._values: dict = {}

    def value(self, key, default=None):
        return self._values.get(key, default)

    def setValue(self, key, value) -> None:
        self._values[key] = value


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    import pathlearn.data.bank as bank_module
    import pathlearn.data.geometry_bank as geometry_module
    import pathlearn.ui.main_window as main_window

    monkeypatch.setattr(main_window, "QSettings", _Settings)
    monkeypatch.setattr(bank_module, "default_bank_path",
                        lambda: tmp_path / "live" / "bank.db")
    monkeypatch.setattr(geometry_module, "default_geometry_bank_path",
                        lambda: tmp_path / "live" / "geometry_bank.json")
    (tmp_path / "live").mkdir(exist_ok=True)

    widget = main_window.MainWindow()
    qtbot.addWidget(widget)
    yield widget
    widget.close()


def a_profile(name="Test palette"):
    return ClassificationProfile(name=name, classes=[
        Classification("PanIN-1a", AnnotationColor(10, 20, 30)),
        Classification("PanIN-3", AnnotationColor(40, 50, 60), is_null=True),
    ])


def a_model(labels=("a", "b"), dim=4):
    rng = np.random.default_rng(0)
    return MLClassifier(class_labels=list(labels),
                        weights=rng.normal(0, 1, (dim, len(labels))).astype(np.float32),
                        biases=np.zeros(len(labels), dtype=np.float32),
                        extractor_identity="geometry:handcrafted:r2")


def a_slide_with_annotations(tmp_path, name="case.svs", count=3):
    """A stand-in slide file plus a real sidecar beside it."""
    slide = tmp_path / name
    slide.write_bytes(b"not really a slide")
    store = AnnotationStore()
    store.bind(slide, 4096)
    for i in range(count):
        store.add(Annotation(points=[Point(i * 50, 0), Point(i * 50 + 40, 0),
                                     Point(i * 50 + 40, 40)],
                             classification="PanIN-1a",
                             color=AnnotationColor.default()))
    return slide


def a_patch(slide="a.svs"):
    return Patch(slide_path=f"E:/{slide}", slide_name=slide,
                 annotation_id=uuid.uuid4(), classification="PanIN-2",
                 patch_x=0, patch_y=0, patch_level=0, patch_size_level=224,
                 features=np.zeros(8, dtype=np.float32),
                 extractor_identity="onnx:test:r1", white_fraction=0.1,
                 nucleus_count=12)


class TestSaving:
    def test_it_writes_a_zip_with_a_manifest(self, tmp_path):
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json())
        assert zipfile.is_zipfile(target)
        with zipfile.ZipFile(target) as archive:
            assert "manifest.json" in archive.namelist()
            assert "profile.json" in archive.namelist()

    def test_the_manifest_reads_back(self, tmp_path):
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(),
                   profile_name="Test palette", patch_count=1596,
                   geometry_count=41)
        manifest = read_manifest(target)
        assert manifest.profile_name == "Test palette"
        assert manifest.patch_count == 1596
        assert "1,596 patches" in manifest.summary()

    def test_banks_are_included(self, tmp_path):
        bank_path = tmp_path / "bank.db"
        with PatchBank(bank_path) as bank:
            bank.add(a_patch())
        geometry = GeometryBank(tmp_path / "geometry_bank.json")
        geometry.add([GeometryRecord(
            slide_path="S.svs", slide_name="S.svs", annotation_id=uuid.uuid4(),
            classification="PanIN-2",
            panin_features=np.zeros(PANIN_DIMENSION, dtype=np.float32),
            panin_version=PANIN_VERSION)])

        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(),
                   patch_bank_path=bank_path, geometry_bank_path=geometry.path)
        with zipfile.ZipFile(target) as archive:
            assert "bank.db" in archive.namelist()
            assert "geometry_bank.json" in archive.namelist()

    def test_models_are_embedded_not_referenced(self, tmp_path):
        """A state must not depend on a .cl file staying where it was."""
        target = tmp_path / f"s{STATE_SUFFIX}"
        manifest = save_state(target, profile_json=a_profile().to_json(),
                              models={"geometry": a_model().to_json()})
        assert manifest.models["geometry"] == "models/geometry.cl"
        with zipfile.ZipFile(target) as archive:
            document = archive.read("models/geometry.cl").decode("utf-8")
        assert MLClassifier.from_json(document).class_labels == ["a", "b"]

    def test_annotations_are_copied_for_each_slide(self, tmp_path):
        slide = a_slide_with_annotations(tmp_path, count=3)
        target = tmp_path / f"s{STATE_SUFFIX}"
        manifest = save_state(target, profile_json=a_profile().to_json(),
                              slide_paths=[slide])
        assert len(manifest.slides) == 1
        assert manifest.slides[0].annotation_count == 3
        assert manifest.slides[0].slide_path == str(slide)

    def test_a_slide_without_a_sidecar_is_skipped(self, tmp_path):
        bare = tmp_path / "bare.svs"
        bare.write_bytes(b"x")
        manifest = save_state(tmp_path / f"s{STATE_SUFFIX}",
                              profile_json=a_profile().to_json(),
                              slide_paths=[bare])
        assert manifest.slides == []

    def test_the_same_slide_twice_is_stored_once(self, tmp_path):
        slide = a_slide_with_annotations(tmp_path)
        manifest = save_state(tmp_path / f"s{STATE_SUFFIX}",
                              profile_json=a_profile().to_json(),
                              slide_paths=[slide, slide])
        assert len(manifest.slides) == 1

    def test_slides_sharing_a_filename_do_not_collide(self, tmp_path):
        """Two folders, one filename — a sanitised name would overwrite one."""
        first = a_slide_with_annotations(tmp_path / "a", count=1) \
            if (tmp_path / "a").mkdir() or True else None
        (tmp_path / "b").mkdir()
        second = a_slide_with_annotations(tmp_path / "b", count=2)
        assert _entry_name(str(first)) != _entry_name(str(second))

        manifest = save_state(tmp_path / f"s{STATE_SUFFIX}",
                              profile_json=a_profile().to_json(),
                              slide_paths=[first, second])
        assert len({s.entry for s in manifest.slides}) == 2


class TestRestoring:
    def test_the_profile_comes_back(self, tmp_path):
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json())
        report = load_state(target)
        restored = ClassificationProfile.from_json(report.profile_json)
        assert [c.name for c in restored.classes] == ["PanIN-1a", "PanIN-3"]
        assert restored.classes[1].is_null

    def test_the_patch_bank_comes_back(self, tmp_path):
        source = tmp_path / "bank.db"
        with PatchBank(source) as bank:
            bank.add_many([a_patch() for _ in range(5)])
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(),
                   patch_bank_path=source)

        restored_path = tmp_path / "restored" / "bank.db"
        load_state(target, patch_bank_path=restored_path)
        with PatchBank(restored_path) as bank:
            assert bank.stats().total == 5

    def test_the_geometry_bank_comes_back(self, tmp_path):
        geometry = GeometryBank(tmp_path / "geometry_bank.json")
        geometry.add([GeometryRecord(
            slide_path="S.svs", slide_name="S.svs", annotation_id=uuid.uuid4(),
            classification="PanIN-2",
            panin_features=np.ones(PANIN_DIMENSION, dtype=np.float32),
            panin_version=PANIN_VERSION)])
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(),
                   geometry_bank_path=geometry.path)

        restored_path = tmp_path / "restored" / "geometry_bank.json"
        load_state(target, geometry_bank_path=restored_path)
        assert len(GeometryBank(restored_path)) == 1

    def test_models_come_back(self, tmp_path):
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(),
                   models={"geometry": a_model(("PanIN-1a", "PanIN-2")).to_json()})
        report = load_state(target)
        model = MLClassifier.from_json(report.models["geometry"])
        assert model.class_labels == ["PanIN-1a", "PanIN-2"]

    def test_annotations_are_written_back(self, tmp_path):
        slide = a_slide_with_annotations(tmp_path, count=3)
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(), slide_paths=[slide])

        slide.with_suffix(".geojson").unlink()
        report = load_state(target)
        assert report.restored_slides == [str(slide)]
        reloaded = AnnotationStore()
        reloaded.bind(slide, 4096)
        assert len(reloaded.annotations) == 3

    def test_an_existing_sidecar_is_backed_up_before_being_overwritten(self, tmp_path):
        slide = a_slide_with_annotations(tmp_path, count=3)
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(), slide_paths=[slide])

        # Replace the live annotations with something different.
        store = AnnotationStore()
        store.bind(slide, 4096)
        store.remove_all()
        store.add(Annotation(points=[Point(0, 0), Point(9, 0), Point(9, 9)],
                             classification="Other",
                             color=AnnotationColor.default()))

        report = load_state(target)
        assert report.backed_up, "the previous sidecar was not backed up"
        backup = Path(report.backed_up[0])
        assert backup.is_file()
        assert RESTORE_BACKUP_SUFFIX in backup.name
        assert "Other" in backup.read_text(encoding="utf-8")

    def test_a_missing_slide_is_reported_not_skipped_silently(self, tmp_path):
        slide = a_slide_with_annotations(tmp_path)
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(), slide_paths=[slide])
        slide.unlink()
        slide.with_suffix(".geojson").unlink()

        report = load_state(target)
        assert report.missing_slides == [str(slide)]
        assert report.restored_slides == []
        assert "not found" in report.summary()

    def test_annotations_can_be_left_alone(self, tmp_path):
        slide = a_slide_with_annotations(tmp_path, count=3)
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(), slide_paths=[slide])
        report = load_state(target, restore_annotations=False)
        assert report.restored_slides == []

    def test_stale_sqlite_sidecars_are_removed(self, tmp_path):
        """A -wal beside a replaced .db makes SQLite read a mixture."""
        source = tmp_path / "bank.db"
        with PatchBank(source) as bank:
            bank.add(a_patch())
        target = tmp_path / f"s{STATE_SUFFIX}"
        save_state(target, profile_json=a_profile().to_json(),
                   patch_bank_path=source)

        destination = tmp_path / "live" / "bank.db"
        destination.parent.mkdir()
        destination.write_bytes(b"old")
        stale = destination.with_name("bank.db-wal")
        stale.write_bytes(b"stale")
        load_state(target, patch_bank_path=destination)
        assert not stale.exists()


class TestRefusals:
    def test_a_non_zip_is_refused(self, tmp_path):
        bogus = tmp_path / f"x{STATE_SUFFIX}"
        bogus.write_text("not a zip", encoding="utf-8")
        with pytest.raises(StateError, match="not a PathLearn state"):
            read_manifest(bogus)

    def test_a_zip_without_a_manifest_is_refused(self, tmp_path):
        bogus = tmp_path / f"x{STATE_SUFFIX}"
        with zipfile.ZipFile(bogus, "w") as archive:
            archive.writestr("hello.txt", "hi")
        with pytest.raises(StateError, match="not a PathLearn state"):
            read_manifest(bogus)

    def test_a_newer_state_version_is_refused_clearly(self, tmp_path):
        target = tmp_path / f"x{STATE_SUFFIX}"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"version": 99}))
        with pytest.raises(StateError, match="newer PathLearn"):
            read_manifest(target)


class TestThroughTheWindow:
    """The whole round trip, driven through MainWindow."""

    def test_saving_then_restoring_recovers_everything(self, window, tmp_path):
        from pathlearn.data.geometry_bank import FeatureSource

        window.profile = a_profile("Saved palette")
        window.sidebar.reload_profile(window.profile)
        window.bank.add_many([a_patch() for _ in range(4)])
        window.geometry_bank.add([GeometryRecord(
            slide_path="S.svs", slide_name="S.svs", annotation_id=uuid.uuid4(),
            classification="PanIN-2",
            panin_features=np.zeros(PANIN_DIMENSION, dtype=np.float32),
            panin_version=PANIN_VERSION)])
        window.classifier = a_model(("x", "y"), dim=768)
        window.geometry_panel.min_nuclei.setValue(37)

        target = tmp_path / f"state{STATE_SUFFIX}"
        manifest = window.write_state(target)
        assert manifest.patch_count == 4
        assert manifest.settings["minNuclei"] == 37

        # Wreck the live state, then restore it.
        window.bank.clear()
        window.geometry_bank.clear()
        window.profile = ClassificationProfile.default()
        window.classifier = None
        window.geometry_panel.min_nuclei.setValue(5)

        window.apply_state(target)
        assert window.bank.stats().total == 4
        assert len(window.geometry_bank) == 1
        assert window.profile.name == "Saved palette"
        assert window.classifier is not None
        assert window.geometry_panel.min_nuclei.value() == 37

    def test_the_bank_is_usable_after_a_restore(self, window, tmp_path):
        """Its SQLite file is replaced underneath it, so it must reconnect."""
        window.bank.add_many([a_patch() for _ in range(3)])
        target = tmp_path / f"state{STATE_SUFFIX}"
        window.write_state(target)
        window.apply_state(target)
        assert window.bank.is_open
        window.bank.add(a_patch())          # would raise on a closed bank
        assert window.bank.stats().total == 4

    def test_annotations_round_trip_through_the_window(self, window, tmp_path):
        slide = a_slide_with_annotations(tmp_path, count=2)
        window._note_slide(slide)
        target = tmp_path / f"state{STATE_SUFFIX}"
        manifest = window.write_state(target)
        assert [s.annotation_count for s in manifest.slides] == [2]

        slide.with_suffix(".geojson").unlink()
        window.apply_state(target)
        assert slide.with_suffix(".geojson").is_file()

    def test_the_palette_survives_a_restart_without_a_profile_file(self, window):
        """What the Profile menu really provided, kept without the menu."""
        window.profile = a_profile("Remembered")
        window._remember_profile()
        assert "Remembered" in window.settings.value("profileJson", "")
        assert window._load_current_profile().name == "Remembered"

    def test_there_is_no_profile_menu(self, window):
        titles = {m.title() for m in window.menuBar().findChildren(type(
            window.menuBar().addMenu("tmp")))}
        assert "&Profile" not in titles
        assert not hasattr(window, "action_load_profile")
        assert not hasattr(window, "action_save_profile")

    def test_state_actions_live_in_the_file_menu(self, window):
        file_menu = next(m for m in window.menuBar().findChildren(type(
            window.menuBar().addMenu("tmp"))) if m.title() == "&File")
        texts = [a.text() for a in file_menu.actions()]
        assert any("Save &State" in t for t in texts)
        assert any("Open State" in t for t in texts)
