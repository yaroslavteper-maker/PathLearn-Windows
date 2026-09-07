"""The rename-class sheet and the window that carries the rename out."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from pathlearn.data.bank import Patch
from pathlearn.data.geometry_bank import GeometryRecord
from pathlearn.io import geojson
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.ui.sheets.rename_class import RenameClassSheet
from synthetic_slide import write_synthetic_slide


class _Settings:
    """Stand-in for QSettings so a test never touches the user's registry."""

    def __init__(self, *_args) -> None:
        self._values: dict = {}

    def value(self, key, default=None):
        return self._values.get(key, default)

    def setValue(self, key, value) -> None:
        self._values[key] = value


def box(x=0, label="PanINN"):
    return Annotation(points=[Point(x, 0), Point(x + 10, 0), Point(x + 10, 10)],
                      classification=label, color=AnnotationColor(1, 2, 3))


def a_patch(label="PanINN"):
    return Patch(slide_path="E:/a.svs", slide_name="a.svs",
                 annotation_id=uuid.uuid4(), classification=label,
                 patch_x=0, patch_y=0, patch_level=0, patch_size_level=224,
                 features=np.zeros(8, dtype=np.float32),
                 extractor_identity="onnx:test:r1", white_fraction=0.1,
                 nucleus_count=12)


def a_record(label="PanINN"):
    return GeometryRecord(slide_path="E:/a.svs", slide_name="a.svs",
                          annotation_id=uuid.uuid4(), classification=label,
                          shape_features=np.arange(12, dtype=np.float32))


@pytest.fixture(scope="session")
def slide_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("slides") / "case-01.tiff"
    write_synthetic_slide(path, width=2048, height=1536)
    return path


@pytest.fixture
def slide(slide_file, tmp_path):
    import shutil

    target = tmp_path / slide_file.name
    shutil.copy(slide_file, target)
    return target


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


def accept_rename(monkeypatch, new_name, **flags):
    """Make the sheet answer without showing itself."""
    def fake_exec(self):
        self.name_edit.setText(new_name)
        for attribute, value in flags.items():
            getattr(self, attribute).setChecked(value)
        return 1

    monkeypatch.setattr(RenameClassSheet, "exec", fake_exec)


class TestTheSheet:
    def sheet(self, qtbot, **kw):
        kw.setdefault("existing_names", ["PanINN", "Acinar"])
        widget = RenameClassSheet("PanINN", **kw)
        qtbot.addWidget(widget)
        return widget

    def test_it_starts_on_the_current_name_and_refuses_it(self, qtbot):
        sheet = self.sheet(qtbot)
        assert sheet.name_edit.text() == "PanINN"
        assert "current name" in sheet.warning.text()

    def test_an_empty_name_is_refused(self, qtbot):
        sheet = self.sheet(qtbot)
        sheet.name_edit.setText("   ")
        assert "Give the class a name" in sheet.warning.text()

    def test_a_fresh_name_is_accepted(self, qtbot):
        sheet = self.sheet(qtbot)
        sheet.name_edit.setText("PanIN")
        assert sheet.new_name == "PanIN"
        assert not sheet.is_merge

    def test_renaming_onto_an_existing_class_warns_that_it_merges(self, qtbot):
        """A merge is a real operation; it should not be discovered after."""
        sheet = self.sheet(qtbot)
        sheet.name_edit.setText("Acinar")
        assert sheet.is_merge
        assert "MERGES" in sheet.warning.text()

    def test_the_counts_are_shown_before_anything_happens(self, qtbot):
        sheet = self.sheet(qtbot, patches=1596, geometry_records=41)
        assert "1,596" in sheet.also_patches.text()
        assert "41" in sheet.also_geometry.text()

    def test_an_empty_bank_offers_nothing_to_tick(self, qtbot):
        sheet = self.sheet(qtbot, patches=0, geometry_records=0)
        assert not sheet.also_patches.isEnabled()
        assert not sheet.update_patches

    def test_other_slides_are_counted_from_their_sidecars(self, qtbot,
                                                          tmp_path):
        first = tmp_path / "a.svs"
        (tmp_path / "a.geojson").write_text(
            geojson.encode([box(0), box(20), box(40, label="Acinar")]),
            encoding="utf-8")
        second = tmp_path / "b.svs"
        (tmp_path / "b.geojson").write_text(
            geojson.encode([box(0)]), encoding="utf-8")
        sheet = self.sheet(qtbot, other_slides=[first, second])
        assert sheet.other_slide_total == 3
        assert len(sheet.sidecars_to_update()) == 2

    def test_a_slide_without_the_class_is_not_offered(self, qtbot, tmp_path):
        path = tmp_path / "a.svs"
        (tmp_path / "a.geojson").write_text(
            geojson.encode([box(0, label="Acinar")]), encoding="utf-8")
        sheet = self.sheet(qtbot, other_slides=[path])
        assert sheet.other_slide_total == 0
        assert not sheet.also_slides.isEnabled()

    def test_a_missing_sidecar_is_skipped(self, qtbot, tmp_path):
        sheet = self.sheet(qtbot, other_slides=[tmp_path / "gone.svs"])
        assert sheet.other_slide_total == 0

    def test_it_says_unopened_slides_keep_the_old_name(self, qtbot):
        sheet = self.sheet(qtbot)
        sheet.name_edit.setText("PanIN")
        assert "never opened" in sheet.warning.text()


class TestTheWindowCarriesItOut:
    def setup_slide(self, window, slide, label="PanINN"):
        window.open_slide(slide)
        window.store.add_batch([box(0, label), box(50, label),
                                box(100, "Acinar")])
        window.profile.classes.append(
            type(window.profile.classes[0])(name=label,
                                            color=AnnotationColor(1, 2, 3)))
        window.sidebar.reload_profile(window.profile)
        window.sidebar.refresh()

    def test_it_renames_the_annotations_and_the_palette(self, window, slide,
                                                        monkeypatch):
        self.setup_slide(window, slide)
        accept_rename(monkeypatch, "PanIN")
        message = window.rename_class("PanINN")
        assert "Renamed" in message
        assert [a.classification for a in window.store.annotations] == \
            ["PanIN", "PanIN", "Acinar"]
        assert "PanIN" in [c.name for c in window.profile.classes]
        assert "PanINN" not in [c.name for c in window.profile.classes]

    def test_cancelling_changes_nothing(self, window, slide, monkeypatch):
        self.setup_slide(window, slide)
        monkeypatch.setattr(RenameClassSheet, "exec", lambda self: 0)
        assert "cancelled" in window.rename_class("PanINN")
        assert window.store.annotations[0].classification == "PanINN"

    def test_it_relabels_the_patch_bank_when_asked(self, window, slide,
                                                   monkeypatch):
        self.setup_slide(window, slide)
        window.bank.add(a_patch())
        window.bank.add(a_patch())
        accept_rename(monkeypatch, "PanIN")
        message = window.rename_class("PanINN")
        assert "2 patch(es)" in message
        assert window.bank.count(classifications=["PanIN"]) == 2

    def test_the_bank_can_be_left_alone(self, window, slide, monkeypatch):
        self.setup_slide(window, slide)
        window.bank.add(a_patch())
        accept_rename(monkeypatch, "PanIN", also_patches=False)
        window.rename_class("PanINN")
        assert window.bank.count(classifications=["PanINN"]) == 1

    def test_it_relabels_the_geometry_bank(self, window, slide, monkeypatch):
        self.setup_slide(window, slide)
        window.geometry_bank.add([a_record(), a_record()])
        accept_rename(monkeypatch, "PanIN")
        assert "2 geometry record(s)" in window.rename_class("PanINN")

    def test_it_updates_other_slides_when_asked(self, window, slide,
                                                monkeypatch, tmp_path):
        self.setup_slide(window, slide)
        other = tmp_path / "other.svs"
        other.write_bytes(b"x")
        sidecar = tmp_path / "other.geojson"
        sidecar.write_text(geojson.encode([box(0), box(20)]), encoding="utf-8")
        window.settings.setValue("knownSlides", [str(other)])

        accept_rename(monkeypatch, "PanIN")
        message = window.rename_class("PanINN")
        assert "2 annotation(s) on 1 other slide(s)" in message
        labels = [a.classification
                  for a in geojson.decode(sidecar.read_text(encoding="utf-8"))]
        assert labels == ["PanIN", "PanIN"]

    def test_the_open_slide_is_not_double_counted(self, window, slide,
                                                  monkeypatch):
        """It is in knownSlides too, and is handled by the store already."""
        self.setup_slide(window, slide)
        window.settings.setValue("knownSlides", [str(slide)])
        accept_rename(monkeypatch, "PanIN")
        message = window.rename_class("PanINN")
        assert "other slide(s)" not in message
        assert "2 annotation(s) on this slide" in message

    def test_an_unreadable_sidecar_does_not_abandon_the_rest(
            self, window, slide, monkeypatch, tmp_path):
        self.setup_slide(window, slide)
        good = tmp_path / "good.svs"
        (tmp_path / "good.geojson").write_text(geojson.encode([box(0)]),
                                               encoding="utf-8")
        window.settings.setValue("knownSlides", [str(good)])
        accept_rename(monkeypatch, "PanIN")

        import pathlearn.ui.main_window as main_window
        real = main_window.rename_class_in_sidecar

        def explode(path, old, new):
            raise OSError("locked by another process")

        monkeypatch.setattr(main_window, "rename_class_in_sidecar", explode)
        message = window.rename_class("PanINN")
        # The slide in front of the user still got renamed.
        assert window.store.annotations[0].classification == "PanIN"
        assert "could not be updated" in message
        monkeypatch.setattr(main_window, "rename_class_in_sidecar", real)

    def test_a_merge_collapses_the_palette_entry(self, window, slide,
                                                 monkeypatch):
        self.setup_slide(window, slide)
        # Onto a class the palette really has, rather than assuming one.
        target = window.profile.classes[0].name
        assert target != "PanINN"
        accept_rename(monkeypatch, target)
        message = window.rename_class("PanINN")
        assert "Merged" in message
        names = [c.name for c in window.profile.classes]
        assert names.count(target) == 1
        assert "PanINN" not in names

    def test_renaming_an_unused_class_says_so(self, window, slide,
                                              monkeypatch):
        window.open_slide(slide)
        accept_rename(monkeypatch, "Brand new")
        assert "nothing was using it yet" in window.rename_class("Unused")

    def test_the_sidebar_button_asks_the_window(self, window, slide, qtbot,
                                                monkeypatch):
        """The sidebar owns the palette but not the banks or other slides."""
        # The signal is wired to the window, which opens the sheet for real;
        # without this the click blocks the test on a modal dialog.
        monkeypatch.setattr(RenameClassSheet, "exec", lambda self: 0)
        self.setup_slide(window, slide)
        window.sidebar.class_combo.setCurrentIndex(
            window.sidebar.class_combo.findData("PanINN"))
        with qtbot.waitSignal(window.sidebar.rename_class_requested,
                              timeout=500) as caught:
            window.sidebar.rename_class_button.click()
        assert caught.args[0] == "PanINN"
