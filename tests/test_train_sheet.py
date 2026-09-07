"""Train Model sheet: class selection, running, and results display."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from PySide6.QtCore import Qt

from pathlearn.data.bank import Patch, PatchBank
from pathlearn.models.annotation import AnnotationColor
from pathlearn.models.classification import Classification, ClassificationProfile
from pathlearn.ui.sheets.train_model import TrainModelSheet

PHIKON = "onnx:phikon-v1:r1"
UNI2 = "onnx:uni2-h:r1"
DIM = 10


def direction(index: int, scale: float = 6.0) -> np.ndarray:
    v = np.zeros(DIM)
    v[index % DIM] = scale
    return v


def patch(classification, vector, identity=PHIKON, **kw):
    defaults = dict(slide_path="E:/a.svs", slide_name="a.svs",
                    annotation_id=uuid.uuid4(), classification=classification,
                    patch_x=0, patch_y=0, patch_level=0, patch_size_level=224,
                    features=np.asarray(vector, dtype=np.float32),
                    extractor_identity=identity, white_fraction=0.1, nucleus_count=20)
    defaults.update(kw)
    return Patch(**defaults)


@pytest.fixture
def bank(tmp_path):
    with PatchBank(tmp_path / "b.db") as b:
        rng = np.random.default_rng(0)
        for i, name in enumerate(("Acinar", "Islets")):
            for _ in range(30):
                b.add(patch(name, direction(i) + rng.normal(0, 0.4, DIM)))
        for _ in range(12):
            b.add(patch("Lumen", direction(5) + rng.normal(0, 0.4, DIM)))
        yield b


@pytest.fixture
def profile():
    return ClassificationProfile(name="test", classes=[
        Classification("Acinar", AnnotationColor(80, 180, 80)),
        Classification("Islets", AnnotationColor(240, 180, 30)),
        Classification("Lumen", AnnotationColor(255, 255, 255), is_null=True),
    ])


@pytest.fixture
def sheet(qtbot, bank, profile):
    dialog = TrainModelSheet(bank, profile)
    qtbot.addWidget(dialog)
    yield dialog
    dialog.reject()


class TestInitialState:
    def test_lists_the_bank_feature_spaces(self, sheet):
        assert sheet.extractor_combo.count() == 1
        assert sheet.extractor_identity == PHIKON

    def test_lists_every_class(self, sheet):
        assert sheet.class_list.count() == 3

    def test_profile_null_class_is_preselected_as_null(self, sheet):
        """Lumen is flagged isNull in the profile, so it defaults to the null list."""
        assert sheet._checked(sheet.null_list) == ["Lumen"]
        assert "Lumen" not in [c for c in sheet._checked(sheet.class_list)
                               if c not in sheet._checked(sheet.null_list)]

    def test_train_is_enabled(self, sheet):
        assert sheet.train_button.isEnabled()

    def test_save_is_disabled_before_training(self, sheet):
        assert not sheet.save_button.isEnabled()


class TestSelection:
    def test_fewer_than_two_classes_disables_train(self, sheet):
        for i in range(sheet.class_list.count()):
            sheet.class_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        assert not sheet.train_button.isEnabled()
        assert "at least 2" in sheet.counts_label.text()

    def test_null_selection_excludes_from_classes(self, sheet):
        settings = sheet._settings()
        assert "Lumen" in settings.null_labels
        assert "Lumen" not in settings.class_labels

    def test_null_threshold_only_sent_when_nulls_chosen(self, sheet):
        assert sheet._settings().null_threshold is not None
        for i in range(sheet.null_list.count()):
            sheet.null_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        assert sheet._settings().null_threshold is None

    def test_white_filter_of_one_means_no_filter(self, sheet):
        sheet.max_white.setValue(1.0)
        assert sheet._settings().max_white_fraction is None
        sheet.max_white.setValue(0.5)
        assert sheet._settings().max_white_fraction == 0.5


class TestRun:
    def test_training_produces_a_model(self, sheet, qtbot):
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.model is not None
        assert sheet.model.class_labels == ["Acinar", "Islets"]
        assert sheet.save_button.isEnabled()

    def test_results_tables_are_filled(self, sheet, qtbot):
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.per_class_table.rowCount() == 2
        assert sheet.confusion_table.rowCount() == 2
        assert sheet.confusion_table.columnCount() == 2

    def test_switches_to_results_tab(self, sheet, qtbot):
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.tabs.currentIndex() == 2

    def test_null_reference_is_carried(self, sheet, qtbot):
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.model.uses_null_filter

    def test_saves_and_reloads(self, sheet, qtbot, tmp_path):
        from pathlearn.models.classifier import MLClassifier
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        path = tmp_path / "m.cl"
        sheet.model.save(path)
        assert MLClassifier.load(path).class_labels == sheet.model.class_labels

    def test_controls_re_enabled_after(self, sheet, qtbot):
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.class_list.isEnabled()
        assert not sheet.progress.isVisible()

    def test_failure_is_reported_not_crashed(self, sheet, qtbot, monkeypatch):
        """A single trainable class must surface as a message, not an exception."""
        shown: list[str] = []
        monkeypatch.setattr(sheet, "_on_failed",
                            lambda message: shown.append(message) or sheet._set_running(False))
        for i in range(sheet.class_list.count()):
            item = sheet.class_list.item(i)
            item.setCheckState(Qt.CheckState.Checked
                               if item.data(Qt.ItemDataRole.UserRole) == "Acinar"
                               else Qt.CheckState.Unchecked)
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert shown and "2 classes" in shown[0]


class TestEmptyBank:
    def test_empty_bank_disables_training(self, qtbot, tmp_path, profile):
        with PatchBank(tmp_path / "empty.db") as empty:
            dialog = TrainModelSheet(empty, profile)
            qtbot.addWidget(dialog)
            assert not dialog.train_button.isEnabled()
            assert "Extract Patches" in dialog.source_note.text()
            dialog.reject()
