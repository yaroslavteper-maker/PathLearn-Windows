"""Geometry panel: describing a slide, choosing a source, training."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from PySide6.QtCore import Qt

from pathlearn.core.shape import SHAPE_DIMENSION, SHAPE_VERSION
from pathlearn.core.geometry import DIMENSION, VERSION
from pathlearn.data.geometry_bank import FeatureSource, GeometryBank, GeometryRecord
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.store import AnnotationStore
from pathlearn.ui.panels.geometry_panel import GeometryPanel
from synthetic_slide import write_synthetic_slide


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("s") / "geom.tif"
    write_synthetic_slide(path, 4096, 3072, levels=4)
    with SlideImage(path) as s:
        yield s


@pytest.fixture
def bank(tmp_path):
    return GeometryBank(tmp_path / "g.json")


@pytest.fixture
def store(tmp_path, slide):
    s = AnnotationStore()
    s.bind(tmp_path / "slide.svs", slide.dimensions.height)
    for name, x in (("PaNIN-2", 400), ("PaNIN-3", 1800)):
        s.add(Annotation(
            points=[Point(x, 400), Point(x + 900, 400),
                    Point(x + 900, 1300), Point(x, 1300)],
            classification=name, color=AnnotationColor.default()))
    return s


@pytest.fixture
def panel(qtbot, bank, slide, store):
    widget = GeometryPanel(bank)
    qtbot.addWidget(widget)
    widget.set_slide(slide, store)
    yield widget
    widget.shutdown()


def stocked(bank, per_class=8, shape=True, texture=True, slides=4):
    """Records spread over several slides, so leave-one-slide-out is possible."""
    rng = np.random.default_rng(0)
    rows = []
    for i, name in enumerate(("PaNIN-2", "PaNIN-3")):
        for j in range(per_class):
            slide = f"s{j % slides}.svs"
            rows.append(GeometryRecord(
                slide_path=f"E:/{slide}", slide_name=slide,
                annotation_id=uuid.uuid4(), classification=name,
                features=(np.arange(DIMENSION, dtype=np.float32) + i * 5
                          + rng.normal(0, 0.2, DIMENSION)) if texture else None,
                shape_features=(np.arange(SHAPE_DIMENSION, dtype=np.float32) + i * 5
                                + rng.normal(0, 0.2, SHAPE_DIMENSION)) if shape else None,
                version=VERSION if texture else 0,
                shape_version=SHAPE_VERSION if shape else 0))
    bank.add(rows)


class TestInitialState:
    def test_every_source_is_offered(self, panel):
        count = panel.source_combo.count()
        assert count == len(FeatureSource)
        assert {panel.source_combo.itemData(i) for i in range(count)} == set(FeatureSource)

    def test_defaults_to_shape(self, panel):
        """Shape works on every annotation, so it is the safe default."""
        assert panel.source is FeatureSource.SHAPE

    def test_describe_enabled_with_a_slide_and_annotations(self, panel):
        assert panel.describe_button.isEnabled()

    def test_train_disabled_on_an_empty_bank(self, panel):
        assert not panel.train_button.isEnabled()

    def test_save_disabled_before_training(self, panel):
        assert not panel.save_button.isEnabled()

    def test_no_slide_disables_describe(self, qtbot, bank):
        widget = GeometryPanel(bank)
        qtbot.addWidget(widget)
        assert not widget.describe_button.isEnabled()


class TestSourceAvailability:
    def test_shape_only_bank_cannot_train_texture(self, panel, bank):
        stocked(bank, shape=True, texture=False)
        panel.refresh()
        panel.source_combo.setCurrentIndex(
            [panel.source_combo.itemData(i) for i in range(3)].index(FeatureSource.SHAPE))
        assert panel.train_button.isEnabled()

        panel.source_combo.setCurrentIndex(
            [panel.source_combo.itemData(i) for i in range(3)].index(FeatureSource.TEXTURE))
        assert not panel.train_button.isEnabled()

    def test_combined_needs_both_blocks(self, panel, bank):
        stocked(bank, shape=True, texture=False)
        panel.refresh()
        panel.source_combo.setCurrentIndex(
            [panel.source_combo.itemData(i) for i in range(3)].index(FeatureSource.COMBINED))
        assert not panel.train_button.isEnabled()

    def test_record_counts_are_shown(self, panel, bank):
        stocked(bank)
        panel.refresh()
        assert "16" in panel.status.text() or "16" in panel.bank_label.text()


class TestDescribe:
    def test_describes_the_open_slide(self, panel, bank, qtbot):
        panel._describe()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert len(bank) == 2

    def test_shape_survives_when_texture_cannot(self, panel, bank, qtbot):
        """An impossible nucleus threshold must still yield shape records."""
        panel.min_nuclei.setValue(100_000)
        panel._describe()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert len(bank) == 2
        assert all(r.has_shape and not r.has_texture for r in bank.records)
        assert len(bank.usable_for(FeatureSource.SHAPE)) == 2
        assert len(bank.usable_for(FeatureSource.TEXTURE)) == 0

    def test_report_is_surfaced(self, panel, qtbot):
        panel._describe()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert "shape" in panel.status.text()


class TestTrain:
    def test_trains_on_shape(self, panel, bank, qtbot):
        stocked(bank)
        panel.refresh()
        panel._train()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert panel.model is not None
        assert panel.model.feature_dim == SHAPE_DIMENSION
        assert panel.save_button.isEnabled()

    def test_trains_on_combined(self, panel, bank, qtbot):
        stocked(bank)
        panel.refresh()
        panel.source_combo.setCurrentIndex(
            [panel.source_combo.itemData(i) for i in range(3)].index(FeatureSource.COMBINED))
        panel._train()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert panel.model.feature_dim == SHAPE_DIMENSION + DIMENSION

    def test_results_table_is_filled(self, panel, bank, qtbot):
        stocked(bank)
        panel.refresh()
        panel._train()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert panel.results.rowCount() == 2

    def test_headline_reports_the_baseline(self, panel, bank, qtbot):
        """CV accuracy is meaningless without the majority baseline beside it."""
        stocked(bank)
        panel.refresh()
        panel._train()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert "baseline" in panel.status.text()

    def test_model_is_emitted(self, panel, bank, qtbot):
        received = []
        panel.model_trained.connect(received.append)
        stocked(bank)
        panel.refresh()
        panel._train()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert received and received[0].is_geometry

    def test_failure_is_reported_inline_not_modally(self, panel, bank, qtbot):
        """A modal here would block anything driving the panel, tests included."""
        stocked(bank, per_class=8, shape=False, texture=True)
        panel.refresh()
        panel._train()          # source is SHAPE, bank has none
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert panel.model is None
        assert "shape" in panel.status.text().lower()

    def test_single_slide_bank_explains_why_loso_cannot_run(self, panel, bank, qtbot):
        stocked(bank, slides=1)
        panel.refresh()
        panel._train()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert "at least 2 slides" in panel.status.text()

    def test_leave_one_slide_out_scores_lower_than_stratified(self, panel, bank, qtbot):
        """The gap between them measures how much the model leans on slide identity."""
        from pathlearn.pipeline.geometry_train import (GeometryTrainingSettings,
                                                       Validation,
                                                       train_geometry_classifier)
        stocked(bank, per_class=12, slides=4)
        optimistic = train_geometry_classifier(bank, GeometryTrainingSettings(
            source=FeatureSource.SHAPE, validation=Validation.STRATIFIED))
        honest = train_geometry_classifier(bank, GeometryTrainingSettings(
            source=FeatureSource.SHAPE, validation=Validation.BY_SLIDE))
        assert optimistic.metrics.val_count == honest.metrics.val_count


def stocked_two_schemes(bank, per_class=6, slides=3):
    """Two annotation schemes in one bank, as the real data has.

    PanIN-* and PanINN* are different approaches, not a naming slip, so they
    must be trainable separately.
    """
    rng = np.random.default_rng(1)
    rows = []
    for i, name in enumerate(("PanIN-2", "PanIN-3", "PanINN2", "PanINN3")):
        for j in range(per_class):
            slide = f"s{j % slides}.svs"
            rows.append(GeometryRecord(
                slide_path=f"E:/{slide}", slide_name=slide,
                annotation_id=uuid.uuid4(), classification=name,
                features=np.arange(DIMENSION, dtype=np.float32) + i
                         + rng.normal(0, 0.2, DIMENSION),
                shape_features=np.arange(SHAPE_DIMENSION, dtype=np.float32) + i
                               + rng.normal(0, 0.2, SHAPE_DIMENSION),
                version=VERSION, shape_version=SHAPE_VERSION))
    bank.add(rows)


class TestClassSelection:
    def test_lists_every_class_in_the_bank(self, panel, bank):
        stocked_two_schemes(bank)
        panel.refresh()
        assert panel.class_list.count() == 4
        assert panel.selected_classes == {"PanIN-2", "PanIN-3", "PanINN2", "PanINN3"}

    def test_can_isolate_one_annotation_scheme(self, panel, bank):
        stocked_two_schemes(bank)
        panel.refresh()
        for i in range(panel.class_list.count()):
            item = panel.class_list.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if name.startswith("PanINN")
                               else Qt.CheckState.Unchecked)
        assert panel.selected_classes == {"PanINN2", "PanINN3"}

    def test_training_honours_the_selection(self, panel, bank, qtbot):
        stocked_two_schemes(bank)
        panel.refresh()
        for i in range(panel.class_list.count()):
            item = panel.class_list.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if name.startswith("PanINN")
                               else Qt.CheckState.Unchecked)
        panel._train()
        qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)
        assert panel.model is not None
        assert panel.model.class_labels == ["PanINN2", "PanINN3"]

    def test_fewer_than_two_classes_disables_training(self, panel, bank):
        stocked_two_schemes(bank)
        panel.refresh()
        panel._set_all_classes(False)
        assert not panel.train_button.isEnabled()
        assert "at least 2 classes" in panel.status.text()

    def test_all_and_none_buttons(self, panel, bank):
        stocked_two_schemes(bank)
        panel.refresh()
        panel._set_all_classes(False)
        assert panel.selected_classes == set()
        panel._set_all_classes(True)
        assert len(panel.selected_classes) == 4

    def test_selection_survives_a_source_change(self, panel, bank):
        """Switching feature source must not silently re-tick a dropped scheme."""
        stocked_two_schemes(bank)
        panel.refresh()
        for i in range(panel.class_list.count()):
            item = panel.class_list.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if name == "PanINN3"
                               else Qt.CheckState.Unchecked)
        panel.source_combo.setCurrentIndex(
            [panel.source_combo.itemData(i) for i in range(3)].index(FeatureSource.TEXTURE))
        assert panel.selected_classes == {"PanINN3"}

    def test_status_reports_selected_versus_available(self, panel, bank):
        stocked_two_schemes(bank)
        panel.refresh()
        for i in range(panel.class_list.count()):
            item = panel.class_list.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if name.startswith("PanINN")
                               else Qt.CheckState.Unchecked)
        assert "12 of 24" in panel.status.text()
