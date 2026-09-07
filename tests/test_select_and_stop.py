"""Region selection in Predict, and stopping a training run."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from PySide6.QtCore import Qt

from pathlearn.core.logistic import train as train_logistic
from pathlearn.data.geometry_bank import FeatureSource, GeometryBank, GeometryRecord
from pathlearn.core.panin import PANIN_DIMENSION, PANIN_VERSION
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.classification import ClassificationProfile
from pathlearn.models.store import AnnotationStore
from synthetic_slide import write_synthetic_slide


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("sel") / "s.tif"
    write_synthetic_slide(path, 2048, 1536, levels=3)
    with SlideImage(path) as s:
        yield s


def box(x, y, size=300, cls="PanIN-2", name=None):
    return Annotation(points=[Point(x, y), Point(x + size, y),
                              Point(x + size, y + size), Point(x, y + size)],
                      classification=cls, color=AnnotationColor.default(),
                      name=name)


class TestPredictRegionSelection:
    @pytest.fixture
    def sheet(self, qtbot, slide, tmp_path):
        from pathlearn.extractors.registry import ExtractorRegistry
        from pathlearn.ui.sheets.predict import PredictSheet

        store = AnnotationStore()
        store.bind(tmp_path / "s.svs", slide.dimensions.height)
        for i in range(4):
            store.add(box(100 + i * 350, 200, name=f"R{i}"))
        widget = PredictSheet(slide, store, ExtractorRegistry([tmp_path]),
                              ClassificationProfile.default(), classifier=None)
        qtbot.addWidget(widget)
        yield widget
        widget.reject()

    def test_every_region_starts_selected(self, sheet):
        assert sheet.region_list.count() == 4
        assert len(sheet._selected_regions()) == 4

    def test_the_count_is_shown(self, sheet):
        assert "4 of 4 selected" in sheet.region_count.text()

    def test_select_none_clears_them(self, sheet):
        sheet.select_none_button.click()
        assert sheet._selected_regions() == []
        assert "0 of 4 selected" in sheet.region_count.text()

    def test_select_all_restores_them(self, sheet):
        sheet.select_none_button.click()
        sheet.select_all_button.click()
        assert len(sheet._selected_regions()) == 4

    def test_none_then_pick_two(self, sheet):
        """The reason the buttons exist: clear, then choose a couple."""
        sheet.select_none_button.click()
        for row in (1, 3):
            sheet.region_list.item(row).setCheckState(Qt.CheckState.Checked)
        chosen = sheet._selected_regions()
        assert len(chosen) == 2
        assert {a.display_name for a in chosen} == {"R1", "R3"}
        assert "2 of 4 selected" in sheet.region_count.text()

    def test_subtractive_regions_are_not_listed(self, qtbot, slide, tmp_path):
        from pathlearn.extractors.registry import ExtractorRegistry
        from pathlearn.ui.sheets.predict import PredictSheet

        store = AnnotationStore()
        store.bind(tmp_path / "sub.svs", slide.dimensions.height)
        store.add(box(100, 200))
        hole = box(600, 200)
        hole.is_subtractive = True
        store.add(hole)
        widget = PredictSheet(slide, store, ExtractorRegistry([tmp_path]),
                              ClassificationProfile.default(), classifier=None)
        qtbot.addWidget(widget)
        assert widget.region_list.count() == 1
        widget.reject()


class TestLogisticCancellation:
    """A Stop button that only takes effect between phases is a lie."""

    def data(self, n=400, d=8):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (n, d)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.intp)
        return x, y

    def test_cancelling_stops_the_descent_loop(self):
        x, y = self.data()
        seen = []
        train_logistic(x, y, 2, iterations=500,
                       progress=lambda done, total: seen.append(done),
                       should_cancel=lambda: len(seen) >= 3)
        assert len(seen) < 500, "the loop ran to completion despite cancelling"
        assert len(seen) <= 4

    def test_without_cancelling_it_runs_every_iteration(self):
        x, y = self.data()
        seen = []
        train_logistic(x, y, 2, iterations=25,
                       progress=lambda done, total: seen.append(done))
        assert len(seen) == 25

    def test_a_cancelled_fit_still_returns_a_model(self):
        """Partially trained, and the caller is expected to discard it."""
        x, y = self.data()
        model = train_logistic(x, y, 2, iterations=500,
                               should_cancel=lambda: True)
        assert model.weights.shape == (8, 2)


class TestGeometryPanelStop:
    @pytest.fixture
    def panel(self, qtbot, tmp_path):
        from pathlearn.ui.panels.geometry_panel import GeometryPanel

        bank = GeometryBank(tmp_path / "g.json")
        rng = np.random.default_rng(0)
        rows = []
        for i, name in enumerate(("PanIN-1a", "PanIN-2")):
            for j in range(8):
                rows.append(GeometryRecord(
                    slide_path=f"S{j % 4}.svs", slide_name=f"S{j % 4}.svs",
                    annotation_id=uuid.uuid4(), classification=name,
                    panin_features=rng.normal(i * 4, .3,
                                              PANIN_DIMENSION).astype(np.float32),
                    panin_version=PANIN_VERSION))
        bank.add(rows)
        widget = GeometryPanel(bank)
        qtbot.addWidget(widget)
        index = next(i for i in range(widget.source_combo.count())
                     if widget.source_combo.itemData(i) is FeatureSource.PANIN)
        widget.source_combo.setCurrentIndex(index)
        yield widget
        widget.shutdown()

    def test_stop_is_disabled_when_idle(self, panel):
        assert not panel.stop_button.isEnabled()

    def test_stop_enables_while_running(self, panel):
        panel._set_running(True)
        assert panel.stop_button.isEnabled()
        assert not panel.train_button.isEnabled()
        panel._set_running(False)
        assert not panel.stop_button.isEnabled()

    def test_stopping_an_idle_panel_is_harmless(self, panel):
        panel._stop()
        assert not panel.stop_button.isEnabled()

    def test_a_cancelled_run_produces_no_model(self, qtbot, panel):
        panel._train()
        panel._stop()
        for _ in range(200):
            qtbot.wait(50)
            if panel.task is None:
                break
        assert panel.model is None, "a cancelled run must not leave a model"

    def test_the_status_says_it_is_stopping(self, panel):
        panel._train()
        panel._stop()
        assert "Stopping" in panel.status.text()
        panel.shutdown()


class TestTrainSheetStop:
    @pytest.fixture
    def sheet(self, qtbot, tmp_path):
        from pathlearn.data.bank import PatchBank
        from pathlearn.ui.sheets.train_model import TrainModelSheet

        with PatchBank(tmp_path / "b.db") as bank:
            widget = TrainModelSheet(bank, ClassificationProfile.default())
            qtbot.addWidget(widget)
            yield widget
            widget.reject()

    def test_stop_starts_disabled(self, sheet):
        assert not sheet.stop_button.isEnabled()

    def test_stop_follows_the_running_state(self, sheet):
        sheet._set_running(True)
        assert sheet.stop_button.isEnabled()
        assert not sheet.close_button.isEnabled()
        sheet._set_running(False)
        assert not sheet.stop_button.isEnabled()
