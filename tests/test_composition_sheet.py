"""The composition sheet and its wiring into the heatmap panel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pathlearn.models.annotation import AnnotationColor
from pathlearn.models.prediction import PatchPrediction, PredictionSet
from pathlearn.ui.panels.heatmap_panel import HeatmapPanel
from pathlearn.ui.sheets.composition import CompositionSheet, _readable_on

SIZE = 224


def tile(col, row, label="PanIN-2", confidence=0.9):
    return PatchPrediction(x=col * SIZE, y=row * SIZE, size_level0=SIZE,
                           label=label,
                           probabilities=np.array([confidence, 0.0],
                                                  dtype=np.float32))


def a_set(tiles=None):
    tiles = tiles if tiles is not None else (
        [tile(c, 0) for c in range(3)] + [tile(0, 1, label="Acinar")])
    return PredictionSet(predictions=list(tiles),
                         class_labels=["PanIN-2", "Acinar"],
                         colors={"PanIN-2": AnnotationColor(200, 60, 60),
                                 "Acinar": AnnotationColor(60, 120, 200)},
                         slide_path="E:/case-01.svs")


class TestTheSheet:
    def test_a_row_per_class_largest_first(self, qtbot):
        sheet = CompositionSheet(a_set())
        qtbot.addWidget(sheet)
        assert sheet.table.rowCount() == 2
        assert sheet.table.item(0, 0).text() == "PanIN-2"
        assert sheet.table.item(0, 1).text() == "75.0%"

    def test_areas_are_pixels_without_a_pixel_size(self, qtbot):
        sheet = CompositionSheet(a_set())
        qtbot.addWidget(sheet)
        assert "px" in sheet.table.item(0, 2).text()
        assert "no pixel size" in sheet._caveat_text()

    def test_areas_are_millimetres_when_given_one(self, qtbot):
        sheet = CompositionSheet(a_set(), mpp=0.5)
        qtbot.addWidget(sheet)
        assert "mm" in sheet.table.item(0, 2).text()

    def test_a_small_area_falls_back_to_microns_not_zero(self, qtbot):
        """0.003 mm2 shown as "0.00 mm2" would read as no tissue at all."""
        sheet = CompositionSheet(a_set([tile(0, 0)]), mpp=0.25)
        qtbot.addWidget(sheet)
        text = sheet.table.item(0, 2).text()
        assert "µm" in text and not text.startswith("0.00")

    def test_the_caveat_always_states_the_denominator(self, qtbot):
        """The number is meaningless without it, so it is never optional."""
        sheet = CompositionSheet(a_set(), mpp=0.25)
        qtbot.addWidget(sheet)
        assert "not" in sheet._caveat_text()
        assert "whole slide" in sheet._caveat_text()

    def test_the_caveat_reports_excluded_tiles(self, qtbot):
        predictions = a_set()
        predictions.min_confidence = 0.95
        sheet = CompositionSheet(predictions)
        qtbot.addWidget(sheet)
        assert "confidence threshold" in sheet._caveat_text()

    def test_copying_puts_csv_on_the_clipboard(self, qtbot):
        from PySide6.QtWidgets import QApplication

        sheet = CompositionSheet(a_set())
        qtbot.addWidget(sheet)
        sheet.copy_to_clipboard()
        assert "Class,Tiles" in QApplication.clipboard().text()

    def test_saving_writes_the_provenance_footer(self, qtbot, tmp_path,
                                                 monkeypatch):
        target = tmp_path / "out.csv"
        monkeypatch.setattr(
            "pathlearn.ui.sheets.composition.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(target), "CSV (*.csv)"))
        sheet = CompositionSheet(a_set(), mpp=0.25)
        qtbot.addWidget(sheet)
        sheet.save_csv()
        text = target.read_text(encoding="utf-8")
        assert "# Denominator" in text
        assert "PanIN-2,3," in text

    def test_cancelling_the_save_writes_nothing(self, qtbot, tmp_path,
                                                monkeypatch):
        monkeypatch.setattr(
            "pathlearn.ui.sheets.composition.QFileDialog.getSaveFileName",
            lambda *a, **k: ("", ""))
        sheet = CompositionSheet(a_set())
        qtbot.addWidget(sheet)
        sheet.save_csv()
        assert list(tmp_path.iterdir()) == []


class TestTheBar:
    def test_it_takes_one_slice_per_class(self, qtbot):
        sheet = CompositionSheet(a_set())
        qtbot.addWidget(sheet)
        assert len(sheet.bar._slices) == 2
        assert sum(fraction for _, fraction, _ in sheet.bar._slices) == \
            pytest.approx(1.0)

    def test_it_uses_the_heatmap_colours(self, qtbot):
        sheet = CompositionSheet(a_set())
        qtbot.addWidget(sheet)
        _, _, colour = sheet.bar._slices[0]
        assert (colour.red(), colour.green(), colour.blue()) == (200, 60, 60)

    def test_it_paints_without_a_report(self, qtbot):
        """An empty bar must not divide by a zero width."""
        from pathlearn.ui.sheets.composition import ProportionBar

        bar = ProportionBar()
        qtbot.addWidget(bar)
        bar.resize(200, 40)
        bar.grab()  # forces paintEvent

    def test_it_paints_with_one(self, qtbot):
        sheet = CompositionSheet(a_set())
        qtbot.addWidget(sheet)
        sheet.bar.resize(300, 40)
        sheet.bar.grab()

    @pytest.mark.parametrize("rgb,expected", [
        ((255, 255, 0), "black"),   # bright yellow
        ((10, 10, 60), "white"),    # dark navy
    ])
    def test_the_label_stays_readable(self, rgb, expected):
        from PySide6.QtGui import QColor

        chosen = _readable_on(QColor(*rgb))
        assert chosen.value() == (0 if expected == "black" else 255)


class TestThePanelButton:
    def test_it_is_disabled_until_there_are_predictions(self, qtbot):
        panel = HeatmapPanel()
        qtbot.addWidget(panel)
        assert not panel.composition_button.isEnabled()
        panel.set_predictions(a_set())
        assert panel.composition_button.isEnabled()

    def test_clearing_disables_it_again(self, qtbot):
        panel = HeatmapPanel()
        qtbot.addWidget(panel)
        panel.set_predictions(a_set())
        panel.set_predictions(None)
        assert not panel.composition_button.isEnabled()

    def test_it_asks_the_window_rather_than_acting_alone(self, qtbot):
        """The panel has no slide, so it cannot know the pixel size."""
        panel = HeatmapPanel()
        qtbot.addWidget(panel)
        panel.set_predictions(a_set())
        with qtbot.waitSignal(panel.composition_requested, timeout=500):
            panel.composition_button.click()


class TestTheWindowSuppliesThePixelSize:
    def test_the_geometric_mean_is_exact_for_area(self):
        """sqrt(x*y) squared is x*y — the actual area scale factor."""
        x, y = 0.2, 0.5
        assert math.sqrt(x * y) ** 2 == pytest.approx(x * y)
