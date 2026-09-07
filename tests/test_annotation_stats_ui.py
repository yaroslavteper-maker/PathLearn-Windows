"""The annotation class-breakdown sheet and its sidebar button."""

from __future__ import annotations

import pytest

from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.ui.sheets.annotation_stats import COLUMNS, AnnotationStatsSheet


def square(x, y, side, label="PanIN-2", subtractive=False, colour=None,
           selected=True):
    annotation = Annotation(
        points=[Point(x, y), Point(x + side, y),
                Point(x + side, y + side), Point(x, y + side)],
        classification=label,
        color=colour or AnnotationColor(200, 60, 60),
        is_subtractive=subtractive)
    annotation.is_selected = selected
    return annotation


def a_tracing():
    """Nine small 1a ducts and one big grade-3 lesion: count != area."""
    return ([square(i * 20, 0, 10, label="PanIN-1a",
                    colour=AnnotationColor(60, 120, 200)) for i in range(9)]
            + [square(0, 500, 100, label="PanIN-3")])


def column_for(sheet, header: str) -> int:
    return COLUMNS.index(header)


def row_for(sheet, label: str) -> int:
    for row in range(sheet.table.rowCount()):
        if sheet.table.item(row, 0).text() == label:
            return row
    raise AssertionError(f"{label} is not in the table")


class TestTheTable:
    def test_a_row_per_class(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        assert sheet.table.rowCount() == 2

    def test_count_and_area_are_both_shown_and_differ(self, qtbot):
        """The whole point: 90% by count, under 10% by area."""
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        row = row_for(sheet, "PanIN-1a")
        assert sheet.table.item(row, column_for(sheet, "Count %")).text() == \
            "90.0%"
        area = sheet.table.item(row, column_for(sheet, "Area %")).text()
        assert area == "8.3%"

    def test_the_mean_size_column(self, qtbot):
        sheet = AnnotationStatsSheet([square(0, 0, 10), square(100, 0, 30)])
        qtbot.addWidget(sheet)
        assert "500" in sheet.table.item(0, column_for(sheet, "Mean size")).text()

    def test_the_class_colour_comes_from_the_annotation(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        colour = sheet._colour_for("PanIN-1a")
        assert (colour.red(), colour.green(), colour.blue()) == (60, 120, 200)

    def test_areas_are_millimetres_when_given_a_pixel_size(self, qtbot):
        sheet = AnnotationStatsSheet([square(0, 0, 4000)], mpp=0.5)
        qtbot.addWidget(sheet)
        assert "mm" in sheet.table.item(0, column_for(sheet, "Area")).text()


class TestTheBars:
    def test_there_are_two_and_they_disagree(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        counts = dict((label, fraction)
                      for label, fraction, _ in sheet.count_bar._slices)
        areas = dict((label, fraction)
                     for label, fraction, _ in sheet.area_bar._slices)
        assert counts["PanIN-1a"] == pytest.approx(0.9)
        assert areas["PanIN-1a"] < 0.1

    def test_each_bar_sums_to_one(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        for bar in (sheet.count_bar, sheet.area_bar):
            assert sum(f for _, f, _ in bar._slices) == pytest.approx(1.0)

    def test_they_paint(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        for bar in (sheet.count_bar, sheet.area_bar):
            bar.resize(300, 40)
            bar.grab()


class TestScope:
    def test_it_covers_everything_by_default(self, qtbot):
        everything = a_tracing()
        sheet = AnnotationStatsSheet(everything, in_use=everything[:2])
        qtbot.addWidget(sheet)
        assert sheet.report.total_count == 10
        assert not sheet.use_only.isChecked()

    def test_ticking_use_restricts_it(self, qtbot):
        everything = a_tracing()
        sheet = AnnotationStatsSheet(everything, in_use=everything[:2])
        qtbot.addWidget(sheet)
        sheet.use_only.setChecked(True)
        assert sheet.report.total_count == 2
        assert "checked under Use" in sheet.report.scope

    def test_the_option_is_dead_when_everything_is_in_use(self, qtbot):
        everything = a_tracing()
        sheet = AnnotationStatsSheet(everything, in_use=everything)
        qtbot.addWidget(sheet)
        assert not sheet.use_only.isEnabled()

    def test_the_scope_reaches_the_csv(self, qtbot):
        everything = a_tracing()
        sheet = AnnotationStatsSheet(everything, in_use=everything[:2])
        qtbot.addWidget(sheet)
        sheet.use_only.setChecked(True)
        assert "# Scope,annotations checked under Use" in sheet.report.to_csv()


class TestCaveats:
    def test_subtractive_polygons_are_explained(self, qtbot):
        sheet = AnnotationStatsSheet([square(0, 0, 100),
                                      square(10, 10, 20, subtractive=True)])
        qtbot.addWidget(sheet)
        assert "subtractive" in sheet._caveat_text()
        assert "deducted" in sheet._caveat_text()

    def test_an_orphan_hole_is_flagged(self, qtbot):
        sheet = AnnotationStatsSheet([square(0, 0, 100),
                                      square(9000, 9000, 20,
                                             subtractive=True)])
        qtbot.addWidget(sheet)
        assert "inside no region" in sheet._caveat_text()

    def test_overlap_makes_the_areas_upper_bounds(self, qtbot):
        sheet = AnnotationStatsSheet([square(0, 0, 100),
                                      square(50, 50, 100, label="Acinar")])
        qtbot.addWidget(sheet)
        assert "upper bounds" in sheet._caveat_text()

    def test_a_clean_tracing_has_no_caveat(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing(), mpp=0.25)
        qtbot.addWidget(sheet)
        assert sheet._caveat_text() == ""

    def test_a_missing_pixel_size_is_mentioned(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        assert "no pixel size" in sheet._caveat_text()


class TestActions:
    def test_copying_puts_csv_on_the_clipboard(self, qtbot):
        from PySide6.QtWidgets import QApplication

        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        sheet.copy_to_clipboard()
        assert "Class,Annotations,Count %" in QApplication.clipboard().text()

    def test_saving_writes_the_table(self, qtbot, tmp_path, monkeypatch):
        target = tmp_path / "classes.csv"
        monkeypatch.setattr(
            "pathlearn.ui.sheets.annotation_stats.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(target), "CSV (*.csv)"))
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        sheet.save_csv()
        assert "PanIN-1a,9,90.00" in target.read_text(encoding="utf-8")

    def test_cancelling_the_save_writes_nothing(self, qtbot, tmp_path,
                                                monkeypatch):
        monkeypatch.setattr(
            "pathlearn.ui.sheets.annotation_stats.QFileDialog.getSaveFileName",
            lambda *a, **k: ("", ""))
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        sheet.save_csv()
        assert list(tmp_path.iterdir()) == []


class TestTheProjectButton:
    def test_it_is_disabled_with_no_project(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing())
        qtbot.addWidget(sheet)
        assert not sheet.project_button.isEnabled()
        assert "No project is open" in sheet.project_button.toolTip()

    def test_it_names_the_open_project(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing(), project_name="PanIN cohort")
        qtbot.addWidget(sheet)
        assert sheet.project_button.isEnabled()
        assert "PanIN cohort" in sheet.project_button.text()

    def test_it_asks_the_window_rather_than_acting_alone(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing(), project_name="Cohort")
        qtbot.addWidget(sheet)
        with qtbot.waitSignal(sheet.add_to_project_requested, timeout=500):
            sheet.project_button.click()
        assert sheet.added_to_project

    def test_the_window_reports_back_into_the_sheet(self, qtbot):
        sheet = AnnotationStatsSheet(a_tracing(), project_name="Cohort")
        qtbot.addWidget(sheet)
        sheet.note("Added the annotation breakdown for case-01.svs")
        assert "annotation breakdown" in sheet.headline.text()


class TestTheSidebarButton:
    def test_it_asks_the_window_rather_than_acting_alone(self, qtbot):
        """The sidebar has no slide, so it cannot know the pixel size."""
        from pathlearn.models.classification import ClassificationProfile
        from pathlearn.models.store import AnnotationStore
        from pathlearn.ui.panels.annotation_sidebar import AnnotationSidebar

        sidebar = AnnotationSidebar(AnnotationStore(),
                                    ClassificationProfile.default())
        qtbot.addWidget(sidebar)
        with qtbot.waitSignal(sidebar.breakdown_requested, timeout=500):
            sidebar.breakdown_button.click()
