"""The project table window and its wiring into the main window."""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.models.prediction import PatchPrediction, PredictionSet
from pathlearn.pipeline.composition import composition
from pathlearn.project import PROJECT_SUFFIX, Project, ProjectEntry
from pathlearn.ui.sheets.composition import CompositionSheet
from pathlearn.ui.windows.project_window import FIXED_COLUMNS, ProjectWindow

SIZE = 224


class _Settings:
    """Stand-in for QSettings so a test never touches the user's registry."""

    def __init__(self, *_args) -> None:
        self._values: dict = {}

    def value(self, key, default=None):
        return self._values.get(key, default)

    def setValue(self, key, value) -> None:
        self._values[key] = value


def tile(col, row, label="PanIN-2", confidence=0.9):
    return PatchPrediction(x=col * SIZE, y=row * SIZE, size_level0=SIZE,
                           label=label,
                           probabilities=np.array([confidence, 0.0],
                                                  dtype=np.float32))


def a_set(tiles=None, labels=("PanIN-2", "Acinar"), slide="E:/case-01.svs"):
    tiles = tiles if tiles is not None else (
        [tile(c, 0) for c in range(3)] + [tile(0, 1, label="Acinar")])
    return PredictionSet(predictions=list(tiles), class_labels=list(labels),
                         slide_path=slide, extractor_identity="onnx:uni2-h:r1")


def an_entry(slide="E:/case-01.svs", labels=("PanIN-2", "Acinar"), tiles=None,
             mpp=0.25):
    predictions = a_set(tiles, labels, slide)
    return ProjectEntry.from_report(composition(predictions, mpp=mpp),
                                    predictions, model_name="4-class model")


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / f"cohort{PROJECT_SUFFIX}", "PanIN cohort")


@pytest.fixture(scope="session")
def slide_file(tmp_path_factory):
    from synthetic_slide import write_synthetic_slide

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


def row_for(view, slide_name: str) -> int:
    """Find a row by slide, since the table is sortable and rows move."""
    for row in range(view.table.rowCount()):
        if view.table.item(row, 0).text() == slide_name:
            return row
    raise AssertionError(f"{slide_name} is not in the table")


def column_for(view, header: str) -> int:
    headers = [view.table.horizontalHeaderItem(c).text()
               for c in range(view.table.columnCount())]
    return headers.index(header)


class TestTheTable:
    def test_it_opens_in_ascending_slide_order(self, qtbot, project):
        """Qt defaults the sort indicator to descending; that reads as shuffled."""
        project.add(an_entry(slide="E:/b.svs"))
        project.add(an_entry(slide="E:/a.svs"))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        assert [view.table.item(r, 0).text() for r in range(2)] ==             ["a.svs", "b.svs"]

    def test_a_row_per_slide_and_a_column_per_class(self, qtbot, project):
        project.add(an_entry(slide="E:/a.svs"))
        project.add(an_entry(slide="E:/b.svs"))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        assert view.table.rowCount() == 2
        assert view.table.columnCount() == len(FIXED_COLUMNS) + 2

    def test_a_class_the_model_never_had_stays_blank(self, qtbot, project):
        """Blank means not measured. Filling it with 0 would be a lie."""
        project.add(an_entry(slide="E:/a.svs", labels=("PanIN-2", "Acinar")))
        project.add(an_entry(slide="E:/b.svs", labels=("PanIN-3",),
                             tiles=[tile(0, 0, label="PanIN-3")]))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        column = column_for(view, "PanIN-3 %")
        blank = view.table.item(row_for(view, "a.svs"), column)
        assert blank.text() == ""
        assert "not measured" in blank.toolTip()
        assert view.table.item(row_for(view, "b.svs"), column).text() == "100.0"

    def test_a_class_found_none_shows_a_zero(self, qtbot, project):
        project.add(an_entry(slide="E:/a.svs", tiles=[tile(0, 0)],
                             labels=("PanIN-2", "Acinar")))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        assert view.table.item(row_for(view, "a.svs"),
                               column_for(view, "Acinar %")).text() == "0.0"

    def test_an_empty_project_still_opens(self, qtbot, project):
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        assert view.table.rowCount() == 0
        assert "empty" in view.headline.text()
        assert not view.remove_button.isEnabled()

    def test_the_slide_path_rides_along_for_removal(self, qtbot, project):
        project.add(an_entry(slide="E:/slides/a.svs"))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        assert view.table.item(0, 0).text() == "a.svs"
        # Path AND source: a slide can carry a row per source, and Remove has
        # to take the row that was picked rather than the whole slide.
        assert view.table.item(0, 0).data(0x0100) == ("E:/slides/a.svs",
                                                      "predicted")

    def test_reloading_picks_up_a_new_slide(self, qtbot, project):
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        project.add(an_entry(slide="E:/a.svs"))
        view.reload()
        assert view.table.rowCount() == 1


class TestTheMixedExtractorWarning:
    def test_it_is_hidden_for_one_extractor(self, qtbot, project):
        project.add(an_entry(slide="E:/a.svs"))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        assert not view.warning.isVisible()

    def test_it_appears_for_two(self, qtbot, project):
        first = an_entry(slide="E:/a.svs")
        second = an_entry(slide="E:/b.svs")
        second.extractor_identity = "onnx:phikon-v1:r1"
        project.add(first)
        project.add(second)
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        view.show()
        assert view.warning.isVisible()
        assert "not comparable" in view.warning.text()


class TestExporting:
    def test_csv_writes_both_shapes(self, qtbot, project, tmp_path,
                                    monkeypatch):
        project.add(an_entry(slide="E:/a.svs"))
        target = tmp_path / "cohort.csv"
        monkeypatch.setattr(
            "pathlearn.ui.windows.project_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(target), "CSV (*.csv)"))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        view.export_csv()
        assert target.exists()
        # The long form goes beside it, so nobody has to reshape by hand.
        assert (tmp_path / "cohort-long.csv").exists()

    def test_excel_writes_a_workbook(self, qtbot, project, tmp_path,
                                     monkeypatch):
        pytest.importorskip("openpyxl")
        project.add(an_entry(slide="E:/a.svs"))
        target = tmp_path / "cohort.xlsx"
        monkeypatch.setattr(
            "pathlearn.ui.windows.project_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(target), "Excel (*.xlsx)"))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        view.export_xlsx()
        assert target.exists()

    def test_cancelling_writes_nothing(self, qtbot, project, tmp_path,
                                       monkeypatch):
        project.add(an_entry(slide="E:/a.svs"))
        monkeypatch.setattr(
            "pathlearn.ui.windows.project_window.QFileDialog.getSaveFileName",
            lambda *a, **k: ("", ""))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        view.export_csv()
        view.export_xlsx()
        assert not list(tmp_path.glob("*.csv"))

    def test_copying_puts_the_table_on_the_clipboard(self, qtbot, project):
        from PySide6.QtWidgets import QApplication

        project.add(an_entry(slide="E:/a.svs"))
        view = ProjectWindow(project)
        qtbot.addWidget(view)
        view.copy_table()
        assert "Slide,Source,Added,Model" in QApplication.clipboard().text()


class TestTheMainWindow:
    def test_no_project_is_open_at_first(self, window):
        assert window.project is None
        assert not window.action_add_to_project.isEnabled()
        assert not window.action_project_table.isEnabled()

    def test_adopting_a_project_enables_the_actions(self, window, project):
        window._adopt_project(project)
        assert window.action_add_to_project.isEnabled()
        assert window.action_close_project.isEnabled()

    def test_the_title_carries_the_project(self, window, project):
        window._adopt_project(project)
        assert "PanIN cohort" in window.windowTitle()

    def test_closing_takes_it_out_of_the_title(self, window, project):
        window._adopt_project(project)
        window.close_project()
        assert "PanIN cohort" not in window.windowTitle()
        assert window.project is None

    def test_adding_without_a_project_refuses_quietly(self, window):
        assert "No project is open" in window.add_to_project(quiet=True)

    def test_adding_without_predictions_refuses(self, window, project):
        window._adopt_project(project)
        assert "Run a prediction first" in window.add_to_project(quiet=True)

    def test_adding_files_the_breakdown(self, window, project):
        window._adopt_project(project)
        window.heatmap_panel.set_predictions(a_set())
        message = window.add_to_project(quiet=True)
        assert "Added case-01.svs" in message
        assert len(window.project) == 1
        assert window.project.entries[0].percent_for("PanIN-2") == \
            pytest.approx(75.0)

    def test_adding_writes_through_to_disk(self, window, project):
        """The entry must survive a crash on the next line, not just a close."""
        window._adopt_project(project)
        window.heatmap_panel.set_predictions(a_set())
        window.add_to_project(quiet=True)
        assert len(Project.load(project.path)) == 1

    def test_two_slides_accumulate(self, window, project):
        window._adopt_project(project)
        for slide in ("E:/a.svs", "E:/b.svs"):
            window.heatmap_panel.set_predictions(a_set(slide=slide))
            window.add_to_project(quiet=True)
        assert len(window.project) == 2

    def test_predictions_with_no_slide_cannot_be_filed(self, window, project):
        window._adopt_project(project)
        window.heatmap_panel.set_predictions(a_set(slide=""))
        assert "do not name a slide" in window.add_to_project(quiet=True)
        assert window.project.is_empty

    def test_everything_below_the_threshold_is_not_filed(self, window,
                                                         project):
        window._adopt_project(project)
        predictions = a_set()
        predictions.min_confidence = 0.99
        window.heatmap_panel.set_predictions(predictions)
        assert "no breakdown to file" in window.add_to_project(quiet=True)
        assert window.project.is_empty

    def test_reopening_the_last_project_at_launch(self, window, project):
        window._adopt_project(project)
        window.settings.setValue("lastProjectPath", str(project.path))
        window.project = None
        window._reopen_last_project()
        assert window.project is not None
        assert window.project.name == "PanIN cohort"

    def test_a_project_that_has_moved_does_not_error_at_launch(self, window,
                                                               project):
        """A missing file must not greet the user with a dialog."""
        window.settings.setValue("lastProjectPath",
                                 str(project.path) + "-gone")
        window.project = None
        window._reopen_last_project()
        assert window.project is None

    def test_a_corrupt_project_does_not_error_at_launch(self, window, project):
        project.path.write_text("junk", encoding="utf-8")
        window.settings.setValue("lastProjectPath", str(project.path))
        window.project = None
        window._reopen_last_project()
        assert window.project is None


class TestAddingAnnotations:
    """The annotation breakdown files as its own row, not over the prediction."""

    def a_traced_slide(self, window):
        from pathlearn.models.annotation import (Annotation, AnnotationColor,
                                                 Point)

        def square(x, side, label):
            return Annotation(
                points=[Point(x, 0), Point(x + side, 0),
                        Point(x + side, side), Point(x, side)],
                classification=label, color=AnnotationColor(200, 60, 60))

        for annotation in (square(0, 10, "PanIN-1a"), square(50, 10, "PanIN-1a"),
                           square(200, 100, "PanIN-3")):
            window.store.add(annotation)

    def test_without_a_project_it_refuses(self, window):
        assert "No project is open" in window.add_annotations_to_project(
            quiet=True)

    def test_without_a_slide_it_refuses(self, window, project):
        window._adopt_project(project)
        assert "Open a slide first" in window.add_annotations_to_project(
            quiet=True)

    def test_the_action_is_disabled_until_a_project_is_open(self, window,
                                                            project):
        assert not window.action_add_annotations_to_project.isEnabled()
        window._adopt_project(project)
        assert window.action_add_annotations_to_project.isEnabled()

    def test_a_slide_with_no_annotations_refuses(self, window, project, slide):
        window._adopt_project(project)
        window.open_slide(slide)
        assert "no annotations" in window.add_annotations_to_project(
            quiet=True)

    def test_it_files_what_was_drawn(self, window, project, slide):
        window._adopt_project(project)
        window.open_slide(slide)
        self.a_traced_slide(window)
        message = window.add_annotations_to_project(quiet=True)
        assert "annotation breakdown" in message
        entry = window.project.entry_for(str(slide), "annotated")
        assert entry is not None
        assert entry.total_tiles == 3
        assert entry.unit_name == "regions"

    def test_it_does_not_disturb_the_predicted_row(self, window, project,
                                                   slide):
        """Both sources for one slide, which is the point of the feature."""
        window._adopt_project(project)
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set(slide=str(slide)))
        window.add_to_project(quiet=True)
        self.a_traced_slide(window)
        window.add_annotations_to_project(quiet=True)

        assert len(window.project) == 2
        assert window.project.entry_for(str(slide), "predicted") is not None
        assert window.project.entry_for(str(slide), "annotated") is not None

    def test_the_palette_makes_undrawn_classes_a_real_zero(self, window,
                                                           project, slide):
        window._adopt_project(project)
        window.open_slide(slide)
        self.a_traced_slide(window)
        window.add_annotations_to_project(quiet=True)
        entry = window.project.entry_for(str(slide), "annotated")
        palette = [c.name for c in window.profile.classes]
        undrawn = [n for n in palette if n not in ("PanIN-1a", "PanIN-3")]
        if undrawn:
            assert entry.percent_for(undrawn[0]) == 0.0
        assert entry.percent_for("Not in any palette") is None

    def test_it_writes_through_to_disk(self, window, project, slide):
        window._adopt_project(project)
        window.open_slide(slide)
        self.a_traced_slide(window)
        window.add_annotations_to_project(quiet=True)
        assert len(Project.load(project.path)) == 1

    def test_restricting_to_use_is_recorded_as_the_scope(self, window,
                                                         project, slide):
        window._adopt_project(project)
        window.open_slide(slide)
        self.a_traced_slide(window)
        window.store.select_all(False)
        for annotation in list(window.store.annotations)[:1]:
            window.store.set_selected(annotation.id, True)
        window.add_annotations_to_project(quiet=True, use_only=True)
        entry = window.project.entry_for(str(slide), "annotated")
        assert entry.total_tiles == 1
        assert "checked under Use" in entry.scope

    def test_both_rows_show_in_the_table(self, window, project, slide):
        window._adopt_project(project)
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set(slide=str(slide)))
        window.add_to_project(quiet=True)
        self.a_traced_slide(window)
        window.add_annotations_to_project(quiet=True)

        view = ProjectWindow(window.project)
        assert view.table.rowCount() == 2
        sources = {view.table.item(r, 1).text() for r in range(2)}
        assert sources == {"predicted", "annotated"}
        view.deleteLater()


class TestTheCompositionSheetButton:
    def test_it_is_disabled_with_no_project(self, qtbot):
        sheet = CompositionSheet(a_set())
        qtbot.addWidget(sheet)
        assert not sheet.project_button.isEnabled()
        assert "No project is open" in sheet.project_button.toolTip()

    def test_it_names_the_open_project(self, qtbot):
        sheet = CompositionSheet(a_set(), project_name="PanIN cohort")
        qtbot.addWidget(sheet)
        assert sheet.project_button.isEnabled()
        assert "PanIN cohort" in sheet.project_button.text()

    def test_it_asks_the_window_rather_than_acting_alone(self, qtbot):
        sheet = CompositionSheet(a_set(), project_name="Cohort")
        qtbot.addWidget(sheet)
        with qtbot.waitSignal(sheet.add_to_project_requested, timeout=500):
            sheet.project_button.click()
        assert sheet.added_to_project

    def test_the_window_reports_back_into_the_sheet(self, qtbot):
        """Not a dialog on top of a dialog."""
        sheet = CompositionSheet(a_set(), project_name="Cohort")
        qtbot.addWidget(sheet)
        sheet.note("Added case-01.svs — 1 slide(s) in the project.")
        assert "1 slide(s)" in sheet.headline.text()
