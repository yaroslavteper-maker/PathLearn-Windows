"""Projects: per-slide composition accumulated across sessions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pathlearn.models.prediction import PatchPrediction, PredictionSet
from pathlearn.pipeline.composition import composition
from pathlearn.project import (PROJECT_SUFFIX, EntryShare, Project,
                               ProjectEntry, ProjectError)

SIZE = 224


def tile(col, row, label="PanIN-2", confidence=0.9):
    return PatchPrediction(x=col * SIZE, y=row * SIZE, size_level0=SIZE,
                           label=label,
                           probabilities=np.array([confidence, 0.0],
                                                  dtype=np.float32))


def a_set(tiles, labels=("PanIN-2", "Acinar"), slide="E:/case-01.svs",
          extractor="onnx:uni2-h:r1"):
    return PredictionSet(predictions=list(tiles), class_labels=list(labels),
                         slide_path=slide, extractor_identity=extractor)


def an_entry(slide="E:/case-01.svs", tiles=None, labels=("PanIN-2", "Acinar"),
             mpp=0.25, extractor="onnx:uni2-h:r1", model="4-class model"):
    tiles = tiles if tiles is not None else (
        [tile(c, 0) for c in range(3)] + [tile(0, 1, label="Acinar")])
    predictions = a_set(tiles, labels, slide, extractor)
    report = composition(predictions, mpp=mpp)
    return ProjectEntry.from_report(report, predictions, model_name=model)


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / f"cohort{PROJECT_SUFFIX}", "PanIN cohort")


class TestEntries:
    def test_it_freezes_the_shares(self):
        entry = an_entry()
        assert entry.percent_for("PanIN-2") == pytest.approx(75.0)
        assert entry.percent_for("Acinar") == pytest.approx(25.0)
        assert entry.total_tiles == 4

    def test_it_takes_the_slide_name_from_the_path(self):
        assert an_entry(slide="F:/slides/9659_HFD.svs").slide_name == \
            "9659_HFD.svs"

    def test_it_records_the_run_not_the_app_state(self):
        """Provenance has to come from the prediction that produced it."""
        entry = an_entry(extractor="onnx:phikon-v1:r1")
        assert entry.extractor_identity == "onnx:phikon-v1:r1"
        assert entry.model_classes == ["PanIN-2", "Acinar"]

    def test_the_dominant_class(self):
        assert an_entry().dominant == "PanIN-2"


class TestBlankIsNotZero:
    """The distinction the whole wide table rests on."""

    def test_a_class_the_model_had_but_never_found_is_zero(self):
        entry = an_entry(tiles=[tile(0, 0)], labels=("PanIN-2", "Acinar"))
        assert entry.percent_for("Acinar") == 0.0

    def test_a_class_the_model_never_had_is_none(self):
        entry = an_entry(labels=("PanIN-2", "Acinar"))
        assert entry.percent_for("PanIN-3") is None

    def test_the_same_rule_governs_the_area_column(self):
        entry = an_entry(tiles=[tile(0, 0)], labels=("PanIN-2", "Acinar"))
        assert entry.mm2_for("Acinar") == 0.0
        assert entry.mm2_for("PanIN-3") is None

    def test_no_pixel_size_means_no_area_even_for_a_known_class(self):
        entry = an_entry(tiles=[tile(0, 0)], labels=("PanIN-2", "Acinar"),
                         mpp=None)
        assert entry.percent_for("Acinar") == 0.0
        assert entry.mm2_for("Acinar") is None

    def test_the_wide_table_leaves_it_blank_rather_than_zero(self, project):
        project.add(an_entry(slide="E:/a.svs", labels=("PanIN-2", "Acinar")))
        project.add(an_entry(slide="E:/b.svs", labels=("PanIN-3",),
                             tiles=[tile(0, 0, label="PanIN-3")]))
        rows = project.to_csv().splitlines()
        header = rows[0].split(",")
        panin3 = header.index("PanIN-3 %")
        assert rows[1].split(",")[panin3] == ""      # model never had it
        acinar = header.index("Acinar %")
        assert rows[2].split(",")[acinar] == ""      # likewise, other way round


class TestAccumulating:
    def test_slides_add_up(self, project):
        project.add(an_entry(slide="E:/a.svs"))
        project.add(an_entry(slide="E:/b.svs"))
        assert len(project) == 2

    def test_re_adding_a_slide_replaces_it_by_default(self, project):
        project.add(an_entry(slide="E:/a.svs", tiles=[tile(0, 0)]))
        replaced = project.add(an_entry(slide="E:/a.svs",
                                        tiles=[tile(0, 0), tile(1, 0)]))
        assert replaced is True
        assert len(project) == 1
        assert project.entries[0].total_tiles == 2

    def test_both_can_be_kept_on_purpose(self, project):
        """Comparing two models on one slide is a legitimate thing to want."""
        project.add(an_entry(slide="E:/a.svs"))
        replaced = project.add(an_entry(slide="E:/a.svs"), replace=False)
        assert replaced is False
        assert len(project) == 2

    def test_one_slide_two_cases_is_still_one_slide(self, project):
        """F:\\A.SVS and f:\\a.svs are one file on Windows."""
        project.add(an_entry(slide="F:/Slides/A.SVS"))
        assert project.entry_for("f:/slides/a.svs") is not None
        assert project.add(an_entry(slide="f:/slides/a.svs")) is True
        assert len(project) == 1

    def test_a_class_only_in_the_shares_still_reaches_the_union(self, project):
        """An old prediction set with no class list must not lose its labels."""
        entry = an_entry(slide="E:/a.svs")
        entry.model_classes = []
        project.add(entry)
        assert set(project.class_union()) == {"PanIN-2", "Acinar"}

    def test_removing_a_slide(self, project):
        project.add(an_entry(slide="E:/a.svs"))
        assert project.remove("E:/a.svs") is True
        assert project.is_empty

    def test_removing_something_absent_says_so(self, project):
        assert project.remove("E:/nope.svs") is False

    def test_the_class_union_keeps_model_order(self, project):
        """PanIN-1a, 1b, 2, 3 must not be scrambled into alphabetical."""
        project.add(an_entry(slide="E:/a.svs",
                             labels=("PanIN-1a", "PanIN-1b", "PanIN-2"),
                             tiles=[tile(0, 0, label="PanIN-1a")]))
        project.add(an_entry(slide="E:/b.svs",
                             labels=("PanIN-1a", "PanIN-3"),
                             tiles=[tile(0, 0, label="PanIN-3")]))
        assert project.class_union() == ["PanIN-1a", "PanIN-1b", "PanIN-2",
                                         "PanIN-3"]

    def test_the_summary_counts_slides_and_classes(self, project):
        project.add(an_entry(slide="E:/a.svs"))
        assert "1 slide(s)" in project.summary()

    def test_an_empty_project_says_how_to_start(self, project):
        assert "empty" in project.summary()


class TestMixedExtractors:
    """Two feature spaces in one table is a real problem, not a footnote."""

    def test_one_extractor_is_not_a_warning(self, project):
        project.add(an_entry(slide="E:/a.svs", extractor="onnx:uni2-h:r1"))
        project.add(an_entry(slide="E:/b.svs", extractor="onnx:uni2-h:r1"))
        assert len(project.mixed_extractors()) == 1
        assert "WARNING" not in project.summary()

    def test_two_extractors_warn_in_the_summary(self, project):
        project.add(an_entry(slide="E:/a.svs", extractor="onnx:uni2-h:r1"))
        project.add(an_entry(slide="E:/b.svs", extractor="onnx:phikon-v1:r1"))
        assert len(project.mixed_extractors()) == 2
        assert "WARNING" in project.summary()

    def test_the_warning_reaches_the_exported_table(self, project):
        project.add(an_entry(slide="E:/a.svs", extractor="onnx:uni2-h:r1"))
        project.add(an_entry(slide="E:/b.svs", extractor="geometry:hand:r2"))
        assert "mixed extractors" in project.to_csv()


class TestPersistence:
    def test_a_new_project_is_on_disk_immediately(self, tmp_path):
        path = tmp_path / f"c{PROJECT_SUFFIX}"
        Project.create(path, "Cohort")
        assert path.exists()

    def test_it_round_trips(self, project):
        project.add(an_entry(slide="E:/a.svs"))
        project.save()
        reloaded = Project.load(project.path)
        assert reloaded.name == "PanIN cohort"
        assert len(reloaded) == 1
        assert reloaded.entries[0].percent_for("PanIN-2") == pytest.approx(75.0)

    def test_blank_versus_zero_survives_the_round_trip(self, project):
        project.add(an_entry(slide="E:/a.svs", tiles=[tile(0, 0)],
                             labels=("PanIN-2", "Acinar")))
        project.save()
        entry = Project.load(project.path).entries[0]
        assert entry.percent_for("Acinar") == 0.0
        assert entry.percent_for("PanIN-3") is None

    def test_an_absent_pixel_size_stays_absent(self, project):
        project.add(an_entry(slide="E:/a.svs", mpp=None))
        project.save()
        assert Project.load(project.path).entries[0].total_area_mm2 is None

    def test_saving_is_atomic(self, project, monkeypatch):
        """An interrupted save must not leave half a project behind."""
        project.add(an_entry(slide="E:/a.svs"))
        project.save()
        good = project.path.read_text(encoding="utf-8")

        import pathlib

        def explode(self, target):
            raise OSError("disk full")

        monkeypatch.setattr(pathlib.Path, "replace", explode)
        project.add(an_entry(slide="E:/b.svs"))
        with pytest.raises(ProjectError):
            project.save()
        assert project.path.read_text(encoding="utf-8") == good

    def test_junk_is_refused_clearly(self, tmp_path):
        path = tmp_path / f"bad{PROJECT_SUFFIX}"
        path.write_text("not json at all", encoding="utf-8")
        with pytest.raises(ProjectError, match="not a readable project"):
            Project.load(path)

    def test_some_other_json_is_refused(self, tmp_path):
        path = tmp_path / f"other{PROJECT_SUFFIX}"
        path.write_text('{"hello": 1}', encoding="utf-8")
        with pytest.raises(ProjectError, match="not a PathLearn project"):
            Project.load(path)

    def test_a_newer_format_is_refused_rather_than_half_read(self, tmp_path):
        path = tmp_path / f"future{PROJECT_SUFFIX}"
        path.write_text(json.dumps({"version": 99, "entries": []}),
                        encoding="utf-8")
        with pytest.raises(ProjectError, match="newer PathLearn"):
            Project.load(path)

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(ProjectError, match="Could not open"):
            Project.load(tmp_path / "nope.json")

    def test_a_project_with_nowhere_to_go(self):
        with pytest.raises(ProjectError, match="nowhere to save"):
            Project().save()


class TestCsvExport:
    def test_wide_is_one_row_per_slide(self, project):
        project.add(an_entry(slide="E:/a.svs"))
        project.add(an_entry(slide="E:/b.svs"))
        rows = project.to_csv().splitlines()
        assert rows[0].startswith("Slide,Source,Added,Model,Count")
        assert rows[1].startswith("a.svs,")
        assert rows[2].startswith("b.svs,")

    def test_wide_has_a_percent_column_per_class(self, project):
        project.add(an_entry(slide="E:/a.svs"))
        header = project.to_csv().splitlines()[0]
        assert "PanIN-2 %" in header and "Acinar %" in header
        assert "PanIN-2 mm2" in header

    def test_long_is_one_row_per_slide_and_class(self, project):
        project.add(an_entry(slide="E:/a.svs"))
        rows = project.to_csv(layout="long").splitlines()
        assert rows[0].startswith("Slide,Source,Class,Count")
        assert len(rows) == 3  # header + two classes

    def test_long_omits_classes_the_model_never_had(self, project):
        project.add(an_entry(slide="E:/a.svs", labels=("PanIN-2", "Acinar")))
        project.add(an_entry(slide="E:/b.svs", labels=("PanIN-3",),
                             tiles=[tile(0, 0, label="PanIN-3")]))
        rows = project.to_csv(layout="long").splitlines()[1:]
        by_slide = {}
        for row in rows:
            slide, _source, label = row.split(",")[:3]
            by_slide.setdefault(slide, []).append(label)
        assert by_slide["a.svs"] == ["PanIN-2", "Acinar"]
        assert by_slide["b.svs"] == ["PanIN-3"]

    def test_the_footer_explains_the_denominator(self, project):
        project.add(an_entry(slide="E:/a.svs"))
        text = project.to_csv()
        assert "# What the percentages are of" in text
        assert "# Blank versus zero" in text

    def test_an_unknown_layout_is_refused(self, project):
        with pytest.raises(ProjectError, match="layout"):
            project.to_csv(layout="sideways")

    def test_a_slide_name_with_a_comma_is_quoted(self, project):
        project.add(an_entry(slide="E:/case, second block.svs"))
        assert '"case, second block.svs"' in project.to_csv()


class TestExcelExport:
    def test_it_writes_three_sheets(self, project, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        project.add(an_entry(slide="E:/a.svs"))
        target = project.to_xlsx(tmp_path / "out.xlsx")
        book = openpyxl.load_workbook(target)
        assert book.sheetnames == ["Composition", "Detail", "About"]

    def test_numbers_go_in_as_numbers_not_text(self, project, tmp_path):
        """So the columns can be averaged and charted without a re-import."""
        openpyxl = pytest.importorskip("openpyxl")
        project.add(an_entry(slide="E:/a.svs"))
        book = openpyxl.load_workbook(project.to_xlsx(tmp_path / "out.xlsx"))
        sheet = book["Composition"]
        header = [c.value for c in sheet[1]]
        value = sheet.cell(row=2, column=header.index("PanIN-2 %") + 1).value
        assert isinstance(value, (int, float))
        assert value == pytest.approx(75.0)

    def test_a_class_the_model_never_had_is_an_empty_cell(self, project,
                                                          tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        project.add(an_entry(slide="E:/a.svs", labels=("PanIN-2", "Acinar")))
        project.add(an_entry(slide="E:/b.svs", labels=("PanIN-3",),
                             tiles=[tile(0, 0, label="PanIN-3")]))
        book = openpyxl.load_workbook(project.to_xlsx(tmp_path / "out.xlsx"))
        sheet = book["Composition"]
        header = [c.value for c in sheet[1]]
        column = header.index("PanIN-3 %") + 1
        assert sheet.cell(row=2, column=column).value is None
        assert sheet.cell(row=3, column=column).value == pytest.approx(100.0)

    def test_a_class_found_none_is_a_real_zero(self, project, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        project.add(an_entry(slide="E:/a.svs", tiles=[tile(0, 0)],
                             labels=("PanIN-2", "Acinar")))
        book = openpyxl.load_workbook(project.to_xlsx(tmp_path / "out.xlsx"))
        sheet = book["Composition"]
        header = [c.value for c in sheet[1]]
        assert sheet.cell(row=2,
                          column=header.index("Acinar %") + 1).value == 0.0

    def test_percentages_are_formatted_not_rounded(self, project, tmp_path):
        """Excel should show 67.42 while the cell still holds 67.4157...

        Rounding on write would quietly destroy precision that a later
        average or chart depends on; a display format does not.
        """
        openpyxl = pytest.importorskip("openpyxl")
        project.add(an_entry(slide="E:/a.svs",
                             tiles=[tile(c, 0) for c in range(3)]
                                   + [tile(c, 1, label="Acinar")
                                      for c in range(5)]))
        book = openpyxl.load_workbook(project.to_xlsx(tmp_path / "out.xlsx"))
        sheet = book["Composition"]
        header = [c.value for c in sheet[1]]
        cell = sheet.cell(row=2, column=header.index("PanIN-2 %") + 1)
        assert cell.number_format == "0.00"
        assert cell.value == pytest.approx(37.5)

    def test_the_about_sheet_carries_the_caveats(self, project, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        project.add(an_entry(slide="E:/a.svs"))
        book = openpyxl.load_workbook(project.to_xlsx(tmp_path / "out.xlsx"))
        text = "\n".join(str(c.value) for row in book["About"] for c in row)
        assert "NOT the slide" in text
        assert "Do not fill blanks with zeros" in text

    def test_the_mixed_extractor_warning_reaches_the_workbook(self, project,
                                                              tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        project.add(an_entry(slide="E:/a.svs", extractor="onnx:uni2-h:r1"))
        project.add(an_entry(slide="E:/b.svs", extractor="onnx:phikon-v1:r1"))
        book = openpyxl.load_workbook(project.to_xlsx(tmp_path / "out.xlsx"))
        text = "\n".join(str(c.value) for row in book["About"] for c in row)
        assert "mixed extractors" in text
