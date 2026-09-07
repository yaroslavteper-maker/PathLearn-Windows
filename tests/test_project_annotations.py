"""Filing the annotation class breakdown in a project, beside the predictions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.prediction import PatchPrediction, PredictionSet
from pathlearn.pipeline.annotation_stats import breakdown
from pathlearn.pipeline.composition import composition
from pathlearn.project import (ANNOTATED, PREDICTED, PROJECT_SUFFIX, Project,
                               ProjectEntry)

SIZE = 224
PALETTE = ["PanIN-1a", "PanIN-1b", "PanIN-2", "PanIN-3"]


def square(x, y, side, label="PanIN-2", subtractive=False):
    return Annotation(
        points=[Point(x, y), Point(x + side, y),
                Point(x + side, y + side), Point(x, y + side)],
        classification=label, color=AnnotationColor(200, 60, 60),
        is_subtractive=subtractive)


def tile(col, label="PanIN-2"):
    return PatchPrediction(x=col * SIZE, y=0, size_level0=SIZE, label=label,
                           probabilities=np.array([0.9, 0.0],
                                                  dtype=np.float32))


def a_predicted_entry(slide="E:/a.svs"):
    predictions = PredictionSet(
        predictions=[tile(0), tile(1), tile(2), tile(3, label="Acinar")],
        class_labels=["PanIN-2", "Acinar"], slide_path=slide,
        extractor_identity="onnx:uni2-h:r1")
    return ProjectEntry.from_report(composition(predictions, mpp=0.25),
                                    predictions, model_name="2-class model")


def an_annotated_entry(slide="E:/a.svs", annotations=None, palette=PALETTE,
                       scope="all annotations"):
    annotations = annotations if annotations is not None else [
        square(0, 0, 10, label="PanIN-1a"), square(100, 0, 10,
                                                   label="PanIN-1a"),
        square(200, 0, 100, label="PanIN-3")]
    report = breakdown(annotations, mpp=0.25, scope=scope)
    return ProjectEntry.from_breakdown(report, slide_path=slide,
                                       palette_classes=palette)


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / f"cohort{PROJECT_SUFFIX}", "PanIN cohort")


class TestTheEntry:
    def test_it_is_marked_as_annotated(self):
        entry = an_annotated_entry()
        assert entry.source == ANNOTATED
        assert entry.is_annotated
        assert entry.unit_name == "regions"

    def test_a_predicted_entry_still_counts_tiles(self):
        assert a_predicted_entry().unit_name == "tiles"
        assert a_predicted_entry().source == PREDICTED

    def test_the_count_share_is_carried(self):
        entry = an_annotated_entry()
        assert entry.total_tiles == 3
        assert entry.shares["PanIN-1a"].tile_percent == pytest.approx(200 / 3)

    def test_the_area_share_is_carried_and_differs(self):
        """The whole point of the annotation breakdown survives into the row."""
        entry = an_annotated_entry()
        assert entry.percent_for("PanIN-1a") == pytest.approx(200 / 10200 * 100)
        assert entry.percent_for("PanIN-1a") < 5.0

    def test_the_scope_is_recorded(self):
        entry = an_annotated_entry(scope="annotations checked under Use")
        assert entry.scope == "annotations checked under Use"

    def test_subtractive_bookkeeping_travels(self):
        entry = an_annotated_entry(annotations=[
            square(0, 0, 100), square(10, 10, 20, subtractive=True)])
        assert entry.subtractive_count == 1
        assert entry.carved_px == pytest.approx(400.0)

    def test_regions_carry_no_confidence(self):
        """Nobody scored a region — you drew it."""
        assert all(s.mean_confidence == 0.0
                   for s in an_annotated_entry().shares.values())


class TestThePalette:
    """The palette plays the part a model's class list plays for predictions."""

    def test_a_palette_class_you_drew_none_of_is_a_real_zero(self):
        entry = an_annotated_entry()
        assert entry.percent_for("PanIN-2") == 0.0
        assert entry.percent_for("PanIN-1b") == 0.0

    def test_a_class_outside_the_palette_is_blank(self):
        entry = an_annotated_entry()
        assert entry.percent_for("Acinar") is None

    def test_a_class_you_drew_counts_even_if_it_is_not_in_the_palette(self):
        """Drawing off-palette must not vanish from its own row."""
        entry = an_annotated_entry(
            annotations=[square(0, 0, 10, label="Weird")], palette=PALETTE)
        assert entry.percent_for("Weird") == pytest.approx(100.0)

    def test_no_palette_means_undrawn_classes_stay_blank(self):
        """The safe reading when we cannot know what was on offer."""
        entry = an_annotated_entry(palette=[])
        assert entry.percent_for("PanIN-2") is None
        assert entry.percent_for("PanIN-1a") is not None


class TestTheTwoSourcesCoexist:
    def test_filing_annotations_does_not_replace_the_prediction(self, project):
        """The bug this keying exists to prevent."""
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        assert len(project) == 2
        assert project.entry_for("E:/a.svs", PREDICTED) is not None
        assert project.entry_for("E:/a.svs", ANNOTATED) is not None

    def test_re_adding_replaces_only_its_own_source(self, project):
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        assert project.add(an_annotated_entry()) is True
        assert len(project) == 2
        assert project.entry_for("E:/a.svs", PREDICTED) is not None

    def test_entries_for_returns_both(self, project):
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        assert len(project.entries_for("E:/a.svs")) == 2

    def test_removing_one_source_leaves_the_other(self, project):
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        assert project.remove("E:/a.svs", ANNOTATED) is True
        assert len(project) == 1
        assert project.entries[0].source == PREDICTED

    def test_removing_without_a_source_takes_both(self, project):
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        assert project.remove("E:/a.svs") is True
        assert project.is_empty

    def test_the_summary_counts_slides_and_rows_separately(self, project):
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        assert "1 slide(s), 2 row(s)" in project.summary()

    def test_sources_lists_what_is_present(self, project):
        project.add(an_annotated_entry())
        assert project.sources() == [ANNOTATED]
        project.add(a_predicted_entry())
        assert project.sources() == [PREDICTED, ANNOTATED]

    def test_an_annotated_row_has_no_extractor_to_clash(self, project):
        """It must not trip the mixed-extractor warning on its own."""
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        assert len(project.mixed_extractors()) == 1
        assert "WARNING" not in project.summary()


class TestPersistence:
    def test_the_source_round_trips(self, project):
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        project.save()
        reloaded = Project.load(project.path)
        assert {e.source for e in reloaded.entries} == {PREDICTED, ANNOTATED}

    def test_the_annotation_bookkeeping_round_trips(self, project):
        project.add(an_annotated_entry(annotations=[
            square(0, 0, 100), square(10, 10, 20, subtractive=True)]))
        project.save()
        entry = Project.load(project.path).entries[0]
        assert entry.subtractive_count == 1
        assert entry.carved_px == pytest.approx(400.0)
        assert entry.scope == "all annotations"

    def test_blank_versus_zero_survives(self, project):
        project.add(an_annotated_entry())
        project.save()
        entry = Project.load(project.path).entries[0]
        assert entry.percent_for("PanIN-2") == 0.0
        assert entry.percent_for("Acinar") is None

    def test_a_version_one_project_loads_as_predictions(self, tmp_path):
        """Every pre-existing project is a project of predictions."""
        path = tmp_path / f"old{PROJECT_SUFFIX}"
        payload = {"version": 1, "name": "Old", "entries": [
            {"slidePath": "E:/a.svs", "totalTiles": 12,
             "modelClasses": ["PanIN-2"],
             "shares": {"PanIN-2": {"tiles": 12, "areaPercent": 100.0}}}]}
        path.write_text(json.dumps(payload), encoding="utf-8")
        entry = Project.load(path).entries[0]
        assert entry.source == PREDICTED
        assert entry.unit_name == "tiles"
        assert entry.percent_for("PanIN-2") == pytest.approx(100.0)


class TestExports:
    def both(self, project):
        project.add(a_predicted_entry())
        project.add(an_annotated_entry())
        return project

    def test_the_wide_csv_has_a_source_column(self, project):
        rows = self.both(project).to_csv().splitlines()
        assert rows[0].startswith("Slide,Source,Added,Model,Count")
        assert rows[1].split(",")[1] == "predicted"
        assert rows[2].split(",")[1] == "annotated"

    def test_the_long_csv_leaves_confidence_empty_for_a_drawn_region(self,
                                                                    project):
        """A 0.00 there would read as "no confidence", not "not scored"."""
        rows = self.both(project).to_csv(layout="long").splitlines()
        header = rows[0].split(",")
        column = header.index("Mean confidence")
        annotated = [r for r in rows[1:] if r.split(",")[1] == "annotated"]
        assert annotated
        assert all(r.split(",")[column] == "" for r in annotated)

    def test_the_long_csv_carries_the_annotation_bookkeeping(self, project):
        project.add(an_annotated_entry(annotations=[
            square(0, 0, 100), square(10, 10, 20, subtractive=True)]))
        rows = project.to_csv(layout="long").splitlines()
        header = rows[0].split(",")
        assert "Subtractive excluded" in header
        assert rows[1].split(",")[header.index("Subtractive excluded")] == "1"

    def test_the_footer_explains_the_two_sources(self, project):
        text = self.both(project).to_csv()
        assert "# Source column" in text
        assert "do not average them together" in text

    def test_the_workbook_has_a_source_column(self, project, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        book = openpyxl.load_workbook(
            self.both(project).to_xlsx(tmp_path / "out.xlsx"))
        sheet = book["Composition"]
        header = [c.value for c in sheet[1]]
        assert header[1] == "Source"
        assert {sheet.cell(row=r, column=2).value for r in (2, 3)} == \
            {"predicted", "annotated"}

    def test_the_workbook_percentages_still_line_up(self, project, tmp_path):
        """The Source column shifted every class column along by one."""
        openpyxl = pytest.importorskip("openpyxl")
        book = openpyxl.load_workbook(
            self.both(project).to_xlsx(tmp_path / "out.xlsx"))
        sheet = book["Composition"]
        header = [c.value for c in sheet[1]]
        cell = sheet.cell(row=2, column=header.index("PanIN-2 %") + 1)
        assert cell.value == pytest.approx(75.0)
        assert cell.number_format == "0.00"

    def test_the_about_sheet_warns_against_mixing_sources(self, project,
                                                          tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        book = openpyxl.load_workbook(
            self.both(project).to_xlsx(tmp_path / "out.xlsx"))
        text = "\n".join(str(c.value) for row in book["About"] for c in row)
        assert "do not average them together" in text
        assert "Compare within a source" in text
