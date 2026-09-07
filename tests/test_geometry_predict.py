"""Grading traced annotations with a geometry model.

A geometry model's input is a traced outline, so its inference path is one
verdict per annotation, not a per-tile heatmap. These cover the pipeline, the
verdict table, and the guard that keeps a geometry model out of the Predict
sheet — which used to enable Run and then fail with "Extractor failed to load".
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from pathlearn.core.shape import SHAPE_DIMENSION, SHAPE_VERSION
from pathlearn.core.geometry import DIMENSION, VERSION
from pathlearn.data.geometry_bank import FeatureSource, GeometryBank, GeometryRecord
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.classification import ClassificationProfile
from pathlearn.models.classifier import MLClassifier
from pathlearn.models.store import AnnotationStore
from pathlearn.pipeline.geometry_predict import (Grade, grade_annotations, source_of)
from pathlearn.pipeline.geometry_train import (GeometryError, GeometryTrainingSettings,
                                               Validation, train_geometry_classifier)
from pathlearn.ui.sheets.grade_results import GradeResultsSheet
from synthetic_slide import write_synthetic_slide

CLASSES = ("PanIN-1a", "PanIN-2")


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("grade") / "g.tif"
    write_synthetic_slide(path, 4096, 3072, levels=4)
    with SlideImage(path) as s:
        yield s


def stock(bank, source=FeatureSource.SHAPE, per_class=10, slides=4):
    """Separable records over several slides, so leave-one-slide-out works."""
    rng = np.random.default_rng(0)
    rows = []
    for i, name in enumerate(CLASSES):
        for j in range(per_class):
            shape = texture = None
            if source in (FeatureSource.SHAPE, FeatureSource.COMBINED):
                shape = rng.normal(i * 4.0, 0.3, SHAPE_DIMENSION).astype(np.float32)
            if source in (FeatureSource.TEXTURE, FeatureSource.COMBINED):
                texture = rng.normal(i * 4.0, 0.3, DIMENSION).astype(np.float32)
            rows.append(GeometryRecord(
                slide_path=f"S{j % slides}.svs", slide_name=f"S{j % slides}.svs",
                annotation_id=uuid.uuid4(), classification=name,
                features=texture, shape_features=shape,
                version=VERSION, shape_version=SHAPE_VERSION))
    bank.add(rows)
    return bank


@pytest.fixture
def shape_model(tmp_path):
    bank = stock(GeometryBank(tmp_path / "g.json"))
    return train_geometry_classifier(bank=bank, settings=GeometryTrainingSettings(
        source=FeatureSource.SHAPE, class_labels=list(CLASSES),
        validation=Validation.BY_SLIDE, folds=3))


def blob(x, y, radius=300, sides=24, cls="PanIN-1a"):
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return Annotation(
        points=[Point(x + radius * np.cos(a), y + radius * np.sin(a)) for a in angles],
        classification=cls, color=AnnotationColor.default())


class TestSourceOf:
    def test_reads_the_trained_source(self, shape_model):
        assert source_of(shape_model) is FeatureSource.SHAPE

    def test_survives_a_save_and_load(self, shape_model, tmp_path):
        path = tmp_path / "m.cl"
        shape_model.save(path)
        assert source_of(MLClassifier.load(path)) is FeatureSource.SHAPE

    def test_combined_is_recovered(self, tmp_path):
        bank = stock(GeometryBank(tmp_path / "c.json"), FeatureSource.COMBINED)
        model = train_geometry_classifier(bank=bank, settings=GeometryTrainingSettings(
            source=FeatureSource.COMBINED, class_labels=list(CLASSES),
            validation=Validation.BY_SLIDE))
        assert source_of(model) is FeatureSource.COMBINED
        assert model.feature_dim == SHAPE_DIMENSION + DIMENSION

    def test_falls_back_to_width_when_unstamped(self, shape_model):
        """A model written before the source was recorded still resolves."""
        shape_model.aggregation = None
        assert source_of(shape_model) is FeatureSource.SHAPE

    def test_rejects_a_patch_model(self):
        patch_model = MLClassifier(class_labels=["a", "b"],
                                   weights=np.zeros((768, 2), dtype=np.float32),
                                   biases=np.zeros(2, dtype=np.float32),
                                   extractor_identity="onnx:phikon-v1:r1")
        with pytest.raises(GeometryError, match="not a geometry model"):
            source_of(patch_model)


class TestGrading:
    def test_every_annotation_gets_a_verdict(self, slide, shape_model):
        annotations = [blob(800, 800), blob(2400, 1600)]
        report = grade_annotations(slide, annotations, shape_model)
        assert len(report.graded) == 2
        assert all(g.predicted in CLASSES for g in report.graded)

    def test_probabilities_cover_every_class_and_sum_to_one(self, slide, shape_model):
        report = grade_annotations(slide, [blob(800, 800)], shape_model)
        grade = report.graded[0]
        assert set(grade.probabilities) == set(CLASSES)
        assert sum(grade.probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    def test_confidence_is_the_winning_probability(self, slide, shape_model):
        grade = grade_annotations(slide, [blob(800, 800)], shape_model).graded[0]
        assert grade.confidence == pytest.approx(max(grade.probabilities.values()))
        assert grade.predicted == max(grade.probabilities, key=grade.probabilities.get)

    def test_matches_predicting_from_the_bank_vector(self, slide, shape_model, tmp_path):
        """Grading must produce the same vector the trainer would have banked."""
        from pathlearn.pipeline.geometry_train import describe_annotations

        annotation = blob(800, 800)
        bank = GeometryBank(tmp_path / "compare.json")
        describe_annotations(slide, [annotation], bank)
        banked, _ = bank.matrix(bank.records, FeatureSource.SHAPE)
        expected, _ = shape_model.predict(banked[0])

        grade = grade_annotations(slide, [annotation], shape_model).graded[0]
        assert grade.predicted == expected

    def test_degenerate_outline_is_skipped_not_guessed(self, slide, shape_model):
        flat = Annotation(points=[Point(0, 0), Point(100, 0), Point(200, 0)],
                          classification="PanIN-1a", color=AnnotationColor.default())
        report = grade_annotations(slide, [flat], shape_model)
        assert report.graded == []
        assert "degenerate" in report.skipped[0].skipped

    def test_subtractive_regions_are_not_graded(self, slide, shape_model):
        hole = blob(800, 800)
        hole.is_subtractive = True
        assert grade_annotations(slide, [hole], shape_model).grades == []

    def test_nothing_is_written_back(self, slide, shape_model, tmp_path):
        store = AnnotationStore()
        store.bind(tmp_path / "s.svs", slide.dimensions.height)
        store.add(blob(800, 800, cls="PanIN-2"))
        grade_annotations(slide, store.annotations, shape_model)
        assert store.annotations[0].classification == "PanIN-2"

    def test_progress_is_reported_per_annotation(self, slide, shape_model):
        seen = []
        grade_annotations(slide, [blob(800, 800), blob(2400, 1600)], shape_model,
                          progress=lambda d, t, m: seen.append((d, t)))
        assert seen[0] == (0, 2)
        assert seen[-1] == (2, 2)

    def test_cancelling_stops_early(self, slide, shape_model):
        report = grade_annotations(slide, [blob(800, 800), blob(2400, 1600)],
                                   shape_model, should_cancel=lambda: True)
        assert report.cancelled and report.grades == []

    def test_texture_model_skips_what_it_cannot_measure(self, slide, tmp_path):
        """A tiny annotation has too few nuclei; that must be a reason, not a guess."""
        from pathlearn.core.geometry import GeometryConfig

        bank = stock(GeometryBank(tmp_path / "t.json"), FeatureSource.TEXTURE)
        model = train_geometry_classifier(bank=bank, settings=GeometryTrainingSettings(
            source=FeatureSource.TEXTURE, class_labels=list(CLASSES),
            validation=Validation.BY_SLIDE))
        tiny = blob(800, 800, radius=12, sides=8)
        report = grade_annotations(slide, [tiny], model,
                                   GeometryConfig(min_nuclei=500))
        assert report.graded == []
        assert "nuclei" in report.skipped[0].skipped

    def test_summary_counts_the_changes(self, slide, shape_model):
        report = grade_annotations(slide, [blob(800, 800, cls="PanIN-1a"),
                                           blob(2400, 1600, cls="PanIN-2")],
                                   shape_model)
        assert "Graded 2 of 2" in report.summary()
        assert "would change" in report.summary()

    def test_agreement_is_none_when_nothing_graded(self, slide, shape_model):
        flat = Annotation(points=[Point(0, 0), Point(10, 0), Point(20, 0)],
                          classification="PanIN-1a", color=AnnotationColor.default())
        assert grade_annotations(slide, [flat], shape_model).agreement is None


def make_report(*rows):
    from pathlearn.pipeline.geometry_predict import GradeReport
    return GradeReport(source=FeatureSource.SHAPE, class_labels=list(CLASSES),
                       grades=list(rows))


class TestGradeResultsSheet:
    @pytest.fixture
    def store(self, tmp_path, slide):
        s = AnnotationStore()
        s.bind(tmp_path / "sheet.svs", slide.dimensions.height)
        s.add(blob(800, 800, cls="PanIN-1a"))
        s.add(blob(2400, 1600, cls="PanIN-1a"))
        return s

    @pytest.fixture
    def sheet(self, qtbot, store):
        first, second = store.annotations
        report = make_report(
            Grade(annotation_id=first.id, display_name="A", annotated_as="PanIN-1a",
                  predicted="PanIN-2", confidence=0.91,
                  probabilities={"PanIN-1a": 0.09, "PanIN-2": 0.91}),
            Grade(annotation_id=second.id, display_name="B", annotated_as="PanIN-1a",
                  predicted="PanIN-1a", confidence=0.4,
                  probabilities={"PanIN-1a": 0.6, "PanIN-2": 0.4}),
            Grade(annotation_id=uuid.uuid4(), display_name="C",
                  annotated_as="PanIN-2", skipped="outline is degenerate"),
        )
        widget = GradeResultsSheet(report, store, ClassificationProfile.default())
        qtbot.addWidget(widget)
        yield widget
        widget.reject()

    def test_one_row_per_annotation(self, sheet):
        assert sheet.table.rowCount() == 3

    def test_changes_only_filter(self, sheet):
        sheet.changes_only.setChecked(True)
        assert sheet.table.rowCount() == 1

    def test_apply_button_counts_the_changes(self, sheet):
        assert sheet.apply_button.isEnabled()
        assert "(1)" in sheet.apply_button.text()

    def test_no_changes_means_nothing_to_apply(self, qtbot, store):
        report = make_report(Grade(annotation_id=store.annotations[0].id,
                                   display_name="A", annotated_as="PanIN-1a",
                                   predicted="PanIN-1a", confidence=0.8))
        widget = GradeResultsSheet(report, store, ClassificationProfile.default())
        qtbot.addWidget(widget)
        assert not widget.apply_button.isEnabled()
        widget.reject()

    def test_skipped_rows_carry_their_reason(self, sheet):
        reasons = [sheet.table.item(r, 3).text() for r in range(3)]
        assert any("degenerate" in text for text in reasons)

    def test_confidence_sorts_numerically(self, sheet):
        values = [sheet.table.item(r, 3).data(Qt_display())
                  for r in range(sheet.table.rowCount())]
        numbers = [v for v in values if isinstance(v, (int, float))]
        assert 0.91 in numbers and 0.4 in numbers

    def test_applying_reclassifies_only_the_changed_ones(self, sheet, store):
        changes = [g for g in sheet.report.grades if g.changed]
        assert sheet.apply_now(changes) == 1
        assert store.annotations[0].classification == "PanIN-2"
        assert store.annotations[1].classification == "PanIN-1a"

    def test_applying_takes_the_colour_from_the_profile(self, qtbot, store):
        profile = ClassificationProfile.default()
        target = profile.classes[1]
        report = make_report(Grade(annotation_id=store.annotations[0].id,
                                   display_name="A", annotated_as="PanIN-1a",
                                   predicted=target.name, confidence=0.9))
        widget = GradeResultsSheet(report, store, profile)
        qtbot.addWidget(widget)
        widget.apply_now([g for g in report.grades if g.changed])
        assert store.annotations[0].color == target.color
        widget.reject()

    def test_an_unknown_class_keeps_the_existing_colour(self, qtbot, store):
        before = store.annotations[0].color
        report = make_report(Grade(annotation_id=store.annotations[0].id,
                                   display_name="A", annotated_as="PanIN-1a",
                                   predicted="NotInTheProfile", confidence=0.9))
        widget = GradeResultsSheet(report, store, ClassificationProfile.default())
        qtbot.addWidget(widget)
        widget.apply_now([g for g in report.grades if g.changed])
        assert store.annotations[0].color == before
        assert store.annotations[0].classification == "NotInTheProfile"
        widget.reject()

    def test_applying_survives_a_deleted_annotation(self, sheet, store):
        store.remove(store.annotations[0].id)
        assert sheet.apply_now([g for g in sheet.report.grades if g.changed]) == 0


def Qt_display():
    from PySide6.QtCore import Qt
    return Qt.ItemDataRole.DisplayRole


class TestPredictSheetRefusesGeometryModels:
    def test_run_is_disabled_and_explained(self, qtbot, slide, shape_model, tmp_path):
        from pathlearn.extractors.registry import ExtractorRegistry
        from pathlearn.ui.sheets.predict import PredictSheet

        store = AnnotationStore()
        store.bind(tmp_path / "p.svs", slide.dimensions.height)
        store.add(blob(800, 800))

        sheet = PredictSheet(slide, store, ExtractorRegistry([tmp_path]),
                             ClassificationProfile.default(), classifier=shape_model)
        qtbot.addWidget(sheet)
        assert not sheet.run_button.isEnabled()
        assert "Geometry panel" in sheet.model_note.text()
        # isVisible() is False while the dialog itself is unshown; ask relative
        # to the parent instead.
        assert sheet.model_note.isVisibleTo(sheet)
        sheet.reject()


class TestGeometryPanelGrading:
    """The panel's own wiring: what enables Grade, and what it emits."""

    @pytest.fixture
    def panel(self, qtbot, tmp_path, slide):
        from pathlearn.ui.panels.geometry_panel import GeometryPanel

        store = AnnotationStore()
        store.bind(tmp_path / "panel.svs", slide.dimensions.height)
        store.add(blob(800, 800))
        widget = GeometryPanel(GeometryBank(tmp_path / "panel.json"))
        qtbot.addWidget(widget)
        widget.set_slide(slide, store)
        yield widget
        widget.shutdown()

    def test_disabled_without_a_model(self, panel):
        assert not panel.grade_button.isEnabled()
        assert "No model" in panel.grade_model_label.text()

    def test_a_trained_model_enables_grading(self, panel, shape_model):
        panel._on_trained(shape_model)
        assert panel.grade_button.isEnabled()
        assert "Grade 1 Checked Annotation(s)" == panel.grade_button.text()

    def test_the_label_names_the_source_and_classes(self, panel, shape_model):
        panel._on_trained(shape_model)
        text = panel.grade_model_label.text()
        assert "Outline shape" in text
        assert "PanIN-1a" in text and "PanIN-2" in text

    def test_unchecking_everything_disables_grading(self, panel, shape_model):
        panel._on_trained(shape_model)
        panel.store.select_all(False)
        panel.annotations_changed()
        assert not panel.grade_button.isEnabled()

    def test_grading_emits_the_report(self, qtbot, panel, shape_model):
        panel._on_trained(shape_model)
        with qtbot.waitSignal(panel.grades_ready, timeout=30_000) as caught:
            panel._grade()
        report = caught.args[0]
        assert len(report.graded) == 1
        assert panel.last_grades is report

    def test_a_patch_model_is_refused_on_load(self, panel, tmp_path):
        patch_model = MLClassifier(class_labels=["a", "b"],
                                   weights=np.zeros((768, 2), dtype=np.float32),
                                   biases=np.zeros(2, dtype=np.float32),
                                   extractor_identity="onnx:phikon-v1:r1")
        path = tmp_path / "patch.cl"
        patch_model.save(path)
        panel._load_model_from(path)
        assert panel.model is None
        assert "not a geometry model" in panel.status.text()
