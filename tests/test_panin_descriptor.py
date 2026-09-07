"""The PanIN architecture descriptor: polarity and inner-lumen shape."""

from __future__ import annotations

import math
import uuid

import numpy as np
import pytest

from pathlearn.core.geometry import GeometryConfig
from pathlearn.core.panin import (PANIN_DESCRIPTOR_NAMES, PANIN_DIMENSION,
                                  PANIN_VERSION, _branch_index, _elongation,
                                  _interior_fraction, _solidity, describe_panin)
from pathlearn.data.geometry_bank import (FeatureSource, GeometryBank,
                                          GeometryRecord)
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from synthetic_slide import write_synthetic_slide


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("panin") / "p.tif"
    write_synthetic_slide(path, 4096, 3072, levels=4)
    with SlideImage(path) as s:
        yield s


def blob(x, y, radius=500, sides=32, cls="PanIN-2"):
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return Annotation(
        points=[Point(x + radius * math.cos(a), y + radius * math.sin(a))
                for a in angles],
        classification=cls, color=AnnotationColor.default())


class TestShape:
    def test_the_vector_has_the_documented_width(self):
        assert PANIN_DIMENSION == len(PANIN_DESCRIPTOR_NAMES) == 17

    def test_names_are_unique(self):
        assert len(set(PANIN_DESCRIPTOR_NAMES)) == len(PANIN_DESCRIPTOR_NAMES)

    def test_it_produces_that_width(self, slide):
        features = describe_panin(slide, blob(1200, 1200))
        assert features is not None
        assert features.shape == (PANIN_DIMENSION,)
        assert features.dtype == np.float32

    def test_every_value_is_finite(self, slide):
        features = describe_panin(slide, blob(1200, 1200))
        assert np.all(np.isfinite(features))

    def test_fractions_stay_in_range(self, slide):
        values = dict(zip(PANIN_DESCRIPTOR_NAMES,
                          describe_panin(slide, blob(1200, 1200))))
        for name in ("nucleiAtLumen", "lumenFraction", "lumenInteriorFraction",
                     "lumenCircularity", "lumenSolidity", "tissueFraction"):
            assert 0.0 <= values[name] <= 1.0, f"{name} = {values[name]}"


class TestRefusals:
    def test_a_two_point_outline(self, slide):
        thin = Annotation(points=[Point(0, 0), Point(10, 0)],
                          classification="x", color=AnnotationColor.default())
        assert describe_panin(slide, thin) is None

    def test_a_zero_area_outline(self, slide):
        line = Annotation(points=[Point(10, 10), Point(200, 10), Point(300, 10)],
                          classification="x", color=AnnotationColor.default())
        assert describe_panin(slide, line) is None

    def test_a_region_smaller_than_the_analysis_grid(self, slide):
        assert describe_panin(slide, blob(1200, 1200, radius=2, sides=8)) is None

    def test_it_never_returns_zeros_instead_of_none(self, slide):
        """A zero vector is indistinguishable from a real empty measurement."""
        tiny = blob(1200, 1200, radius=1, sides=6)
        assert describe_panin(slide, tiny) is None


class TestScaleInvariance:
    """Size is the confound this descriptor exists to avoid."""

    def test_lumen_shape_does_not_change_with_region_size(self, slide):
        """Both radii must actually contain a lumen, or this proves nothing.

        The synthetic slide is not a duct, so a small enough circle lands on
        solid tissue and every lumen feature is legitimately zero. Only the
        *shape* of a found lumen is compared — its area fraction depends on
        how much background the circle happens to enclose, which is a property
        of this fixture rather than of the descriptor.
        """
        names = list(PANIN_DESCRIPTOR_NAMES)
        small = describe_panin(slide, blob(1400, 1200, radius=600))
        large = describe_panin(slide, blob(1400, 1200, radius=900))
        assert small is not None and large is not None
        assert small[names.index("lumenFraction")] > 0
        assert large[names.index("lumenFraction")] > 0

        for name in ("lumenSolidity", "lumenCircularity"):
            a, b = small[names.index(name)], large[names.index(name)]
            assert abs(a - b) < 0.2, f"{name}: {a:.3f} vs {b:.3f}"

    def test_a_region_of_solid_tissue_reports_no_lumen(self, slide):
        """Zero here is a real measurement, not a failure."""
        names = list(PANIN_DESCRIPTOR_NAMES)
        features = describe_panin(slide, blob(1400, 1200, radius=300))
        assert features is not None
        assert features[names.index("lumenFraction")] == 0.0
        assert features[names.index("tissueFraction")] == pytest.approx(1.0, abs=0.01)

    def test_no_feature_is_a_raw_count(self):
        """Counts scale with area; the study showed that wrecks the signal."""
        for name in PANIN_DESCRIPTOR_NAMES:
            assert not name.endswith("Count"), name
        assert "lumenCountPerArea" in PANIN_DESCRIPTOR_NAMES


class TestHelpers:
    def test_interior_fraction_high_for_a_disc(self):
        y, x = np.mgrid[0:80, 0:80]
        disc = (x - 40) ** 2 + (y - 40) ** 2 < 30 ** 2
        assert _interior_fraction(disc) > 0.8

    def test_interior_fraction_low_for_a_slit(self):
        slit = np.zeros((80, 80), dtype=bool)
        slit[38:42, 5:75] = True
        assert _interior_fraction(slit) < 0.5

    def test_solidity_is_one_for_a_square(self):
        square = np.zeros((60, 60), dtype=bool)
        square[10:50, 10:50] = True
        assert _solidity(square) == pytest.approx(1.0, abs=0.1)

    def test_solidity_is_lower_for_a_cross(self):
        cross = np.zeros((60, 60), dtype=bool)
        cross[25:35, 5:55] = True
        cross[5:55, 25:35] = True
        assert _solidity(cross) < 0.6

    def test_elongation_is_one_for_a_disc(self):
        y, x = np.mgrid[0:80, 0:80]
        disc = (x - 40) ** 2 + (y - 40) ** 2 < 25 ** 2
        assert _elongation(disc) == pytest.approx(1.0, abs=0.15)

    def test_elongation_grows_with_a_bar(self):
        bar = np.zeros((80, 80), dtype=bool)
        bar[38:42, 5:75] = True
        assert _elongation(bar) > 4

    def test_branch_index_separates_round_from_stellate(self):
        """The PanIN-1a versus 1b distinction, which nuclei do not carry."""
        y, x = np.mgrid[0:120, 0:120]
        disc = (x - 60) ** 2 + (y - 60) ** 2 < 35 ** 2

        star = np.zeros((120, 120), dtype=bool)
        for angle in range(0, 360, 72):
            for r in range(0, 50):
                cx = int(60 + r * math.cos(math.radians(angle)))
                cy = int(60 + r * math.sin(math.radians(angle)))
                star[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4] = True

        assert _branch_index(star) > _branch_index(disc)


class TestBankIntegration:
    def make(self, panin=True, version=PANIN_VERSION):
        return GeometryRecord(
            slide_path="S.svs", slide_name="S.svs", annotation_id=uuid.uuid4(),
            classification="PanIN-2",
            panin_features=(np.ones(PANIN_DIMENSION, dtype=np.float32)
                            if panin else None),
            panin_version=version)

    def test_the_source_exists_with_the_right_width(self):
        assert FeatureSource.PANIN.dimension == PANIN_DIMENSION
        assert FeatureSource.PANIN.label == "PanIN architecture"

    def test_has_panin(self):
        assert self.make().has_panin
        assert not self.make(panin=False).has_panin

    def test_a_stale_version_is_rejected(self):
        """Recompute, never migrate — the study's numbers depend on semantics."""
        assert not self.make(version=PANIN_VERSION - 1).has_panin

    def test_a_panin_only_record_is_current(self):
        assert self.make().is_current

    def test_it_round_trips_through_json(self, tmp_path):
        bank = GeometryBank(tmp_path / "g.json")
        bank.add([self.make()])
        reloaded = GeometryBank(tmp_path / "g.json")
        assert len(reloaded) == 1
        assert reloaded.records[0].has_panin
        assert reloaded.records[0].panin_features.size == PANIN_DIMENSION

    def test_usable_for_selects_it(self, tmp_path):
        bank = GeometryBank(tmp_path / "g.json")
        bank.add([self.make(), self.make(panin=False)])
        assert len(bank.usable_for(FeatureSource.PANIN)) == 1

    def test_the_matrix_has_the_right_width(self, tmp_path):
        bank = GeometryBank(tmp_path / "g.json")
        bank.add([self.make(), self.make()])
        matrix, labels = bank.matrix(bank.usable_for(FeatureSource.PANIN),
                                     FeatureSource.PANIN)
        assert matrix.shape == (2, PANIN_DIMENSION)
        assert labels == ["PanIN-2", "PanIN-2"]

    def test_stats_count_it(self, tmp_path):
        bank = GeometryBank(tmp_path / "g.json")
        bank.add([self.make(), self.make(panin=False)])
        assert bank.stats().with_panin == 1
        assert "PanIN architecture" in bank.stats().summary()


class TestPipeline:
    def test_describe_annotations_fills_the_block(self, slide, tmp_path):
        from pathlearn.pipeline.geometry_train import describe_annotations

        bank = GeometryBank(tmp_path / "g.json")
        report = describe_annotations(slide, [blob(1200, 1200)], bank)
        assert report.with_panin == 1
        assert bank.records[0].has_panin

    def test_the_summary_mentions_it(self, slide, tmp_path):
        from pathlearn.pipeline.geometry_train import describe_annotations

        bank = GeometryBank(tmp_path / "g.json")
        report = describe_annotations(slide, [blob(1200, 1200)], bank)
        assert "PanIN architecture" in report.summary()

    def test_a_config_with_a_coarser_target_still_works(self, slide):
        features = describe_panin(slide, blob(1200, 1200),
                                  GeometryConfig(target_mpp=2.0))
        assert features is not None and np.all(np.isfinite(features))


class TestMissingBlockIsExplained:
    """A full bank with an empty class list must say why.

    The reported confusion: 41 records in the bank, nothing selectable under
    PanIN architecture, and no indication that those records simply predate
    the block.
    """

    @pytest.fixture
    def panel(self, qtbot, tmp_path):
        from pathlearn.ui.panels.geometry_panel import GeometryPanel

        bank = GeometryBank(tmp_path / "g.json")
        # Ten records with shape only, one with everything.
        from pathlearn.core.shape import SHAPE_DIMENSION, SHAPE_VERSION
        rows = [GeometryRecord(
            slide_path=f"S{i % 3}.svs", slide_name=f"S{i % 3}.svs",
            annotation_id=uuid.uuid4(),
            classification="PanIN-1a" if i % 2 else "PanIN-2",
            shape_features=np.zeros(SHAPE_DIMENSION, dtype=np.float32),
            shape_version=SHAPE_VERSION) for i in range(10)]
        rows.append(GeometryRecord(
            slide_path="S9.svs", slide_name="S9.svs", annotation_id=uuid.uuid4(),
            classification="PanIN-3",
            shape_features=np.zeros(SHAPE_DIMENSION, dtype=np.float32),
            shape_version=SHAPE_VERSION,
            panin_features=np.zeros(PANIN_DIMENSION, dtype=np.float32)))
        bank.add(rows)
        widget = GeometryPanel(bank)
        qtbot.addWidget(widget)
        yield widget
        widget.shutdown()

    def _select(self, panel, source):
        panel.source_combo.setCurrentIndex(
            next(i for i in range(panel.source_combo.count())
                 if panel.source_combo.itemData(i) is source))

    def test_shape_lists_every_class(self, panel):
        """All eleven records carry shape: PanIN-1a, PanIN-2 and PanIN-3."""
        self._select(panel, FeatureSource.SHAPE)
        assert panel.class_list.count() == 3

    def test_panin_lists_only_what_carries_the_block(self, panel):
        self._select(panel, FeatureSource.PANIN)
        assert panel.class_list.count() == 1

    def test_the_shortfall_is_explained(self, panel):
        self._select(panel, FeatureSource.PANIN)
        text = panel.status.text()
        assert "10 record(s) do not carry" in text
        assert "Describe those slides again" in text

    def test_no_complaint_when_nothing_is_missing(self, panel):
        self._select(panel, FeatureSource.SHAPE)
        assert "do not carry" not in panel.status.text()

    def test_training_is_blocked_with_one_class(self, panel):
        self._select(panel, FeatureSource.PANIN)
        assert not panel.train_button.isEnabled()
        assert "tick at least 2 classes" in panel.status.text()


class TestGradingWithEverySource:
    """Every FeatureSource must have a grading path.

    The reported bug: grading with a PanIN model raised "zero-dimensional
    arrays cannot be concatenated". PANIN had been added to the bank and the
    trainer but not to `_vector_for`, so both blocks stayed None and fell
    through to a concatenate of two Nones — an error naming numpy internals
    rather than the actual cause.
    """

    def model_for(self, source, tmp_path):
        from pathlearn.core.geometry import DIMENSION
        from pathlearn.core.shape import SHAPE_DIMENSION, SHAPE_VERSION
        from pathlearn.core.geometry import VERSION
        from pathlearn.pipeline.geometry_train import (GeometryTrainingSettings,
                                                       Validation,
                                                       train_geometry_classifier)

        bank = GeometryBank(tmp_path / f"{source.value}.json")
        rng = np.random.default_rng(0)
        rows = []
        for i, name in enumerate(("PanIN-1a", "PanIN-2")):
            for j in range(8):
                rows.append(GeometryRecord(
                    slide_path=f"S{j % 4}.svs", slide_name=f"S{j % 4}.svs",
                    annotation_id=uuid.uuid4(), classification=name,
                    features=rng.normal(i * 4, .3, DIMENSION).astype(np.float32),
                    shape_features=rng.normal(i * 4, .3,
                                              SHAPE_DIMENSION).astype(np.float32),
                    panin_features=rng.normal(i * 4, .3,
                                              PANIN_DIMENSION).astype(np.float32),
                    version=VERSION, shape_version=SHAPE_VERSION,
                    panin_version=PANIN_VERSION))
        bank.add(rows)
        return train_geometry_classifier(bank=bank, settings=GeometryTrainingSettings(
            source=source, class_labels=["PanIN-1a", "PanIN-2"],
            validation=Validation.BY_SLIDE))

    @pytest.mark.parametrize("source", list(FeatureSource))
    def test_grading_produces_a_verdict(self, slide, tmp_path, source):
        from pathlearn.pipeline.geometry_predict import grade_annotations, source_of

        model = self.model_for(source, tmp_path)
        assert source_of(model) is source
        report = grade_annotations(slide, [blob(1200, 1200)], model)
        # Either a verdict, or a stated reason — never an exception, and never
        # a numpy message about zero-dimensional arrays.
        assert len(report.grades) == 1
        grade = report.grades[0]
        if grade.predicted is None:
            assert grade.skipped
            assert "concatenat" not in grade.skipped
        else:
            assert grade.predicted in ("PanIN-1a", "PanIN-2")

    def test_the_panin_source_actually_grades(self, slide, tmp_path):
        from pathlearn.pipeline.geometry_predict import grade_annotations

        model = self.model_for(FeatureSource.PANIN, tmp_path)
        report = grade_annotations(slide, [blob(1200, 1200)], model)
        assert report.graded, report.grades[0].skipped
        assert len(report.graded[0].probabilities) == 2

    def test_an_unhandled_source_is_a_loud_bug_not_a_numpy_error(self, slide):
        from pathlearn.pipeline import geometry_predict
        from pathlearn.pipeline.geometry_train import GeometryError

        class Fake:
            value = "fake"
            label = "Fake"
        with pytest.raises(GeometryError, match="no grading path"):
            geometry_predict._vector_for(slide, blob(1200, 1200), Fake(),
                                         GeometryConfig(), 0.5)
