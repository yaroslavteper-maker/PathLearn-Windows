"""Prediction pipeline: aggregation modes, gating, and the overlay model."""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.classifier import MLClassifier
from pathlearn.models.prediction import PatchPrediction, PredictionSet, sidecar_path
from pathlearn.pipeline.predict import (Aggregation, PredictionError, PredictionSettings,
                                        predict_regions)
from synthetic_slide import write_synthetic_slide
from test_extraction_pipeline import FakeExtractor

WIDTH, HEIGHT = 4096, 3072
LABELS = ["Acinar", "PaNIN"]


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("s") / "predict.tif"
    write_synthetic_slide(path, WIDTH, HEIGHT, levels=4)
    with SlideImage(path) as s:
        yield s


#: FakeExtractor emits the patch's mean pixel value, so the decision boundary
#: sits in the middle of the tissue/background range.
_BOUNDARY = 180.0
#: Small enough that the softmax does not saturate. With unit weights the
#: logits reach +/-200 and every tile reports confidence exactly 1.0, which
#: makes any confidence threshold untestable — and is not how a real model on
#: normalised embeddings behaves.
_SCALE = 0.01


def make_classifier(dim=16, identity="test:fake:r1", **kw):
    """A two-class model whose verdict and confidence vary with tile content."""
    weights = np.zeros((dim, 2), dtype=np.float32)
    weights[0, 0] = _SCALE
    weights[0, 1] = -_SCALE
    biases = np.array([-_SCALE * _BOUNDARY, _SCALE * _BOUNDARY], dtype=np.float32)
    defaults = dict(class_labels=list(LABELS), weights=weights, biases=biases,
                    extractor_identity=identity)
    defaults.update(kw)
    return MLClassifier(**defaults)


def region(x0=400, y0=400, size=900, classification="ROI", subtractive=False):
    return Annotation(
        points=[Point(x0, y0), Point(x0 + size, y0),
                Point(x0 + size, y0 + size), Point(x0, y0 + size)],
        classification=classification, color=AnnotationColor.default(),
        is_subtractive=subtractive)


def run(slide, classifier=None, extractor=None, **kw):
    extractor = extractor or FakeExtractor(dim=16)
    classifier = classifier or make_classifier()
    settings = PredictionSettings(**{"max_white_fraction": 1.0, **kw})
    return predict_regions(slide, [region()], extractor, classifier, settings)


class TestPredictionSet:
    def test_counts_and_summary(self):
        probs = np.array([0.9, 0.1], dtype=np.float32)
        ps = PredictionSet(
            predictions=[PatchPrediction(0, 0, 224, "A", probs),
                         PatchPrediction(224, 0, 224, "A", probs),
                         PatchPrediction(0, 224, 224, "B", probs)],
            class_labels=["A", "B"])
        assert ps.counts == {"A": 2, "B": 1}
        assert "3 tiles" in ps.summary()

    def test_hidden_classes_are_filtered(self):
        probs = np.array([0.9, 0.1], dtype=np.float32)
        ps = PredictionSet(predictions=[PatchPrediction(0, 0, 224, "A", probs),
                                        PatchPrediction(0, 224, 224, "B", probs)])
        ps.hidden.add("B")
        assert [p.label for p in ps.visible()] == ["A"]

    def test_confidence_threshold_filters(self):
        ps = PredictionSet(predictions=[
            PatchPrediction(0, 0, 224, "A", np.array([0.95, 0.05], dtype=np.float32)),
            PatchPrediction(0, 224, 224, "A", np.array([0.55, 0.45], dtype=np.float32)),
        ])
        ps.min_confidence = 0.9
        assert len(ps.visible()) == 1

    def test_confidence_is_the_max_probability(self):
        p = PatchPrediction(0, 0, 224, "A", np.array([0.2, 0.7, 0.1], dtype=np.float32))
        assert p.confidence == pytest.approx(0.7)

    def test_centre(self):
        assert PatchPrediction(100, 200, 224, "A", np.array([1.0])).centre == (212.0, 312.0)

    def test_round_trip_through_file(self, tmp_path):
        ps = PredictionSet(
            predictions=[PatchPrediction(10, 20, 224, "A",
                                         np.array([0.8, 0.2], dtype=np.float32))],
            class_labels=["A", "B"],
            colors={"A": AnnotationColor(1, 2, 3)},
            slide_path="E:/x.svs")
        path = tmp_path / "p.json"
        ps.save(path)
        restored = PredictionSet.load(path)
        assert len(restored) == 1
        assert restored.predictions[0].x == 10
        assert restored.color_for("A") == AnnotationColor(1, 2, 3)

    def test_sidecar_path(self):
        assert sidecar_path("E:/a/b.svs").name == "b.predictions.json"


class TestPrediction:
    def test_produces_tiles(self, slide):
        result, report = run(slide)
        assert not result.is_empty
        assert report.predicted == len(result)

    def test_tiles_are_inside_the_region(self, slide):
        result, _ = run(slide)
        target = region()
        for prediction in result.predictions:
            assert target.contains(*prediction.centre)

    def test_size_is_level0(self, slide):
        result, _ = run(slide, patch_size_level=224, level=1)
        # Level 1 is 2x downsampled, so a 224 px tile spans 448 level-0 px.
        assert all(p.size_level0 == 448 for p in result.predictions)

    def test_labels_come_from_the_classifier(self, slide):
        result, _ = run(slide)
        assert set(p.label for p in result.predictions) <= set(LABELS)

    def test_probabilities_sum_to_one(self, slide):
        result, _ = run(slide)
        for prediction in result.predictions:
            assert prediction.probabilities.sum() == pytest.approx(1.0, abs=1e-5)

    def test_white_gate(self, slide):
        _, report = run(slide, max_white_fraction=0.0)
        assert report.skipped_white > 0

    def test_confidence_gate_discards(self, slide):
        loose, _ = run(slide, min_confidence=0.0)
        strict, report = run(slide, min_confidence=0.999)
        assert len(strict) < len(loose)
        assert report.below_confidence > 0

    def test_subtractive_regions_are_carved_out(self, slide):
        extractor, classifier = FakeExtractor(dim=16), make_classifier()
        settings = PredictionSettings(max_white_fraction=1.0)
        hole = region(400, 400, 450, subtractive=True)
        full, _ = predict_regions(slide, [region()], extractor, classifier, settings)
        carved, _ = predict_regions(slide, [region()], extractor, classifier, settings,
                                    subtractive=[hole])
        assert len(carved) < len(full)
        for prediction in carved.predictions:
            assert not hole.contains(*prediction.centre)

    def test_progress_completes(self, slide):
        seen: list[tuple[int, int]] = []
        predict_regions(slide, [region()], FakeExtractor(dim=16), make_classifier(),
                        PredictionSettings(max_white_fraction=1.0),
                        progress=lambda d, t, m: seen.append((d, t)))
        assert seen and seen[-1][0] == seen[-1][1]

    def test_cancellation(self, slide):
        _, report = predict_regions(
            slide, [region(300, 300, 1500)], FakeExtractor(dim=16), make_classifier(),
            PredictionSettings(max_white_fraction=1.0), should_cancel=lambda: True)
        assert report.cancelled

    def test_no_regions(self, slide):
        result, report = predict_regions(slide, [], FakeExtractor(dim=16),
                                         make_classifier(), PredictionSettings())
        assert result.is_empty and report.considered == 0

    def test_subtractive_only_is_not_predicted(self, slide):
        result, _ = predict_regions(slide, [region(subtractive=True)],
                                    FakeExtractor(dim=16), make_classifier(),
                                    PredictionSettings(max_white_fraction=1.0))
        assert result.is_empty


class TestExtractorMismatch:
    def test_wrong_feature_space_is_refused(self, slide):
        """The whole point of the identity stamp."""
        classifier = make_classifier(identity="onnx:uni2-h:r1")
        with pytest.raises(PredictionError, match="not comparable"):
            run(slide, classifier=classifier)

    def test_geometry_model_is_exempt(self, slide):
        classifier = make_classifier(identity="anything", feature_source="geometry")
        result, _ = run(slide, classifier=classifier)
        assert not result.is_empty


class TestAggregation:
    def test_per_patch_allows_several_labels(self, slide):
        result, _ = run(slide, aggregation=Aggregation.PER_PATCH)
        assert len(result.predictions) > 1

    def test_mean_probability_gives_one_verdict(self, slide):
        result, _ = run(slide, aggregation=Aggregation.MEAN_PROBABILITY)
        assert len(set(p.label for p in result.predictions)) == 1
        first = result.predictions[0].probabilities
        assert all(np.allclose(p.probabilities, first) for p in result.predictions)

    def test_max_probability_gives_one_verdict(self, slide):
        result, _ = run(slide, aggregation=Aggregation.MAX_PROBABILITY)
        assert len(set(p.label for p in result.predictions)) == 1

    def test_max_takes_the_most_confident_tile(self, slide):
        per_patch, _ = run(slide, aggregation=Aggregation.PER_PATCH)
        best = max(p.confidence for p in per_patch.predictions)
        maxed, _ = run(slide, aggregation=Aggregation.MAX_PROBABILITY)
        assert maxed.predictions[0].confidence == pytest.approx(best, abs=1e-5)

    def test_pooled_model_overrides_aggregation(self, slide):
        """A pooled model wants one vector per region whatever the setting."""
        pooled = make_classifier(dim=48, aggregation="meanmaxstd")
        result, _ = run(slide, classifier=pooled, extractor=FakeExtractor(dim=16),
                        aggregation=Aggregation.PER_PATCH)
        assert len(set(p.label for p in result.predictions)) == 1


class TestNullGating:
    def test_null_like_tiles_are_dropped(self, slide):
        extractor = FakeExtractor(dim=16)
        plain = make_classifier()
        # FakeExtractor returns a constant vector per patch, so a reference of
        # ones matches everything by cosine.
        gated = make_classifier(null_reference=np.ones((1, 16), dtype=np.float32),
                                null_threshold=0.5)
        loose, _ = run(slide, classifier=plain, extractor=extractor)
        strict, report = run(slide, classifier=gated, extractor=extractor)
        assert report.skipped_null > 0
        assert len(strict) < len(loose)

    def test_lenient_threshold_keeps_tiles(self, slide):
        gated = make_classifier(null_reference=np.ones((1, 16), dtype=np.float32),
                                null_threshold=1.01)
        result, report = run(slide, classifier=gated)
        assert report.skipped_null == 0 and not result.is_empty


class TestSettingsValidation:
    @pytest.mark.parametrize("bad", [
        dict(patch_size_level=0), dict(stride_level=0), dict(min_confidence=1.5),
    ])
    def test_invalid_settings_raise(self, slide, bad):
        with pytest.raises(PredictionError):
            predict_regions(slide, [region()], FakeExtractor(dim=16),
                            make_classifier(), PredictionSettings(**bad))
