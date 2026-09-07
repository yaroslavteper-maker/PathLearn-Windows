"""Classifier serialisation, metrics, and the training pipeline."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from pathlearn.data.bank import Patch, PatchBank
from pathlearn.extractors.identity import LEGACY_VISION
from pathlearn.models.classifier import (AGGREGATION_MEAN_MAX_STD, ClassifierError,
                                         MLClassifier)
from pathlearn.models.metrics import TrainingMetrics, evaluate
from pathlearn.pipeline.train import TrainingError, TrainingSettings, train_classifier

PHIKON = "onnx:phikon-v1:r1"
UNI2 = "onnx:uni2-h:r1"
DIM = 12


@pytest.fixture
def bank(tmp_path):
    with PatchBank(tmp_path / "b.db") as b:
        yield b


def patch(classification, vector, identity=PHIKON, annotation_id=None, **kw):
    defaults = dict(
        slide_path="E:/s/a.svs", slide_name="a.svs",
        annotation_id=annotation_id or uuid.uuid4(),
        classification=classification, patch_x=0, patch_y=0, patch_level=0,
        patch_size_level=224, features=np.asarray(vector, dtype=np.float32),
        extractor_identity=identity, white_fraction=0.1, nucleus_count=20,
    )
    defaults.update(kw)
    return Patch(**defaults)


def direction(index: int, dim: int = DIM, scale: float = 6.0) -> np.ndarray:
    """A vector pointing along axis *index*.

    Class centres must differ in **direction**, not just magnitude: the null
    filter compares cosine similarity, which is scale-invariant, so
    ``full(d, 6)`` and ``full(d, 20)`` are cosine-identical and would be treated
    as the same thing.
    """
    vector = np.zeros(dim, dtype=np.float64)
    vector[index % dim] = scale
    return vector


def fill_bank(bank, per_class=25, dim=DIM, identity=PHIKON, seed=0):
    """Two linearly separable classes on orthogonal axes."""
    rng = np.random.default_rng(seed)
    centres = {"PaNIN-1": direction(0, dim), "PaNIN-3": direction(1, dim)}
    rows = []
    for name, centre in centres.items():
        for _ in range(per_class):
            rows.append(patch(name, centre + rng.normal(0, 0.4, dim), identity))
    bank.add_many(rows)
    return rows


def a_model(**kw):
    defaults = dict(class_labels=["A", "B"],
                    weights=np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32),
                    biases=np.array([0.1, -0.1], dtype=np.float32),
                    extractor_identity=PHIKON)
    defaults.update(kw)
    return MLClassifier(**defaults)


class TestEvaluate:
    def test_perfect(self):
        acc, per_class, confusion = evaluate([0, 1, 0, 1], [0, 1, 0, 1], ["A", "B"])
        assert acc == 1.0
        assert all(c.f1 == 1.0 for c in per_class)
        assert confusion == [[2, 0], [0, 2]]

    def test_confusion_orientation(self):
        """confusion[true][predicted] — getting this backwards silently lies."""
        _, _, confusion = evaluate([0, 0, 0], [1, 1, 1], ["A", "B"])
        assert confusion == [[0, 3], [0, 0]]

    def test_never_predicted_class_scores_zero_not_nan(self):
        _, per_class, _ = evaluate([0, 0], [0, 0], ["A", "B"])
        b = next(c for c in per_class if c.label == "B")
        assert b.precision == 0.0 and b.f1 == 0.0

    def test_support_counts_true_labels(self):
        _, per_class, _ = evaluate([0, 0, 1], [0, 0, 0], ["A", "B"])
        assert {c.label: c.support for c in per_class} == {"A": 2, "B": 1}


class TestMetrics:
    def test_collapse_is_detected(self):
        """The failure mode doc 04 §2 records twice."""
        metrics = TrainingMetrics(
            class_labels=["A", "B", "C"], train_accuracy=0.6, val_accuracy=0.6,
            per_class=[], confusion=[[0, 5, 0], [0, 7, 0], [0, 3, 0]],
            final_loss=1.0, train_count=15, val_count=15)
        assert metrics.is_collapsed
        assert "collapsed" in metrics.summary()

    def test_partial_collapse_is_noted(self):
        metrics = TrainingMetrics(
            class_labels=["A", "B", "C"], train_accuracy=0.6, val_accuracy=0.6,
            per_class=[], confusion=[[3, 2, 0], [1, 6, 0], [0, 3, 0]],
            final_loss=1.0, train_count=15, val_count=15)
        assert not metrics.is_collapsed
        assert metrics.predicted_labels_used == 2
        assert "only 2 of 3" in metrics.summary()

    def test_healthy_model_has_no_warning(self):
        metrics = TrainingMetrics(
            class_labels=["A", "B"], train_accuracy=0.9, val_accuracy=0.85,
            per_class=[], confusion=[[8, 2], [1, 9]], final_loss=0.2,
            train_count=40, val_count=20)
        assert "WARNING" not in metrics.summary() and "NOTE" not in metrics.summary()

    def test_round_trip(self):
        original = TrainingMetrics(
            class_labels=["A", "B"], train_accuracy=0.9, val_accuracy=0.8,
            per_class=[], confusion=[[4, 1], [0, 5]], final_loss=0.3,
            train_count=20, val_count=10, extractor_revision=2)
        restored = TrainingMetrics.from_dict(original.to_dict())
        assert restored.val_accuracy == 0.8
        assert restored.confusion == [[4, 1], [0, 5]]
        assert restored.extractor_revision == 2


class TestClassifierFormat:
    def test_round_trip_through_file(self, tmp_path):
        original = a_model()
        path = tmp_path / "m.cl"
        original.save(path)
        restored = MLClassifier.load(path)
        assert restored.class_labels == original.class_labels
        assert np.allclose(restored.weights, original.weights)
        assert np.allclose(restored.biases, original.biases)
        assert restored.extractor_identity == PHIKON
        assert restored.id == original.id

    def test_schema_matches_macos(self, tmp_path):
        path = tmp_path / "m.cl"
        a_model().save(path)
        doc = json.loads(path.read_text())
        for key in ("version", "id", "createdAt", "classLabels", "featureDim",
                    "classCount", "weightsBase64", "biasesBase64",
                    "featureExtractorRevision", "extractorIdentity", "kind",
                    "aggregation", "featureSource", "featureMean", "featureStd",
                    "nullReference", "nullThreshold", "embeddingSnapshot", "metrics"):
            assert key in doc, f"missing {key}"

    def test_weights_are_row_major_d_by_k(self, tmp_path):
        model = MLClassifier(class_labels=["A", "B", "C"],
                             weights=np.arange(15, dtype=np.float32).reshape(5, 3),
                             biases=np.zeros(3, dtype=np.float32))
        path = tmp_path / "m.cl"
        model.save(path)
        doc = json.loads(path.read_text())
        assert doc["featureDim"] == 5 and doc["classCount"] == 3
        assert np.allclose(MLClassifier.load(path).weights, model.weights)

    def test_legacy_file_without_identity(self, tmp_path):
        path = tmp_path / "old.paninmodel.json"
        doc = a_model().to_dict()
        del doc["extractorIdentity"]
        del doc["kind"]
        path.write_text(json.dumps(doc))
        restored = MLClassifier.load(path)
        assert restored.extractor_identity == LEGACY_VISION
        assert restored.kind == "logistic"

    def test_optional_fields_survive(self, tmp_path):
        model = a_model(aggregation=AGGREGATION_MEAN_MAX_STD,
                        feature_mean=np.array([1.0, 2.0], dtype=np.float32),
                        feature_std=np.array([0.5, 0.5], dtype=np.float32),
                        null_reference=np.array([[1.0, 0.0]], dtype=np.float32),
                        null_threshold=0.9)
        path = tmp_path / "m.cl"
        model.save(path)
        restored = MLClassifier.load(path)
        assert restored.is_pooled
        assert restored.uses_null_filter and restored.null_threshold == 0.9
        assert np.allclose(restored.feature_mean, [1.0, 2.0])

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ClassifierError):
            MLClassifier(class_labels=["A", "B", "C"],
                         weights=np.zeros((4, 2), dtype=np.float32),
                         biases=np.zeros(2, dtype=np.float32))

    def test_malformed_file(self, tmp_path):
        path = tmp_path / "bad.cl"
        path.write_text("{ not json")
        with pytest.raises(ClassifierError):
            MLClassifier.load(path)


class TestPrediction:
    def test_predict_returns_label_and_probabilities(self):
        label, probs = a_model().predict(np.array([1.0, 1.0], dtype=np.float32))
        assert label in {"A", "B"}
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_batch_matches_single(self):
        model = a_model()
        rows = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
        labels, probs = model.predict_batch(rows)
        assert labels == [model.predict(r)[0] for r in rows]

    def test_wrong_dimension_is_a_clear_error(self):
        with pytest.raises(ClassifierError, match="expects 2-d"):
            a_model().predict(np.zeros(7, dtype=np.float32))

    def test_zscore_is_applied(self):
        plain = a_model()
        scaled = a_model(feature_mean=np.array([1.0, 1.0], dtype=np.float32),
                         feature_std=np.array([2.0, 2.0], dtype=np.float32))
        row = np.array([3.0, 3.0], dtype=np.float32)
        assert not np.allclose(plain.predict_proba(row), scaled.predict_proba(row))

    def test_accepts_only_its_own_extractor(self):
        model = a_model()
        assert model.accepts(PHIKON)
        assert not model.accepts(UNI2)

    def test_geometry_model_accepts_anything(self):
        """Geometry descriptors have no extractor to match."""
        assert a_model(feature_source="geometry").accepts(UNI2)

    def test_null_filter(self):
        reference = np.array([[1.0, 0.0]], dtype=np.float32)
        model = a_model(null_reference=reference, null_threshold=0.99)
        flags = model.is_null_like(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        assert flags.tolist() == [True, False]

    def test_no_null_filter_flags_nothing(self):
        assert not a_model().is_null_like(np.zeros((3, 2), dtype=np.float32)).any()


class TestTraining:
    def test_separable_classes_train_well(self, bank):
        fill_bank(bank)
        model = train_classifier(bank, TrainingSettings(extractor_identity=PHIKON))
        assert model.class_labels == ["PaNIN-1", "PaNIN-3"]
        assert model.metrics.val_accuracy > 0.9
        assert not model.metrics.is_collapsed

    def test_identity_is_recorded(self, bank):
        fill_bank(bank)
        model = train_classifier(bank, TrainingSettings(extractor_identity=PHIKON))
        assert model.extractor_identity == PHIKON
        assert model.feature_dim == DIM

    def test_only_the_selected_extractor_is_used(self, bank):
        fill_bank(bank, identity=PHIKON)
        fill_bank(bank, identity=UNI2, dim=DIM, seed=5)
        model = train_classifier(bank, TrainingSettings(extractor_identity=PHIKON))
        assert model.metrics.train_count + model.metrics.val_count == 50

    def test_unknown_extractor_is_a_clear_error(self, bank):
        fill_bank(bank)
        with pytest.raises(TrainingError, match="No patches"):
            train_classifier(bank, TrainingSettings(extractor_identity=UNI2))

    def test_single_class_is_refused(self, bank):
        bank.add_many([patch("only", np.zeros(DIM)) for _ in range(10)])
        with pytest.raises(TrainingError, match="at least 2 classes"):
            train_classifier(bank, TrainingSettings(extractor_identity=PHIKON))

    def test_thin_class_is_refused_with_names(self, bank):
        fill_bank(bank)
        bank.add(patch("Rare", direction(4)))
        with pytest.raises(TrainingError, match="Rare"):
            train_classifier(bank, TrainingSettings(extractor_identity=PHIKON))

    def test_class_selection(self, bank):
        fill_bank(bank)
        bank.add_many([patch("Stroma", direction(3)) for _ in range(20)])
        model = train_classifier(bank, TrainingSettings(
            extractor_identity=PHIKON, class_labels=["PaNIN-1", "Stroma"]))
        assert model.class_labels == ["PaNIN-1", "Stroma"]

    def test_metrics_are_attached(self, bank):
        fill_bank(bank)
        model = train_classifier(bank, TrainingSettings(extractor_identity=PHIKON))
        assert model.metrics is not None
        assert model.metrics.train_count > 0 and model.metrics.val_count > 0
        assert len(model.metrics.confusion) == 2

    def test_progress_reaches_completion(self, bank):
        fill_bank(bank)
        seen = []
        train_classifier(bank, TrainingSettings(extractor_identity=PHIKON),
                         progress=lambda d, t, m: seen.append(d))
        assert seen and max(seen) == 100

    def test_cancellation(self, bank):
        fill_bank(bank)
        with pytest.raises(TrainingError, match="Cancel"):
            train_classifier(bank, TrainingSettings(extractor_identity=PHIKON),
                             should_cancel=lambda: True)

    def test_round_trips_through_file(self, bank, tmp_path):
        fill_bank(bank)
        model = train_classifier(bank, TrainingSettings(extractor_identity=PHIKON))
        path = tmp_path / "trained.cl"
        model.save(path)
        restored = MLClassifier.load(path)
        sample = direction(1).astype(np.float32)
        assert restored.predict(sample)[0] == model.predict(sample)[0]
        assert restored.metrics.val_accuracy == model.metrics.val_accuracy


class TestNullExclusion:
    def test_null_class_never_becomes_a_label(self, bank):
        fill_bank(bank)
        bank.add_many([patch("Lumen", direction(2)) for _ in range(15)])
        model = train_classifier(bank, TrainingSettings(
            extractor_identity=PHIKON, null_labels=["Lumen"]))
        assert "Lumen" not in model.class_labels

    def test_reference_is_stored_on_the_model(self, bank):
        fill_bank(bank)
        bank.add_many([patch("Lumen", direction(2)) for _ in range(15)])
        model = train_classifier(bank, TrainingSettings(
            extractor_identity=PHIKON, null_labels=["Lumen"]))
        assert model.uses_null_filter
        assert model.null_reference.shape == (15, DIM)
        assert model.null_threshold == 0.85

    def test_reference_is_capped(self, bank):
        """The reference lives inside every .cl and is scanned per prediction."""
        fill_bank(bank)
        bank.add_many([patch("Lumen", direction(2) + i * 0.01) for i in range(300)])
        model = train_classifier(bank, TrainingSettings(
            extractor_identity=PHIKON, null_labels=["Lumen"]))
        assert model.null_reference.shape[0] == 128

    def test_null_like_training_patches_are_dropped(self, bank):
        """A real-class patch resembling the null reference must not train."""
        fill_bank(bank)
        lumen = direction(2)
        bank.add_many([patch("Lumen", lumen) for _ in range(10)])
        bank.add_many([patch("PaNIN-1", lumen * 1.001) for _ in range(10)])

        without = train_classifier(bank, TrainingSettings(extractor_identity=PHIKON))
        with_filter = train_classifier(bank, TrainingSettings(
            extractor_identity=PHIKON, null_labels=["Lumen"], null_threshold=0.9))
        assert (with_filter.metrics.train_count + with_filter.metrics.val_count
                < without.metrics.train_count + without.metrics.val_count)

    def test_over_aggressive_threshold_is_explained(self, bank):
        fill_bank(bank)
        bank.add_many([patch("Lumen", direction(2)) for _ in range(10)])
        with pytest.raises(TrainingError, match="removed all"):
            train_classifier(bank, TrainingSettings(
                extractor_identity=PHIKON, null_labels=["Lumen"], null_threshold=-1.0))

    def test_all_null_is_refused(self, bank):
        bank.add_many([patch("Lumen", direction(2)) for _ in range(10)])
        with pytest.raises(TrainingError, match="marked null"):
            train_classifier(bank, TrainingSettings(
                extractor_identity=PHIKON, null_labels=["Lumen"]))


class TestPooled:
    def test_pooled_triples_the_dimension(self, bank):
        rng = np.random.default_rng(0)
        for name, centre in (("A", direction(0)), ("B", direction(1))):
            for _ in range(8):
                annotation = uuid.uuid4()
                for _ in range(5):
                    bank.add(patch(name, centre + rng.normal(0, 0.4, DIM),
                                   annotation_id=annotation))
        model = train_classifier(bank, TrainingSettings(
            extractor_identity=PHIKON, pooled=True))
        assert model.is_pooled
        assert model.feature_dim == DIM * 3

    def test_pooled_models_carry_zscoring(self, bank):
        rng = np.random.default_rng(1)
        for name, centre in (("A", direction(0)), ("B", direction(1))):
            for _ in range(8):
                annotation = uuid.uuid4()
                for _ in range(5):
                    bank.add(patch(name, centre + rng.normal(0, 0.4, DIM),
                                   annotation_id=annotation))
        model = train_classifier(bank, TrainingSettings(
            extractor_identity=PHIKON, pooled=True))
        assert model.feature_mean is not None and model.feature_std is not None
        assert model.feature_mean.shape == (DIM * 3,)

    def test_too_few_annotations_is_explained(self, bank):
        """Doc 04 §2: pooling was data-starved at ~136 annotations."""
        for name in ("A", "B"):
            annotation = uuid.uuid4()
            for _ in range(10):
                bank.add(patch(name, np.zeros(DIM), annotation_id=annotation))
        with pytest.raises(TrainingError, match="Pooled training needs far more"):
            train_classifier(bank, TrainingSettings(
                extractor_identity=PHIKON, pooled=True))
