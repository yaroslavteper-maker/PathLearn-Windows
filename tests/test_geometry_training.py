"""Geometry bank and the k-fold grade trainer."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from pathlearn.core.geometry import DIMENSION, VERSION
from pathlearn.core.shape import SHAPE_DIMENSION, SHAPE_VERSION
from pathlearn.data.geometry_bank import FeatureSource, GeometryBank, GeometryRecord
from pathlearn.models.classifier import MLClassifier
from pathlearn.pipeline.geometry_train import (GEOMETRY_IDENTITY, GeometryError,
                                               GeometryTrainingSettings, Validation,
                                               _slide_folds, _stratified_folds,
                                               train_geometry_classifier)


@pytest.fixture
def bank(tmp_path):
    return GeometryBank(tmp_path / "geometry_bank.json")


def record(classification="PaNIN-2", offset=0.0, version=VERSION, dim=DIMENSION,
           seed=None, shape=True, **kw):
    """A record carrying both blocks by default, as real ones do."""
    rng = np.random.default_rng(seed)
    features = (np.arange(dim, dtype=np.float32) * 0.1 + offset
                + rng.normal(0, 0.05, dim).astype(np.float32))
    shape_features = None
    if shape:
        shape_features = (np.arange(SHAPE_DIMENSION, dtype=np.float32) * 0.05 + offset
                          + rng.normal(0, 0.05, SHAPE_DIMENSION).astype(np.float32))
    defaults = dict(slide_path="E:/a.svs", slide_name="a.svs",
                    annotation_id=uuid.uuid4(), classification=classification,
                    features=features, shape_features=shape_features,
                    version=version,
                    shape_version=SHAPE_VERSION if shape else 0)
    defaults.update(kw)
    return GeometryRecord(**defaults)


def fill(bank, per_class=12, seed=0, slides=4):
    """Records spread over several slides, as real ones are."""
    rows = []
    for i, name in enumerate(("PaNIN-2", "PaNIN-3")):
        for j in range(per_class):
            slide = f"s{j % slides}.svs"
            rows.append(record(name, offset=i * 4.0, seed=seed + i * 100 + j,
                               slide_path=f"E:/{slide}", slide_name=slide))
    bank.add(rows)
    return rows


class TestBank:
    def test_round_trip(self, bank):
        r = record()
        bank.add([r])
        reloaded = GeometryBank(bank.path)
        assert len(reloaded) == 1
        assert reloaded.records[0].classification == r.classification
        assert np.allclose(reloaded.records[0].features, r.features)

    def test_file_is_a_flat_array(self, bank):
        """The macOS file is a bare JSON array, not an object."""
        bank.add([record()])
        data = json.loads(bank.path.read_text())
        assert isinstance(data, list)
        assert set(data[0]) == {"slidePath", "slideName", "annotationID",
                                "classification", "features", "version",
                                "shapeFeatures", "shapeVersion"}

    def test_macos_fields_keep_their_names(self, bank):
        """The shape block is additive; the Swift keys must not move."""
        bank.add([record()])
        entry = json.loads(bank.path.read_text())[0]
        assert len(entry["features"]) == DIMENSION
        assert entry["version"] == VERSION

    def test_re_adding_an_annotation_replaces_it(self, bank):
        annotation = uuid.uuid4()
        bank.add([record("old", annotation_id=annotation)])
        bank.add([record("new", annotation_id=annotation)])
        assert len(bank) == 1
        assert bank.records[0].classification == "new"

    def test_stale_records_are_flagged_not_dropped(self, bank):
        bank.add([record(version=1, shape=False), record(version=VERSION)])
        assert len(bank) == 2
        assert bank.stale_count == 1
        assert len(bank.current_records) == 1

    def test_stale_texture_but_valid_shape_is_still_usable(self, bank):
        """Blocks age independently — a stale texture must not bin the shape."""
        bank.add([record(version=1)])
        r = bank.records[0]
        assert not r.has_texture and r.has_shape and r.is_current
        assert bank.usable_for(FeatureSource.SHAPE) != []
        assert bank.usable_for(FeatureSource.TEXTURE) == []

    def test_wrong_dimension_counts_as_stale(self, bank):
        bank.add([record(dim=8, shape=False)])
        assert bank.current_records == []

    def test_remove_stale(self, bank):
        bank.add([record(version=1, shape=False), record(version=VERSION)])
        assert bank.remove_stale() == 1
        assert len(bank) == 1

    def test_remove_class(self, bank):
        bank.add([record("A"), record("B"), record("B")])
        assert bank.remove_class("B") == 2

    def test_clear(self, bank):
        bank.add([record() for _ in range(4)])
        assert bank.clear() == 4
        assert bank.stats().is_empty

    def test_stats_mentions_stale(self, bank):
        bank.add([record(version=1, shape=False)])
        assert "older descriptor version" in bank.stats().summary()

    def test_matrix_shape(self, bank):
        fill(bank, per_class=3)
        matrix, labels = bank.matrix()
        assert matrix.shape == (6, DIMENSION)
        assert sorted(set(labels)) == ["PaNIN-2", "PaNIN-3"]

    def test_corrupt_file_does_not_crash(self, tmp_path):
        path = tmp_path / "g.json"
        path.write_text("{ not json")
        assert len(GeometryBank(path)) == 0

    def test_has_annotation(self, bank):
        r = record()
        bank.add([r])
        assert bank.has_annotation(r.annotation_id)
        assert not bank.has_annotation(uuid.uuid4())


class TestFolds:
    def test_every_index_used_exactly_once(self):
        y = np.array([0] * 7 + [1] * 5)
        folds = _stratified_folds(y, 5, seed=0)
        flat = sorted(i for f in folds for i in f)
        assert flat == list(range(12))

    def test_round_robin_keeps_classes_in_every_fold(self):
        """Slicing 7 examples into 5 folds would empty two of them."""
        y = np.array([0] * 7 + [1] * 7)
        folds = _stratified_folds(y, 5, seed=0)
        for fold in folds:
            assert len(set(y[fold])) == 2

    def test_is_seeded(self):
        y = np.array([0] * 10 + [1] * 10)
        assert _stratified_folds(y, 5, 7) == _stratified_folds(y, 5, 7)


class TestTraining:
    def test_separable_classes(self, bank):
        fill(bank)
        model = train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))
        assert model.class_labels == ["PaNIN-2", "PaNIN-3"]
        assert model.metrics.val_accuracy > 0.8

    def test_model_is_marked_as_geometry(self, bank):
        fill(bank)
        model = train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))
        assert model.is_geometry
        assert model.extractor_identity == GEOMETRY_IDENTITY
        assert model.feature_dim == DIMENSION

    def test_geometry_model_needs_no_extractor(self, bank):
        """It accepts any identity because descriptors have no feature space."""
        fill(bank)
        assert train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED)).accepts("onnx:anything:r1")

    def test_zscoring_is_always_stored(self, bank):
        """Descriptors mix per-mm² counts, µm² areas and unit fractions."""
        fill(bank)
        model = train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))
        assert model.feature_mean is not None
        assert model.feature_std is not None
        assert model.feature_mean.shape == (DIMENSION,)

    def test_cross_validation_uses_every_example_once(self, bank):
        fill(bank, per_class=10)
        model = train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))
        assert model.metrics.val_count == 20

    def test_folds_capped_by_smallest_class(self, bank):
        """A fold that cannot hold one example of a class cannot validate it."""
        bank.add([record("A", offset=0, seed=i) for i in range(10)]
                 + [record("B", offset=4, seed=100 + i) for i in range(3)])
        model = train_geometry_classifier(bank,
                                          GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED, folds=5))
        assert model.metrics.val_count == 13

    def test_empty_bank(self, bank):
        with pytest.raises(GeometryError, match="empty"):
            train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))

    def test_all_stale_explains_the_version_change(self, bank):
        bank.add([record("A", version=1, shape=False), record("B", version=1, shape=False)])
        with pytest.raises(GeometryError, match="Recompute descriptors"):
            train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))

    def test_single_class_is_refused(self, bank):
        bank.add([record("only", seed=i) for i in range(6)])
        with pytest.raises(GeometryError, match="at least 2 classes"):
            train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))

    def test_singleton_class_is_named(self, bank):
        fill(bank)
        bank.add([record("Rare")])
        with pytest.raises(GeometryError, match="Rare"):
            train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))

    def test_class_selection(self, bank):
        fill(bank)
        bank.add([record("Stroma", offset=-6, seed=900 + i) for i in range(6)])
        model = train_geometry_classifier(
            bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE,
                                           validation=Validation.STRATIFIED,
                                           class_labels=["PaNIN-2", "Stroma"]))
        assert model.class_labels == ["PaNIN-2", "Stroma"]

    def test_progress_completes(self, bank):
        fill(bank)
        seen: list[int] = []
        train_geometry_classifier(bank, progress=lambda d, t, m: seen.append(d))
        assert seen and max(seen) == 100

    def test_cancellation(self, bank):
        fill(bank)
        with pytest.raises(GeometryError, match="Cancel"):
            train_geometry_classifier(bank, should_cancel=lambda: True)

    def test_round_trips_through_file(self, bank, tmp_path):
        fill(bank)
        model = train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))
        path = tmp_path / "geom.cl"
        model.save(path)
        restored = MLClassifier.load(path)
        assert restored.is_geometry
        sample = bank.current_records[0].features
        assert restored.predict(sample)[0] == model.predict(sample)[0]

    def test_confusion_is_pooled_out_of_fold(self, bank):
        fill(bank, per_class=10)
        model = train_geometry_classifier(bank, GeometryTrainingSettings(source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))
        assert np.asarray(model.metrics.confusion).sum() == 20


class TestSlideFolds:
    """Leave-one-slide-out — the protocol that says whether this generalises."""

    def test_one_fold_per_slide(self):
        records = [record("A", slide_path="E:/a.svs", slide_name="a.svs"),
                   record("A", slide_path="E:/a.svs", slide_name="a.svs"),
                   record("B", slide_path="E:/b.svs", slide_name="b.svs")]
        folds = _slide_folds(records)
        assert len(folds) == 2
        assert sorted(len(f) for f in folds) == [1, 2]

    def test_every_record_used_exactly_once(self):
        records = [record("A", slide_path=f"E:/s{i % 3}.svs",
                          slide_name=f"s{i % 3}.svs") for i in range(11)]
        folds = _slide_folds(records)
        assert sorted(i for f in folds for i in f) == list(range(11))

    def test_no_slide_appears_in_two_folds(self):
        records = [record("A", slide_path=f"E:/s{i % 3}.svs",
                          slide_name=f"s{i % 3}.svs") for i in range(9)]
        folds = _slide_folds(records)
        for fold in folds:
            assert len({records[i].slide_path for i in fold}) == 1

    def test_single_slide_is_refused_with_a_reason(self, bank):
        """One slide cannot answer 'does this generalise to another slide?'"""
        fill(bank, slides=1)
        with pytest.raises(GeometryError, match="at least 2 slides"):
            train_geometry_classifier(bank, GeometryTrainingSettings(
                source=FeatureSource.TEXTURE, validation=Validation.BY_SLIDE))

    def test_by_slide_is_the_default(self):
        """The honest protocol should be what you get without asking."""
        assert GeometryTrainingSettings().validation is Validation.BY_SLIDE

    def test_both_protocols_score_every_record(self, bank):
        fill(bank, per_class=12, slides=4)
        optimistic = train_geometry_classifier(bank, GeometryTrainingSettings(
            source=FeatureSource.TEXTURE, validation=Validation.STRATIFIED))
        honest = train_geometry_classifier(bank, GeometryTrainingSettings(
            source=FeatureSource.TEXTURE, validation=Validation.BY_SLIDE))
        assert optimistic.metrics.val_count == honest.metrics.val_count == 24

    def test_a_fold_leaving_one_class_in_training_is_skipped(self, bank):
        """Held-out slides can strand the trainer with a single class."""
        bank.add([record("A", offset=0, seed=i, slide_path="E:/a.svs",
                         slide_name="a.svs") for i in range(6)]
                 + [record("B", offset=4, seed=50 + i, slide_path="E:/b.svs",
                           slide_name="b.svs") for i in range(6)])
        # Holding out either slide leaves training with one class, so no fold
        # is usable — the pooled confusion is empty rather than the run crashing.
        model = train_geometry_classifier(bank, GeometryTrainingSettings(
            source=FeatureSource.TEXTURE, validation=Validation.BY_SLIDE))
        assert model.metrics.val_count == 0
        assert model.metrics.train_accuracy > 0
