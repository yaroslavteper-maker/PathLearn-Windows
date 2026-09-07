"""Extractor equivalence: an explicit, recorded, per-pair user decision."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pathlearn.extractors.equivalence import (EquivalenceStore, candidate_substitute)
from pathlearn.extractors.registry import ExtractorRegistry
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.classifier import MLClassifier
from pathlearn.pipeline.predict import (PredictionError, PredictionSettings,
                                        predict_regions)
from synthetic_slide import write_synthetic_slide
from test_extraction_pipeline import FakeExtractor
from test_extractors import write_extractor

COREML_UNI2 = "coreml:uni2-h:r1"
ONNX_UNI2 = "onnx:uni2-h:r1"


@pytest.fixture
def store(tmp_path):
    return EquivalenceStore(tmp_path / "equiv.json")


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("s") / "eq.tif"
    write_synthetic_slide(path, 2048, 1536, levels=3)
    with SlideImage(path) as s:
        yield s


class TestStore:
    def test_nothing_is_equivalent_by_default(self, store):
        """The guard must start strict; equivalence is always opt-in."""
        assert store.approved_for(COREML_UNI2) is None
        assert store.accepted_identities(COREML_UNI2) == {COREML_UNI2}

    def test_approve_and_recall(self, store):
        store.approve(COREML_UNI2, ONNX_UNI2, note="measured")
        assert store.approved_for(COREML_UNI2) == ONNX_UNI2
        assert store.accepted_identities(COREML_UNI2) == {COREML_UNI2, ONNX_UNI2}

    def test_approval_persists(self, tmp_path):
        path = tmp_path / "equiv.json"
        EquivalenceStore(path).approve(COREML_UNI2, ONNX_UNI2)
        assert EquivalenceStore(path).approved_for(COREML_UNI2) == ONNX_UNI2

    def test_approval_is_per_pair(self, store):
        store.approve(COREML_UNI2, ONNX_UNI2)
        assert store.approved_for("coreml:phikon-v1:r1") is None

    def test_reapproving_replaces(self, store):
        store.approve(COREML_UNI2, ONNX_UNI2)
        store.approve(COREML_UNI2, "onnx:other:r1")
        assert store.approved_for(COREML_UNI2) == "onnx:other:r1"
        assert len(store.entries) == 1

    def test_revoke(self, store):
        store.approve(COREML_UNI2, ONNX_UNI2)
        store.revoke(COREML_UNI2)
        assert store.approved_for(COREML_UNI2) is None

    def test_file_is_readable(self, store):
        store.approve(COREML_UNI2, ONNX_UNI2, note="why")
        data = json.loads(store.path.read_text())
        assert data["equivalences"][0]["wanted"] == COREML_UNI2
        assert data["equivalences"][0]["note"] == "why"

    def test_corrupt_file_does_not_crash(self, tmp_path):
        path = tmp_path / "equiv.json"
        path.write_text("{ not json")
        assert EquivalenceStore(path).entries == []


class TestCandidate:
    def test_same_name_different_kind_is_offered(self, tmp_path):
        write_extractor(tmp_path, "uni2-h", dim=1536)
        registry = ExtractorRegistry([tmp_path])
        assert candidate_substitute(COREML_UNI2, registry) == ONNX_UNI2

    def test_different_revision_is_not_offered(self, tmp_path):
        """A revision bump means the model file changed — that is the point of it."""
        write_extractor(tmp_path, "uni2-h", dim=1536, revision=2)
        registry = ExtractorRegistry([tmp_path])
        assert candidate_substitute(COREML_UNI2, registry) is None

    def test_different_name_is_not_offered(self, tmp_path):
        write_extractor(tmp_path, "phikon-v1", dim=768)
        registry = ExtractorRegistry([tmp_path])
        assert candidate_substitute(COREML_UNI2, registry) is None

    def test_nothing_installed(self, tmp_path):
        assert candidate_substitute(COREML_UNI2, ExtractorRegistry([tmp_path])) is None

    def test_malformed_identity(self, tmp_path):
        write_extractor(tmp_path, "uni2-h")
        assert candidate_substitute("garbage", ExtractorRegistry([tmp_path])) is None


class TestPredictionGate:
    def _run(self, slide, accept=None):
        weights = np.zeros((16, 2), dtype=np.float32)
        weights[0] = (0.01, -0.01)
        classifier = MLClassifier(class_labels=["A", "B"], weights=weights,
                                  biases=np.zeros(2, dtype=np.float32),
                                  extractor_identity=COREML_UNI2)
        region = Annotation(points=[Point(300, 300), Point(900, 300),
                                    Point(900, 900), Point(300, 900)],
                            classification="ROI", color=AnnotationColor.default())
        return predict_regions(slide, [region], FakeExtractor(dim=16, name="uni2-h"),
                               classifier,
                               PredictionSettings(max_white_fraction=1.0),
                               accept_identities=accept)

    def test_mismatch_is_refused_without_approval(self, slide):
        with pytest.raises(PredictionError, match="not comparable"):
            self._run(slide)

    def test_approved_identity_is_allowed(self, slide):
        result, _ = self._run(slide, accept={COREML_UNI2, "test:uni2-h:r1"})
        assert not result.is_empty

    def test_unrelated_approval_does_not_help(self, slide):
        """Approving one pair must not open the gate for another."""
        with pytest.raises(PredictionError):
            self._run(slide, accept={"coreml:phikon-v1:r1", "onnx:phikon-v1:r1"})
