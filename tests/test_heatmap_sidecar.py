"""Saving the heatmap beside the slide, and getting it back."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from pathlearn.models.prediction import (PROBABILITY_PLACES, PatchPrediction,
                                         PredictionSet, auto_sidecar_path,
                                         find_sidecar, sidecar_path)

SIZE = 224


def tile(col, row=0, label="PanIN-2", confidence=0.9, classes=2):
    probabilities = np.zeros(classes, dtype=np.float32)
    probabilities[0] = confidence
    return PatchPrediction(x=col * SIZE, y=row * SIZE, size_level0=SIZE,
                           label=label, probabilities=probabilities,
                           patch_size_level=SIZE)


def a_set(count=4, slide="E:/case-01.svs", **kw):
    return PredictionSet(predictions=[tile(i) for i in range(count)],
                         class_labels=["PanIN-2", "Acinar"],
                         slide_path=slide,
                         extractor_identity="onnx:uni2-h:r1", **kw)


class TestPaths:
    def test_the_auto_sidecar_sits_beside_the_slide(self):
        assert auto_sidecar_path("F:/slides/a.svs").name == \
            "a.predictions.json.gz"

    def test_the_hand_saved_one_keeps_the_macos_name(self):
        assert sidecar_path("F:/slides/a.svs").name == "a.predictions.json"

    def test_nothing_beside_the_slide(self, tmp_path):
        assert find_sidecar(tmp_path / "a.svs") is None

    def test_the_compressed_one_wins(self, tmp_path):
        slide = tmp_path / "a.svs"
        a_set().save(auto_sidecar_path(slide))
        a_set().save(sidecar_path(slide))
        assert find_sidecar(slide).suffix == ".gz"

    def test_a_hand_saved_json_is_still_found(self, tmp_path):
        """A sidecar from an older build must not be ignored."""
        slide = tmp_path / "a.svs"
        a_set().save(sidecar_path(slide))
        assert find_sidecar(slide).name == "a.predictions.json"


class TestTheFormat:
    def test_gzip_when_the_name_says_so(self, tmp_path):
        target = tmp_path / "a.predictions.json.gz"
        a_set().save(target)
        assert target.read_bytes()[:2] == b"\x1f\x8b"

    def test_plain_when_it_does_not(self, tmp_path):
        target = tmp_path / "a.predictions.json"
        a_set().save(target)
        assert target.read_bytes()[:1] == b"{"

    def test_compression_is_worth_having(self, tmp_path):
        """The reason this is gzipped at all — pin the order of magnitude."""
        big = PredictionSet(
            predictions=[tile(i, classes=4) for i in range(5_000)],
            class_labels=["a", "b", "c", "d"])
        plain = tmp_path / "p.json"
        squashed = tmp_path / "p.json.gz"
        big.save(plain)
        big.save(squashed)
        assert squashed.stat().st_size < plain.stat().st_size / 5

    def test_it_loads_by_magic_number_not_by_name(self, tmp_path):
        """A renamed sidecar must still open."""
        target = tmp_path / "a.predictions.json.gz"
        a_set().save(target)
        renamed = target.rename(tmp_path / "mystery.bin")
        assert len(PredictionSet.load(renamed)) == 4

    def test_an_interrupted_write_leaves_no_partial(self, tmp_path):
        target = tmp_path / "a.predictions.json.gz"
        a_set().save(target)
        assert not list(tmp_path.glob("*.partial"))

    def test_probabilities_are_rounded_not_dropped(self, tmp_path):
        target = tmp_path / "a.predictions.json"
        a_set(count=1).save(target)
        stored = json.loads(target.read_text(encoding="utf-8"))
        value = stored["predictions"][0]["probabilities"][0]
        assert value == pytest.approx(0.9, abs=10 ** -PROBABILITY_PLACES)
        assert len(str(value).split(".")[-1]) <= PROBABILITY_PLACES

    def test_rounding_cannot_flip_a_call(self, tmp_path):
        """The winning class is stored as a label, never recomputed."""
        marginal = PatchPrediction(
            x=0, y=0, size_level0=SIZE, label="Acinar",
            probabilities=np.array([0.50001, 0.49999], dtype=np.float32))
        target = tmp_path / "a.json"
        PredictionSet(predictions=[marginal],
                      class_labels=["PanIN-2", "Acinar"]).save(target)
        assert PredictionSet.load(target).predictions[0].label == "Acinar"


class TestViewStateTravels:
    def test_the_threshold_comes_back(self, tmp_path):
        target = tmp_path / "a.json.gz"
        a_set(min_confidence=0.65).save(target)
        assert PredictionSet.load(target).min_confidence == pytest.approx(0.65)

    def test_hidden_classes_come_back(self, tmp_path):
        target = tmp_path / "a.json.gz"
        a_set(hidden={"Acinar"}).save(target)
        assert PredictionSet.load(target).hidden == {"Acinar"}

    def test_a_version_one_file_still_loads(self, tmp_path):
        """Older sidecars have neither field; the defaults are the old view."""
        target = tmp_path / "old.json"
        payload = a_set().to_dict()
        payload["version"] = 1
        del payload["minConfidence"]
        del payload["hidden"]
        target.write_text(json.dumps(payload), encoding="utf-8")
        back = PredictionSet.load(target)
        assert back.min_confidence == 0.0
        assert back.hidden == set()
        assert len(back) == 4

    def test_a_gzipped_version_one_file_loads_too(self, tmp_path):
        target = tmp_path / "old.json.gz"
        payload = a_set().to_dict()
        payload["version"] = 1
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        assert len(PredictionSet.load(target)) == 4
