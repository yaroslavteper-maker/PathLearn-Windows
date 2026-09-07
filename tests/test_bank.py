"""Patch bank: SQLite storage, filtering, and .bank interop."""

from __future__ import annotations

import base64
import json
import uuid

import numpy as np
import pytest

from pathlearn.data.bank import (ELEMENT_FLOAT16, ELEMENT_FLOAT32, Patch, PatchBank,
                                 decode_features, encode_features)
from pathlearn.extractors.identity import LEGACY_VISION

PHIKON = "onnx:phikon-v1:r1"
UNI2 = "onnx:uni2-h:r1"


@pytest.fixture
def bank(tmp_path):
    with PatchBank(tmp_path / "bank.db") as b:
        yield b


def make_patch(classification="PaNIN-2", identity=PHIKON, dim=768, **kw):
    defaults = dict(
        slide_path="E:/slides/a.svs", slide_name="a.svs",
        annotation_id=uuid.uuid4(), classification=classification,
        patch_x=1000, patch_y=2000, patch_level=0, patch_size_level=224,
        features=np.arange(dim, dtype=np.float32), extractor_identity=identity,
        white_fraction=0.1, nucleus_count=42,
    )
    defaults.update(kw)
    return Patch(**defaults)


class TestFeatureCoding:
    def test_float32_round_trip(self):
        v = np.array([1.5, -2.25, 0.0], dtype=np.float32)
        assert np.array_equal(decode_features(encode_features(v), 3, ELEMENT_FLOAT32), v)

    def test_float16_halves_the_bytes(self):
        v = np.arange(64, dtype=np.float32)
        assert len(encode_features(v, ELEMENT_FLOAT16)) == len(encode_features(v)) // 2

    def test_float16_decodes_to_float32(self):
        v = np.array([1.0, 2.0], dtype=np.float32)
        out = decode_features(encode_features(v, ELEMENT_FLOAT16), 2, ELEMENT_FLOAT16)
        assert out.dtype == np.float32 and np.allclose(out, v)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            decode_features(encode_features(np.zeros(4, np.float32)), 8, ELEMENT_FLOAT32)


class TestStorage:
    def test_round_trip(self, bank):
        original = make_patch()
        bank.add(original)
        [restored] = bank.fetch()
        assert restored.id == original.id
        assert restored.classification == original.classification
        assert restored.patch_x == original.patch_x
        assert np.array_equal(restored.features, original.features)
        assert restored.extractor_identity == PHIKON

    def test_persists_across_reopen(self, tmp_path):
        path = tmp_path / "b.db"
        with PatchBank(path) as b:
            b.add(make_patch())
        with PatchBank(path) as b:
            assert len(b.fetch()) == 1

    def test_add_many(self, bank):
        assert bank.add_many([make_patch() for _ in range(30)]) == 30
        assert len(bank.fetch()) == 30

    def test_add_empty_is_noop(self, bank):
        assert bank.add_many([]) == 0

    def test_same_id_replaces(self, bank):
        p = make_patch(classification="old")
        bank.add(p)
        bank.add(make_patch(id=p.id, classification="new"))
        assert [x.classification for x in bank.fetch()] == ["new"]

    def test_size_level0_accounts_for_level(self):
        assert make_patch(patch_size_level=224, patch_level=2).size_level0 == 896


class TestFiltering:
    def test_by_extractor(self, bank):
        bank.add_many([make_patch(identity=PHIKON), make_patch(identity=UNI2, dim=1536)])
        assert len(bank.fetch(extractor_identity=PHIKON)) == 1

    def test_by_classification(self, bank):
        bank.add_many([make_patch(classification=c) for c in ("A", "B", "B")])
        assert len(bank.fetch(classifications=["B"])) == 2
        assert len(bank.fetch(classifications=["A", "B"])) == 3

    def test_by_white_fraction(self, bank):
        bank.add_many([make_patch(white_fraction=w) for w in (0.1, 0.5, 0.9)])
        assert len(bank.fetch(max_white_fraction=0.6)) == 2

    def test_min_nuclei_keeps_unmeasured(self, bank):
        """nucleus_count of -1 means never measured, not zero nuclei."""
        bank.add_many([make_patch(nucleus_count=n) for n in (-1, 5, 50)])
        kept = bank.fetch(min_nuclei=10)
        assert sorted(p.nucleus_count for p in kept) == [-1, 50]

    def test_by_slide_and_annotation(self, bank):
        aid = uuid.uuid4()
        bank.add_many([make_patch(annotation_id=aid),
                       make_patch(slide_path="E:/slides/b.svs")])
        assert len(bank.fetch(slide_path="E:/slides/a.svs")) == 1
        assert len(bank.fetch(annotation_id=aid)) == 1

    def test_filters_compose(self, bank):
        bank.add_many([
            make_patch(classification="A", white_fraction=0.1),
            make_patch(classification="A", white_fraction=0.9),
            make_patch(classification="B", white_fraction=0.1),
        ])
        assert len(bank.fetch(classifications=["A"], max_white_fraction=0.5)) == 1

    def test_limit(self, bank):
        bank.add_many([make_patch() for _ in range(10)])
        assert len(bank.fetch(limit=3)) == 3


class TestFeatureMatrix:
    def test_shape_and_labels(self, bank):
        bank.add_many([make_patch(classification=c) for c in ("A", "B")])
        matrix, labels = bank.feature_matrix(bank.fetch())
        assert matrix.shape == (2, 768)
        assert sorted(labels) == ["A", "B"]

    def test_mixed_dimensions_raise(self, bank):
        """Silently truncating two feature spaces would corrupt training."""
        bank.add_many([make_patch(identity=PHIKON, dim=768),
                       make_patch(identity=UNI2, dim=1536)])
        with pytest.raises(ValueError, match="mixed feature dimensions"):
            bank.feature_matrix(bank.fetch())

    def test_empty(self, bank):
        matrix, labels = bank.feature_matrix([])
        assert matrix.shape[0] == 0 and labels == []


class TestStats:
    def test_counts(self, bank):
        bank.add_many([make_patch(classification="A") for _ in range(3)]
                      + [make_patch(classification="B")])
        s = bank.stats()
        assert s.total == 4 and s.by_class == {"A": 3, "B": 1}

    def test_detects_mixed_extractors(self, bank):
        bank.add_many([make_patch(identity=PHIKON), make_patch(identity=UNI2, dim=1536)])
        s = bank.stats()
        assert s.is_mixed
        assert "multiple extractors" in s.summary()

    def test_empty_bank(self, bank):
        assert bank.stats().is_empty
        assert "empty" in bank.stats().summary().lower()

    def test_extractor_identities(self, bank):
        bank.add_many([make_patch(identity=PHIKON), make_patch(identity=UNI2, dim=1536)])
        assert bank.extractor_identities == sorted([PHIKON, UNI2])


class TestDeletion:
    def test_delete_by_extractor(self, bank):
        bank.add_many([make_patch(identity=PHIKON), make_patch(identity=UNI2, dim=1536)])
        assert bank.delete_where(extractor_identity=PHIKON) == 1
        assert len(bank.fetch()) == 1

    def test_delete_by_slide(self, bank):
        bank.add_many([make_patch(), make_patch(slide_path="E:/slides/b.svs")])
        assert bank.delete_where(slide_path="E:/slides/b.svs") == 1

    def test_delete_without_filter_refuses(self, bank):
        bank.add(make_patch())
        with pytest.raises(ValueError, match="at least one filter"):
            bank.delete_where()

    def test_clear(self, bank):
        bank.add_many([make_patch() for _ in range(5)])
        assert bank.clear() == 5
        assert bank.stats().is_empty


class TestBankExport:
    def test_round_trip_through_file(self, bank, tmp_path):
        bank.add_many([make_patch(classification=c) for c in ("A", "B", "C")])
        path = tmp_path / "out.bank"
        assert bank.export_bank(path) == 3

        with PatchBank(tmp_path / "other.db") as other:
            assert other.import_bank(path) == 3
            assert {p.classification for p in other.fetch()} == {"A", "B", "C"}
            assert np.array_equal(other.fetch()[0].features, bank.fetch()[0].features)

    def test_export_shape_matches_macos(self, bank, tmp_path):
        bank.add(make_patch())
        path = tmp_path / "o.bank"
        bank.export_bank(path)
        doc = json.loads(path.read_text())
        assert set(doc) == {"version", "createdAt", "patchCount", "patches"}
        entry = doc["patches"][0]
        for key in ("slidePath", "slideName", "annotationID", "classification",
                    "patchX", "patchY", "patchLevel", "patchSizeLevel",
                    "featureBase64", "featureDim", "featureElementType",
                    "extractorIdentity", "whiteFraction", "nucleusCount"):
            assert key in entry, f"missing {key}"

    def test_import_replace_empties_first(self, bank, tmp_path):
        bank.add(make_patch(classification="old"))
        path = tmp_path / "o.bank"
        bank.export_bank(path, [make_patch(classification="new")])
        bank.import_bank(path, replace=True)
        assert [p.classification for p in bank.fetch()] == ["new"]

    def test_import_append_keeps_existing(self, bank, tmp_path):
        bank.add(make_patch(classification="old"))
        path = tmp_path / "o.bank"
        bank.export_bank(path, [make_patch(classification="new")])
        bank.import_bank(path, replace=False)
        assert len(bank.fetch()) == 2

    def test_legacy_entry_without_identity(self, bank, tmp_path):
        """macOS banks predating the extractor system carry no identity."""
        doc = {"version": 1, "patchCount": 1, "patches": [{
            "slidePath": "x.svs", "slideName": "x.svs",
            "annotationID": str(uuid.uuid4()), "classification": "A",
            "patchX": 0, "patchY": 0, "patchLevel": 0, "patchSizeLevel": 224,
            "featureBase64": base64.b64encode(
                np.zeros(8, np.float32).tobytes()).decode(),
            "featureDim": 8, "featureElementType": 1,
        }]}
        path = tmp_path / "legacy.bank"
        path.write_text(json.dumps(doc))
        assert bank.import_bank(path) == 1
        assert bank.fetch()[0].extractor_identity == LEGACY_VISION

    def test_malformed_entries_are_skipped(self, bank, tmp_path):
        doc = {"patches": [{"garbage": True}, {"also": "bad"}]}
        path = tmp_path / "bad.bank"
        path.write_text(json.dumps(doc))
        assert bank.import_bank(path) == 0

    def test_real_macos_bank_loads(self, bank):
        """The user's actual PhikonPatches224.bank, if the data drive is present."""
        from pathlib import Path
        source = Path("E:/EiblClassifier2/PhikonPatches224.bank")
        if not source.exists():
            pytest.skip("drive E: not available")
        assert bank.import_bank(source) > 1000
        stats = bank.stats()
        assert stats.by_extractor == {"coreml:phikon-v1:r1": stats.total}
        matrix, labels = bank.feature_matrix(bank.fetch(limit=50))
        assert matrix.shape[1] == 768
