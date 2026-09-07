"""Extractor identity, descriptors, and registry discovery.

These run without any model file. The tests that need the real 4 GB of ONNX
models are in `test_extractors_live.py` and skip when they are not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pathlearn.extractors.descriptor import (DESCRIPTOR_SUFFIX, DescriptorError,
                                             ExtractorDescriptor, PixelNormalization)
from pathlearn.extractors.identity import LEGACY_VISION, ExtractorIdentity, normalise
from pathlearn.extractors.registry import ExtractorRegistry


def write_extractor(directory, name="demo", dim=768, kind="onnx", revision=1,
                    model_bytes=b"not really a model", external=False, **overrides):
    """A descriptor plus a stand-in model file, so validate() passes."""
    model_name = f"{name}.onnx"
    (directory / model_name).write_bytes(model_bytes)
    if external:
        (directory / f"{model_name}.data").write_bytes(b"weights")
    body = {
        "identity": {"kind": kind, "name": name, "revision": revision},
        "kind": kind,
        "inputWidth": 224,
        "inputHeight": 224,
        "featureDim": dim,
        "pixelNormalization": {
            "meanRGB": [0.485, 0.456, 0.406],
            "stdRGB": [0.229, 0.224, 0.225],
            "scale": 1 / 255,
        },
        "modelFilename": model_name,
        "externalData": external,
    }
    body.update(overrides)
    path = directory / f"{name}{DESCRIPTOR_SUFFIX}"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


class TestIdentity:
    def test_string_form(self):
        assert str(ExtractorIdentity("onnx", "uni2-h", 1)) == "onnx:uni2-h:r1"

    def test_round_trip(self):
        for text in ("onnx:phikon-v1:r1", "coreml:uni2-h:r3", "vision:vision:r2"):
            assert str(ExtractorIdentity.parse(text)) == text

    def test_parses_bare_kind_name_as_revision_1(self):
        """Some early macOS files omitted the revision."""
        assert ExtractorIdentity.parse("coreml:uni") == ExtractorIdentity("coreml", "uni", 1)

    @pytest.mark.parametrize("bad", ["", "nope", "a:b:c", "a:b:rX", ":::"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            ExtractorIdentity.parse(bad)

    def test_dict_round_trip(self):
        ident = ExtractorIdentity("onnx", "uni-v1", 2)
        assert ExtractorIdentity.from_dict(ident.to_dict()) == ident

    def test_none_normalises_to_legacy_vision(self):
        """Patches written before the extractor system carry no identity."""
        assert normalise(None) == LEGACY_VISION

    def test_same_model_ignores_kind_but_equality_does_not(self):
        onnx = ExtractorIdentity("onnx", "uni2-h", 1)
        coreml = ExtractorIdentity("coreml", "uni2-h", 1)
        assert onnx.same_model(coreml)
        assert onnx != coreml, "the guard must treat runtimes as distinct"

    def test_same_model_respects_revision(self):
        assert not ExtractorIdentity("onnx", "uni", 1).same_model(
            ExtractorIdentity("onnx", "uni", 2))

    def test_is_hashable(self):
        assert len({ExtractorIdentity("onnx", "a", 1),
                    ExtractorIdentity("onnx", "a", 1)}) == 1


class TestPixelNormalization:
    def test_formula(self):
        """normalized = (pixel * scale - mean) / std."""
        norm = PixelNormalization(mean_rgb=(0.5, 0.5, 0.5), std_rgb=(0.5, 0.5, 0.5),
                                  scale=1 / 255)
        out = norm.apply(np.full((2, 2, 3), 255, dtype=np.uint8))
        assert np.allclose(out, 1.0)
        out = norm.apply(np.zeros((2, 2, 3), dtype=np.uint8))
        assert np.allclose(out, -1.0)

    def test_output_is_nchw_float32(self):
        out = PixelNormalization().apply(np.zeros((8, 6, 3), dtype=np.uint8))
        assert out.shape == (1, 3, 8, 6)
        assert out.dtype == np.float32

    def test_batch_is_preserved(self):
        out = PixelNormalization().apply(np.zeros((5, 8, 6, 3), dtype=np.uint8))
        assert out.shape == (5, 3, 8, 6)

    def test_channels_are_not_transposed(self):
        """Channel c of the output must come from channel c of the input."""
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        image[0, 0] = (255, 0, 0)
        norm = PixelNormalization(mean_rgb=(0, 0, 0), std_rgb=(1, 1, 1), scale=1 / 255)
        out = norm.apply(image)
        assert out[0, 0, 0, 0] == pytest.approx(1.0)   # R
        assert out[0, 1, 0, 0] == pytest.approx(0.0)   # G
        assert out[0, 2, 0, 0] == pytest.approx(0.0)   # B

    def test_output_is_contiguous(self):
        assert PixelNormalization().apply(np.zeros((4, 8, 6, 3), np.uint8)).flags["C_CONTIGUOUS"]

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            PixelNormalization().apply(np.zeros((8, 8), dtype=np.uint8))

    def test_defaults_are_imagenet(self):
        norm = PixelNormalization.from_dict(None)
        assert norm.mean_rgb == (0.485, 0.456, 0.406)
        assert norm.std_rgb == (0.229, 0.224, 0.225)
        assert norm.scale == pytest.approx(1 / 255)


class TestDescriptor:
    def test_load(self, tmp_path):
        path = write_extractor(tmp_path, "phikon-v1", dim=768)
        d = ExtractorDescriptor.load(path)
        assert str(d.identity) == "onnx:phikon-v1:r1"
        assert d.feature_dim == 768
        assert d.input_size == (224, 224)
        assert d.model_path == tmp_path / "phikon-v1.onnx"

    def test_io_names_default_when_absent(self, tmp_path):
        """macOS descriptors never recorded these."""
        d = ExtractorDescriptor.load(write_extractor(tmp_path))
        assert (d.input_name, d.output_name) == ("input", "features")

    def test_io_names_are_read_when_present(self, tmp_path):
        path = write_extractor(tmp_path, inputName="image", outputName="embedding")
        d = ExtractorDescriptor.load(path)
        assert (d.input_name, d.output_name) == ("image", "embedding")

    def test_validate_detects_missing_model(self, tmp_path):
        path = write_extractor(tmp_path, "gone")
        (tmp_path / "gone.onnx").unlink()
        with pytest.raises(DescriptorError, match="not found"):
            ExtractorDescriptor.load(path).validate()

    def test_validate_detects_missing_external_weights(self, tmp_path):
        """UNI2-h keeps 2.7 GB of weights in a sidecar; losing it must be clear."""
        path = write_extractor(tmp_path, "big", external=True)
        (tmp_path / "big.onnx.data").unlink()
        with pytest.raises(DescriptorError, match="must stay together"):
            ExtractorDescriptor.load(path).validate()

    @pytest.mark.parametrize("field,value", [
        ("featureDim", 0), ("inputWidth", -1), ("inputHeight", 0),
    ])
    def test_rejects_nonsensical_dimensions(self, tmp_path, field, value):
        path = write_extractor(tmp_path, **{field: value})
        with pytest.raises(DescriptorError):
            ExtractorDescriptor.load(path)

    def test_rejects_missing_required_field(self, tmp_path):
        path = tmp_path / f"x{DESCRIPTOR_SUFFIX}"
        path.write_text(json.dumps({"identity": {"kind": "onnx", "name": "x"}}))
        with pytest.raises(DescriptorError):
            ExtractorDescriptor.load(path)

    def test_rejects_unreadable_json(self, tmp_path):
        path = tmp_path / f"x{DESCRIPTOR_SUFFIX}"
        path.write_text("{ not json")
        with pytest.raises(DescriptorError):
            ExtractorDescriptor.load(path)

    def test_dict_round_trip(self, tmp_path):
        original = ExtractorDescriptor.load(write_extractor(tmp_path, "rt", dim=1536))
        again = ExtractorDescriptor.from_dict(original.to_dict(), base_dir=tmp_path)
        assert again.identity == original.identity
        assert again.feature_dim == original.feature_dim
        assert again.normalization == original.normalization


class TestRegistry:
    def test_discovers_all(self, tmp_path):
        write_extractor(tmp_path, "a", dim=768)
        write_extractor(tmp_path, "b", dim=1024)
        registry = ExtractorRegistry([tmp_path])
        assert len(registry) == 2
        assert {i.name for i in registry.identities} == {"a", "b"}

    def test_empty_directory(self, tmp_path):
        registry = ExtractorRegistry([tmp_path])
        assert registry.is_empty
        assert "No extractors installed" in registry.describe()

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert ExtractorRegistry([tmp_path / "nope"]).is_empty

    def test_broken_descriptor_does_not_hide_good_ones(self, tmp_path):
        write_extractor(tmp_path, "good")
        (tmp_path / f"bad{DESCRIPTOR_SUFFIX}").write_text("{ broken")
        registry = ExtractorRegistry([tmp_path])
        assert len(registry) == 1
        assert len(registry.problems) == 1
        assert "bad" in registry.problems[0].path.name

    def test_missing_model_is_reported_not_raised(self, tmp_path):
        write_extractor(tmp_path, "ghost")
        (tmp_path / "ghost.onnx").unlink()
        registry = ExtractorRegistry([tmp_path])
        assert registry.is_empty and len(registry.problems) == 1

    def test_appledouble_files_are_skipped(self, tmp_path):
        """macOS copies leave ._ siblings that are not JSON."""
        write_extractor(tmp_path, "real")
        (tmp_path / f"._real{DESCRIPTOR_SUFFIX}").write_bytes(b"\x00\x05\x16\x07junk")
        registry = ExtractorRegistry([tmp_path])
        assert len(registry) == 1 and not registry.problems

    def test_lookup_by_identity_and_name(self, tmp_path):
        write_extractor(tmp_path, "uni2-h", dim=1536)
        registry = ExtractorRegistry([tmp_path])
        assert registry.by_identity("onnx:uni2-h:r1").feature_dim == 1536
        assert registry.by_name("uni2-h").feature_dim == 1536
        assert registry.by_identity("onnx:nope:r1") is None

    def test_by_name_prefers_highest_revision(self, tmp_path):
        write_extractor(tmp_path, "dup", dim=100, revision=1)
        second = tmp_path / "v2"
        second.mkdir()
        write_extractor(second, "dup", dim=200, revision=2)
        registry = ExtractorRegistry([tmp_path, second])
        assert registry.by_name("dup").feature_dim == 200

    def test_duplicate_identity_is_reported(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        write_extractor(tmp_path, "same")
        write_extractor(other, "same")
        registry = ExtractorRegistry([tmp_path, other])
        assert len(registry) == 1
        assert any("duplicate" in p.reason for p in registry.problems)

    def test_open_unknown_identity_lists_alternatives(self, tmp_path):
        write_extractor(tmp_path, "installed")
        registry = ExtractorRegistry([tmp_path])
        with pytest.raises(KeyError, match="installed"):
            registry.open("onnx:absent:r1")

    def test_rescan_picks_up_new_extractors(self, tmp_path):
        registry = ExtractorRegistry([tmp_path])
        assert registry.is_empty
        write_extractor(tmp_path, "late")
        registry.rescan()
        assert len(registry) == 1
