"""Extraction pipeline: sampling, filtering, batching, and bank writes.

Unit tests use a stand-in extractor so they need no model files. The live test
at the end runs the real chain on the real installed extractor.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.core.nucleus import ClassicalNucleusSegmenter
from pathlearn.data.bank import PatchBank
from pathlearn.extractors.identity import ExtractorIdentity
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.pipeline.extract import (ExtractionSettings, estimate_patch_count,
                                        extract_annotations, white_fraction)
from synthetic_slide import write_synthetic_slide

WIDTH, HEIGHT = 4096, 3072


class FakeExtractor:
    """Stands in for OnnxFeatureExtractor: records what it was asked to embed."""

    def __init__(self, dim=16, size=(224, 224), name="fake"):
        self.identity = ExtractorIdentity("test", name, 1)
        self.feature_dim = dim
        self.input_size = size
        self.calls: list[int] = []
        self.seen: list[np.ndarray] = []

    def extract(self, patches):
        batch = np.asarray(patches)
        if batch.ndim == 3:
            batch = batch[None]
        self.calls.append(batch.shape[0])
        self.seen.append(batch)
        # Deterministic, content-dependent, so identical patches embed identically.
        means = batch.reshape(batch.shape[0], -1).mean(axis=1)
        return np.tile(means[:, None], (1, self.feature_dim)).astype(np.float32)

    def extract_one(self, patch):
        return self.extract(patch)[0]


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("slides") / "extract.tif"
    write_synthetic_slide(path, WIDTH, HEIGHT, levels=4)
    with SlideImage(path) as s:
        yield s


@pytest.fixture
def bank(tmp_path):
    with PatchBank(tmp_path / "b.db") as b:
        yield b


def region(x0, y0, size, classification="PaNIN-2", subtractive=False):
    return Annotation(
        points=[Point(x0, y0), Point(x0 + size, y0),
                Point(x0 + size, y0 + size), Point(x0, y0 + size)],
        classification=classification, color=AnnotationColor.default(),
        is_subtractive=subtractive)


class TestWhiteFraction:
    def test_all_white(self):
        assert white_fraction(np.full((8, 8, 3), 255, np.uint8)) == 1.0

    def test_all_tissue(self):
        assert white_fraction(np.full((8, 8, 3), 100, np.uint8)) == 0.0

    def test_half(self):
        patch = np.full((10, 10, 3), 100, np.uint8)
        patch[:5] = 255
        assert white_fraction(patch) == pytest.approx(0.5)

    def test_needs_all_three_channels(self):
        """A saturated red pixel is not background."""
        patch = np.zeros((4, 4, 3), np.uint8)
        patch[..., 0] = 255
        assert white_fraction(patch) == 0.0

    def test_empty(self):
        assert white_fraction(np.zeros((0, 0, 3), np.uint8)) == 1.0


class TestExtraction:
    def test_saves_patches_to_the_bank(self, slide, bank):
        extractor = FakeExtractor()
        settings = ExtractionSettings(patch_size_level=224, stride_level=224,
                                      max_white_fraction=1.0)
        report = extract_annotations(slide, [region(500, 500, 900)], extractor,
                                     bank, settings)
        assert report.saved > 0
        assert len(bank.fetch()) == report.saved

    def test_stamps_identity_and_metadata(self, slide, bank):
        extractor = FakeExtractor(dim=16, name="stamped")
        annotation = region(500, 500, 700, classification="PaNIN-3")
        extract_annotations(slide, [annotation], extractor, bank,
                            ExtractionSettings(max_white_fraction=1.0))
        patch = bank.fetch()[0]
        assert patch.extractor_identity == "test:stamped:r1"
        assert patch.classification == "PaNIN-3"
        assert patch.annotation_id == annotation.id
        assert patch.slide_name == slide.name
        assert patch.feature_dim == 16
        assert patch.patch_size_level == 224

    def test_coordinates_are_level0_and_inside(self, slide, bank):
        annotation = region(500, 500, 900)
        extract_annotations(slide, [annotation], FakeExtractor(), bank,
                            ExtractionSettings(max_white_fraction=1.0))
        for patch in bank.fetch():
            centre_x = patch.patch_x + patch.size_level0 / 2
            centre_y = patch.patch_y + patch.size_level0 / 2
            assert annotation.contains(centre_x, centre_y)

    def test_white_filter_rejects(self, slide, bank):
        """max_white_fraction=0 keeps only patches with no background at all."""
        settings = ExtractionSettings(max_white_fraction=0.0)
        report = extract_annotations(slide, [region(200, 200, 1200)],
                                     FakeExtractor(), bank, settings)
        assert report.skipped_white > 0
        assert all(p.white_fraction <= 0.0 for p in bank.fetch())

    def test_white_filter_is_recorded_per_patch(self, slide, bank):
        extract_annotations(slide, [region(500, 500, 700)], FakeExtractor(), bank,
                            ExtractionSettings(max_white_fraction=1.0))
        assert all(0.0 <= p.white_fraction <= 1.0 for p in bank.fetch())

    def test_nucleus_filter_runs_and_records(self, slide, bank):
        settings = ExtractionSettings(patch_size_level=224, stride_level=448,
                                      max_white_fraction=1.0, min_nuclei=1)
        extract_annotations(slide, [region(500, 500, 900)], FakeExtractor(), bank,
                            settings, segmenter=ClassicalNucleusSegmenter())
        saved = bank.fetch()
        if saved:
            assert all(p.nucleus_count >= 1 for p in saved)

    def test_nucleus_count_is_minus_one_when_not_measured(self, slide, bank):
        extract_annotations(slide, [region(500, 500, 700)], FakeExtractor(), bank,
                            ExtractionSettings(max_white_fraction=1.0, min_nuclei=0))
        assert all(p.nucleus_count == -1 for p in bank.fetch())

    def test_subtractive_annotation_carves_out(self, slide, bank):
        big = region(400, 400, 1400)
        hole = region(400, 400, 700, subtractive=True)
        settings = ExtractionSettings(max_white_fraction=1.0)

        without = extract_annotations(slide, [big], FakeExtractor(), bank, settings)
        bank.clear()
        with_hole = extract_annotations(slide, [big], FakeExtractor(), bank, settings,
                                        subtractive=[hole])
        assert with_hole.saved < without.saved
        for patch in bank.fetch():
            centre = (patch.patch_x + patch.size_level0 / 2,
                      patch.patch_y + patch.size_level0 / 2)
            assert not hole.contains(*centre)

    def test_batching_covers_every_patch_exactly_once(self, slide, bank):
        extractor = FakeExtractor()
        settings = ExtractionSettings(max_white_fraction=1.0, batch_size=8)
        report = extract_annotations(slide, [region(500, 500, 900)], extractor,
                                     bank, settings)
        assert sum(extractor.calls) == report.saved
        assert max(extractor.calls) <= 8

    def test_patches_are_resized_to_the_model_input(self, slide, bank):
        """Sampling at 128 but embedding a 224-input model must resize."""
        extractor = FakeExtractor(size=(224, 224))
        settings = ExtractionSettings(patch_size_level=128, stride_level=256,
                                      max_white_fraction=1.0, allow_resize=True)
        extract_annotations(slide, [region(500, 500, 900)], extractor, bank, settings)
        assert extractor.seen
        assert extractor.seen[0].shape[1:3] == (224, 224)
        # ...but the bank records the size actually sampled, not the resized one.
        assert bank.fetch()[0].patch_size_level == 128

    def test_resize_can_be_refused(self, slide, bank):
        extractor = FakeExtractor(size=(224, 224))
        settings = ExtractionSettings(patch_size_level=128, stride_level=256,
                                      max_white_fraction=1.0, allow_resize=False)
        report = extract_annotations(slide, [region(500, 500, 900)], extractor,
                                     bank, settings)
        assert report.saved == 0

    def test_multiple_classes_are_counted_separately(self, slide, bank):
        annotations = [region(400, 400, 600, "A"), region(1600, 400, 600, "B")]
        report = extract_annotations(slide, annotations, FakeExtractor(), bank,
                                     ExtractionSettings(max_white_fraction=1.0))
        assert set(report.by_class) == {"A", "B"}
        assert sum(report.by_class.values()) == report.saved

    def test_progress_is_monotonic_and_completes(self, slide, bank):
        seen: list[tuple[int, int]] = []
        extract_annotations(slide, [region(500, 500, 900)], FakeExtractor(), bank,
                            ExtractionSettings(max_white_fraction=1.0),
                            progress=lambda d, t, m: seen.append((d, t)))
        assert seen
        assert [d for d, _ in seen] == sorted(d for d, _ in seen)
        assert seen[-1][0] == seen[-1][1]

    def test_cancellation_stops_early(self, slide, bank):
        calls = {"n": 0}
        def cancel():
            calls["n"] += 1
            return calls["n"] > 2
        report = extract_annotations(slide, [region(300, 300, 1500)], FakeExtractor(),
                                     bank, ExtractionSettings(max_white_fraction=1.0),
                                     should_cancel=cancel)
        assert report.cancelled

    def test_no_annotations(self, slide, bank):
        report = extract_annotations(slide, [], FakeExtractor(), bank,
                                     ExtractionSettings())
        assert report.saved == 0 and report.considered == 0

    def test_annotation_smaller_than_a_patch(self, slide, bank):
        report = extract_annotations(slide, [region(500, 500, 50)], FakeExtractor(),
                                     bank, ExtractionSettings(patch_size_level=224))
        assert report.saved == 0

    @pytest.mark.parametrize("bad", [
        dict(patch_size_level=0), dict(stride_level=0), dict(max_white_fraction=2.0),
    ])
    def test_invalid_settings_raise(self, slide, bank, bad):
        with pytest.raises(ValueError):
            extract_annotations(slide, [region(0, 0, 500)], FakeExtractor(), bank,
                                ExtractionSettings(**bad))


class TestEstimate:
    def test_matches_actual_extraction(self, slide, bank):
        """The sheet's preview must not lie about how much work is coming."""
        annotation = region(500, 500, 900)
        settings = ExtractionSettings(max_white_fraction=1.0)
        estimate = estimate_patch_count(slide, [annotation], settings)
        report = extract_annotations(slide, [annotation], FakeExtractor(), bank, settings)
        assert estimate == report.considered

    def test_ignores_subtractive_annotations(self, slide):
        settings = ExtractionSettings()
        assert estimate_patch_count(slide, [region(0, 0, 900, subtractive=True)],
                                    settings) == 0

    def test_stride_halving_roughly_quadruples(self, slide):
        annotation = region(500, 500, 1400)
        coarse = estimate_patch_count(slide, [annotation],
                                      ExtractionSettings(stride_level=448))
        fine = estimate_patch_count(slide, [annotation],
                                    ExtractionSettings(stride_level=224))
        assert 3.0 < fine / max(coarse, 1) < 5.0


class TestLiveExtractor:
    """The real chain, with the real installed model."""

    def test_end_to_end(self, slide, bank):
        from pathlearn.extractors.registry import ExtractorRegistry

        registry = ExtractorRegistry()
        if registry.is_empty:
            pytest.skip("no extractors installed")
        descriptor = registry.by_name("phikon-v1") or registry.descriptors[0]
        extractor = registry.open(descriptor.identity)

        settings = ExtractionSettings(patch_size_level=descriptor.input_width,
                                      stride_level=descriptor.input_width * 2,
                                      max_white_fraction=1.0, batch_size=8)
        report = extract_annotations(slide, [region(400, 400, 1400)], extractor,
                                     bank, settings)
        registry.close()

        assert report.saved > 0
        patches = bank.fetch()
        assert all(p.feature_dim == descriptor.feature_dim for p in patches)
        assert all(p.extractor_identity == str(descriptor.identity) for p in patches)

        matrix, labels = bank.feature_matrix(patches)
        assert matrix.shape == (len(patches), descriptor.feature_dim)
        assert np.all(np.isfinite(matrix))
        # Distinct tissue must not collapse to one point.
        assert matrix.std(axis=0).mean() > 0
