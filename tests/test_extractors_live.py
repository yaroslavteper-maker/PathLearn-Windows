"""Tests against the real installed ONNX models.

These skip when the extractors are not installed, so the suite still runs on a
machine without the 4 GB of model files.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.extractors.onnx_extractor import available_providers, best_provider
from pathlearn.extractors.registry import ExtractorRegistry


@pytest.fixture(scope="module")
def registry():
    reg = ExtractorRegistry()
    if reg.is_empty:
        pytest.skip(f"No extractors installed in {reg.directories}")
    yield reg
    reg.close()


class TestInstalledExtractors:
    def test_all_descriptors_validate(self, registry):
        assert not registry.problems, registry.describe()

    def test_each_returns_its_declared_dimension(self, registry):
        """The descriptor's featureDim must match what the model emits."""
        rng = np.random.default_rng(0)
        for descriptor in registry:
            extractor = registry.open(descriptor.identity)
            patch = rng.integers(0, 256, (descriptor.input_height,
                                          descriptor.input_width, 3), dtype=np.uint8)
            assert extractor.extract_one(patch).shape == (descriptor.feature_dim,)

    def test_batch_matches_individual(self, registry):
        """Batching must not change results *materially*.

        It does change them slightly: a different batch size makes cuDNN pick
        different kernels, whose reduction orders differ, so embeddings vary in
        roughly the 4th decimal (~2e-4 absolute). That is inherent to GPU
        inference, not a bug — but it does mean a bank extracted at batch 8 is
        not bit-identical to one extracted at batch 16.

        Cosine is the right yardstick because it is what every downstream
        consumer uses (null exclusion, centroids, the softmax classifier).
        """
        descriptor = registry.descriptors[0]
        extractor = registry.open(descriptor.identity)
        rng = np.random.default_rng(1)
        patches = rng.integers(0, 256, (5, descriptor.input_height,
                                        descriptor.input_width, 3), dtype=np.uint8)
        batched = extractor.extract(patches)
        one_at_a_time = np.stack([extractor.extract_one(p) for p in patches])

        for a, b in zip(batched, one_at_a_time):
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            assert cosine > 0.999999, f"batching changed the embedding: cos={cosine}"
        # And the absolute drift stays in the noise band, not the signal band.
        assert np.abs(batched - one_at_a_time).max() < 1e-2

    def test_batch_smaller_and_larger_than_batch_size(self, registry):
        """Exercise the chunking loop on both sides of batch_size."""
        descriptor = registry.descriptors[0]
        extractor = registry.open(descriptor.identity, batch_size=4)
        rng = np.random.default_rng(2)
        for n in (1, 3, 4, 9):
            patches = rng.integers(0, 256, (n, descriptor.input_height,
                                            descriptor.input_width, 3), dtype=np.uint8)
            assert extractor.extract(patches).shape == (n, descriptor.feature_dim)

    def test_empty_input(self, registry):
        extractor = registry.open(registry.descriptors[0].identity)
        assert extractor.extract(np.zeros((0, 224, 224, 3), np.uint8)).shape[0] == 0

    def test_wrong_patch_size_is_rejected(self, registry):
        """A silently resized patch would corrupt the feature space."""
        descriptor = registry.descriptors[0]
        extractor = registry.open(descriptor.identity)
        with pytest.raises(ValueError, match="expected patches"):
            extractor.extract(np.zeros((1, 64, 64, 3), dtype=np.uint8))

    def test_identical_input_gives_identical_output(self, registry):
        extractor = registry.open(registry.descriptors[0].identity)
        patch = np.full((224, 224, 3), 128, dtype=np.uint8)
        assert np.array_equal(extractor.extract_one(patch), extractor.extract_one(patch))

    def test_different_input_gives_different_output(self, registry):
        extractor = registry.open(registry.descriptors[0].identity)
        rng = np.random.default_rng(3)
        a = extractor.extract_one(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        b = extractor.extract_one(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        assert not np.allclose(a, b)

    def test_embeddings_are_finite(self, registry):
        """White and black patches are the realistic extremes on a slide."""
        for descriptor in registry:
            extractor = registry.open(descriptor.identity)
            for fill in (0, 255):
                patch = np.full((descriptor.input_height, descriptor.input_width, 3),
                                fill, dtype=np.uint8)
                assert np.all(np.isfinite(extractor.extract_one(patch)))


class TestExecutionProvider:
    def test_a_provider_is_available(self):
        assert available_providers()

    def test_best_provider_is_offered(self):
        assert best_provider() in available_providers()

    def test_session_reports_its_actual_provider(self, registry):
        """Availability is not activation — the session must be asked directly."""
        extractor = registry.open(registry.descriptors[0].identity)
        assert extractor.provider in available_providers()

    def test_only_one_gpu_session_is_held(self, registry):
        """Concurrent GPU sessions made throughput depend on load order."""
        if len(registry) < 2:
            pytest.skip("need two extractors")
        for descriptor in registry:
            registry.open(descriptor.identity)
        gpu = [e for e in registry._cache.values()
               if e.provider != "CPUExecutionProvider"]
        assert len(gpu) <= 1
