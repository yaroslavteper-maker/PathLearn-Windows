"""SlideImage tests, run against a generated pyramidal TIFF.

The orientation tests are the important ones: they assert that a level-0
coordinate lands on the pixel a top-left origin says it should, at every
pyramid level.  If anyone ever reintroduces the macOS Y mirror, these fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.io.slide import SlideError, SlideImage, detect_format
from synthetic_slide import CORNER_COLORS, MARKER, corner_centre, write_synthetic_slide

WIDTH, HEIGHT = 4096, 3072


@pytest.fixture(scope="module")
def slide_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("slides") / "synthetic.tif"
    write_synthetic_slide(path, WIDTH, HEIGHT, levels=4)
    return path


@pytest.fixture(scope="module")
def slide(slide_path):
    with SlideImage(slide_path) as s:
        yield s


def dominant_color(patch: np.ndarray) -> tuple[int, int, int]:
    """Median colour of a patch, robust to a stray edge pixel."""
    flat = patch.reshape(-1, 3)
    return tuple(int(v) for v in np.median(flat, axis=0))


def close_to(actual, expected, tolerance: int = 30) -> bool:
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


class TestMetadata:
    def test_detect_format(self, slide_path):
        assert detect_format(slide_path) == "generic-tiff"

    def test_dimensions_and_levels(self, slide):
        assert slide.dimensions.as_tuple() == (WIDTH, HEIGHT)
        assert slide.level_count == 4
        assert slide.level_downsamples == pytest.approx((1.0, 2.0, 4.0, 8.0))
        assert slide.level_dimensions[3].as_tuple() == (512, 384)

    def test_mpp(self, slide):
        assert slide.mpp_x == pytest.approx(0.5, abs=0.01)

    def test_best_level_for_downsample(self, slide):
        assert slide.best_level_for_downsample(1.0) == 0
        assert slide.best_level_for_downsample(4.0) == 2
        assert slide.best_level_for_downsample(1000.0) == 3

    def test_level_for_mpp(self, slide):
        """The geometry pipeline needs ~1 um/px; at 0.5 um/px base that is level 1."""
        assert slide.level_for_mpp(1.0) == 1
        assert slide.level_for_mpp(0.5) == 0
        assert slide.level_for_mpp(4.0) == 3

    def test_unsupported_file_raises(self, tmp_path):
        bogus = tmp_path / "notaslide.svs"
        bogus.write_bytes(b"definitely not a slide")
        with pytest.raises(SlideError):
            SlideImage(bogus)


class TestOrientation:
    """The port's single highest-risk property: no vertical mirror, anywhere."""

    @pytest.mark.parametrize("corner", list(CORNER_COLORS))
    def test_corner_markers_at_level_0(self, slide, corner):
        cx, cy = corner_centre(corner, WIDTH, HEIGHT)
        half = MARKER // 4
        patch = slide.read_region(cx - half, cy - half, 0, 2 * half, 2 * half)
        assert close_to(dominant_color(patch), CORNER_COLORS[corner]), (
            f"{corner} read as {dominant_color(patch)}, expected {CORNER_COLORS[corner]}"
        )

    @pytest.mark.parametrize("corner", list(CORNER_COLORS))
    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_corner_markers_at_every_level(self, slide, corner, level):
        """x/y stay level-0 while w/h are level-local — the OpenSlide asymmetry."""
        downsample = slide.level_downsamples[level]
        cx, cy = corner_centre(corner, WIDTH, HEIGHT)
        size_level = max(4, int(MARKER / 2 / downsample))
        offset0 = int(size_level * downsample / 2)
        patch = slide.read_region(cx - offset0, cy - offset0, level, size_level, size_level)
        assert close_to(dominant_color(patch), CORNER_COLORS[corner], tolerance=40), (
            f"{corner} at level {level} read as {dominant_color(patch)}"
        )

    def test_top_and_bottom_differ(self, slide):
        """A mirrored read would swap these two."""
        top = slide.read_region(WIDTH // 2, 0, 0, 256, 256)
        bottom = slide.read_region(WIDTH // 2, HEIGHT - 256, 0, 256, 256)
        assert not close_to(dominant_color(top), dominant_color(bottom))

    def test_thumbnail_matches_level_0_orientation(self, slide):
        """The overview and a direct read must agree on which corner is which."""
        thumb = slide.thumbnail(512)
        h, w, _ = thumb.shape
        box = max(4, min(h, w) // 12)
        for corner, expected in CORNER_COLORS.items():
            if corner == "top_left":
                patch = thumb[:box, :box]
            elif corner == "top_right":
                patch = thumb[:box, -box:]
            elif corner == "bottom_left":
                patch = thumb[-box:, :box]
            else:
                patch = thumb[-box:, -box:]
            assert close_to(dominant_color(patch), expected, tolerance=60), (
                f"thumbnail {corner} read as {dominant_color(patch)}"
            )


class TestReadRegion:
    def test_shape_and_dtype(self, slide):
        patch = slide.read_region(1000, 1000, 0, 300, 200)
        assert patch.shape == (200, 300, 3)
        assert patch.dtype == np.uint8

    def test_zero_size_is_empty_not_an_error(self, slide):
        assert slide.read_region(0, 0, 0, 0, 0).size == 0

    def test_out_of_bounds_reads_background_not_an_error(self, slide):
        """OpenSlide pads rather than failing; we must surface that as pixels."""
        patch = slide.read_region(WIDTH - 64, HEIGHT - 64, 0, 256, 256)
        assert patch.shape == (256, 256, 3)

    def test_alpha_is_composited_over_white(self, slide):
        """Fully transparent padding must read as white, not black."""
        patch = slide.read_region(WIDTH + 1000, HEIGHT + 1000, 0, 32, 32)
        assert close_to(dominant_color(patch), (255, 255, 255), tolerance=2)

    def test_downsampled_read_matches_level_0(self, slide):
        """Level 1 of a 2x box pyramid should track level 0's local average."""
        x, y = 1536, 1024
        fine = slide.read_region(x, y, 0, 128, 128).astype(np.float32).mean(axis=(0, 1))
        coarse = slide.read_region(x, y, 1, 64, 64).astype(np.float32).mean(axis=(0, 1))
        assert np.allclose(fine, coarse, atol=12)

    def test_rgba_variant_keeps_alpha(self, slide):
        assert slide.read_region_rgba(0, 0, 0, 16, 16).shape == (16, 16, 4)


class TestLifecycle:
    def test_read_after_close_raises(self, slide_path):
        s = SlideImage(slide_path)
        s.close()
        with pytest.raises(SlideError):
            s.read_region(0, 0, 0, 16, 16)

    def test_double_close_is_safe(self, slide_path):
        s = SlideImage(slide_path)
        s.close()
        s.close()

    def test_concurrent_reads(self, slide):
        """The tile pool reads from several threads against one handle."""
        from concurrent.futures import ThreadPoolExecutor

        def read(i: int):
            return slide.read_region(i * 64, i * 64, 0, 128, 128).shape

        with ThreadPoolExecutor(max_workers=8) as pool:
            shapes = list(pool.map(read, range(32)))
        assert all(s == (128, 128, 3) for s in shapes)
