"""Saving annotations as JPEGs."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.store import AnnotationStore
from pathlearn.pipeline.export_images import (ExportSettings, _safe, _unique,
                                              export_annotation_images)
from synthetic_slide import write_synthetic_slide


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("export") / "case-01.tif"
    write_synthetic_slide(path, 4096, 3072, levels=4)
    with SlideImage(path) as s:
        yield s


def box(x, y, size=600, cls="PanIN-2", name=None):
    return Annotation(points=[Point(x, y), Point(x + size, y),
                              Point(x + size, y + size), Point(x, y + size)],
                      classification=cls, color=AnnotationColor(200, 60, 60),
                      name=name)


def triangle(cls="PanIN-2"):
    return Annotation(points=[Point(400, 400), Point(1000, 400), Point(700, 1000)],
                      classification=cls, color=AnnotationColor(200, 60, 60))


class TestWriting:
    def test_one_file_per_annotation(self, slide, tmp_path):
        report = export_annotation_images(slide, [box(200, 200), box(1500, 900)],
                                          tmp_path)
        assert len(report.written) == 2
        assert all(i.path.is_file() for i in report.written)

    def test_the_files_are_real_jpegs(self, slide, tmp_path):
        report = export_annotation_images(slide, [box(200, 200)], tmp_path)
        with Image.open(report.written[0].path) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"

    def test_the_pixels_come_from_the_slide(self, slide, tmp_path):
        """Not a canvas screenshot: compared against a direct slide read."""
        report = export_annotation_images(
            slide, [box(200, 200, size=512)], tmp_path,
            ExportSettings(level=0, folder_per_class=False, quality=100))
        expected = slide.read_region(200, 200, 0, 512, 512).astype(np.int16)
        with Image.open(report.written[0].path) as image:
            got = np.asarray(image, dtype=np.int16)
        assert got.shape == expected.shape
        # JPEG is lossy even at quality 100, so compare on mean error.
        assert np.abs(got - expected).mean() < 6

    def test_folders_per_class(self, slide, tmp_path):
        export_annotation_images(
            slide, [box(200, 200, cls="PanIN-1a"), box(1500, 900, cls="PanIN-3")],
            tmp_path, ExportSettings(folder_per_class=True))
        assert (tmp_path / "PanIN-1a").is_dir()
        assert (tmp_path / "PanIN-3").is_dir()

    def test_flat_when_asked(self, slide, tmp_path):
        export_annotation_images(slide, [box(200, 200)], tmp_path,
                                 ExportSettings(folder_per_class=False))
        assert list(tmp_path.glob("*.jpg"))
        assert not [p for p in tmp_path.iterdir() if p.is_dir()]

    def test_the_name_carries_slide_class_and_label(self, slide, tmp_path):
        report = export_annotation_images(slide, [box(200, 200, name="P2 - 6")],
                                          tmp_path)
        name = report.written[0].path.name
        assert "case-01" in name and "PanIN-2" in name and "P2 - 6" in name

    def test_unnamed_annotations_are_numbered(self, slide, tmp_path):
        report = export_annotation_images(slide, [box(200, 200), box(1500, 900)],
                                          tmp_path)
        names = [i.path.stem for i in report.written]
        assert any(n.endswith("001") for n in names)
        assert any(n.endswith("002") for n in names)

    def test_nothing_is_overwritten(self, slide, tmp_path):
        first = export_annotation_images(slide, [box(200, 200, name="same")], tmp_path)
        second = export_annotation_images(slide, [box(900, 900, name="same")], tmp_path)
        assert first.written[0].path != second.written[0].path
        assert first.written[0].path.is_file()

    def test_an_empty_selection_writes_nothing(self, slide, tmp_path):
        report = export_annotation_images(slide, [], tmp_path)
        assert report.written == []
        assert "Nothing" in report.summary()


class TestGeometry:
    def test_the_crop_matches_the_bounding_box(self, slide, tmp_path):
        report = export_annotation_images(
            slide, [box(300, 400, size=512)], tmp_path,
            ExportSettings(level=0, folder_per_class=False))
        assert (report.written[0].width, report.written[0].height) == (512, 512)

    def test_margin_widens_the_crop(self, slide, tmp_path):
        report = export_annotation_images(
            slide, [box(300, 400, size=512)], tmp_path,
            ExportSettings(level=0, margin=64, folder_per_class=False))
        assert report.written[0].width == 512 + 128

    def test_a_coarser_level_is_smaller(self, slide, tmp_path):
        fine = export_annotation_images(slide, [box(300, 400, size=1024)],
                                        tmp_path / "a", ExportSettings(level=0))
        coarse = export_annotation_images(slide, [box(300, 400, size=1024)],
                                          tmp_path / "b", ExportSettings(level=2))
        assert coarse.written[0].width < fine.written[0].width

    def test_the_size_cap_is_respected(self, slide, tmp_path):
        report = export_annotation_images(
            slide, [box(0, 0, size=3000)], tmp_path,
            ExportSettings(level=0, max_edge=256))
        assert max(report.written[0].width, report.written[0].height) <= 256

    def test_automatic_level_keeps_big_regions_manageable(self, slide, tmp_path):
        report = export_annotation_images(slide, [box(0, 0, size=3000)], tmp_path,
                                          ExportSettings(max_edge=512))
        assert report.written[0].level > 0
        assert max(report.written[0].width, report.written[0].height) <= 512

    def test_a_small_region_stays_at_full_resolution(self, slide, tmp_path):
        report = export_annotation_images(slide, [box(200, 200, size=300)], tmp_path,
                                          ExportSettings(max_edge=4096))
        assert report.written[0].level == 0


class TestOverlays:
    def test_masking_blanks_the_outside(self, slide, tmp_path):
        report = export_annotation_images(
            slide, [triangle()], tmp_path,
            ExportSettings(level=0, mask_outside=True, folder_per_class=False))
        with Image.open(report.written[0].path) as image:
            pixels = np.asarray(image)
        # Bottom-left of the crop. Not the top-left: (400,400) is a vertex, so
        # the interior starts within a couple of pixels of that corner.
        assert pixels[-5, 5].min() > 240, "outside the outline should be white"
        # And the middle, which is inside, must not have been blanked too.
        centre = pixels[pixels.shape[0] // 2, pixels.shape[1] // 2]
        assert centre.min() < 240, "the region itself was blanked"

    def test_masking_actually_changes_the_image(self, slide, tmp_path):
        masked = export_annotation_images(slide, [triangle()], tmp_path / "m",
                                          ExportSettings(level=0, mask_outside=True))
        plain = export_annotation_images(slide, [triangle()], tmp_path / "p",
                                         ExportSettings(level=0))
        with Image.open(masked.written[0].path) as a, \
                Image.open(plain.written[0].path) as b:
            assert np.asarray(a).shape == np.asarray(b).shape
            assert not np.array_equal(np.asarray(a), np.asarray(b))

    def test_drawing_the_outline_changes_the_image(self, slide, tmp_path):
        plain = export_annotation_images(slide, [box(300, 300)], tmp_path / "a",
                                         ExportSettings(level=0))
        drawn = export_annotation_images(slide, [box(300, 300)], tmp_path / "b",
                                         ExportSettings(level=0, draw_outline=True))
        with Image.open(plain.written[0].path) as a, \
                Image.open(drawn.written[0].path) as b:
            assert not np.array_equal(np.asarray(a), np.asarray(b))


class TestRefusals:
    def test_a_two_point_outline_is_skipped(self, slide, tmp_path):
        flat = Annotation(points=[Point(0, 0), Point(100, 0)],
                          classification="x", color=AnnotationColor.default())
        report = export_annotation_images(slide, [flat], tmp_path)
        assert report.written == []

    def test_a_zero_height_outline_is_reported_not_crashed(self, slide, tmp_path):
        line = Annotation(points=[Point(10, 10), Point(200, 10), Point(300, 10)],
                          classification="x", color=AnnotationColor.default())
        report = export_annotation_images(slide, [line], tmp_path)
        assert len(report.written) + len(report.skipped) == 1

    def test_cancelling_stops_early(self, slide, tmp_path):
        report = export_annotation_images(slide, [box(200, 200), box(1500, 900)],
                                          tmp_path, should_cancel=lambda: True)
        assert report.cancelled and report.written == []

    def test_progress_is_reported(self, slide, tmp_path):
        seen = []
        export_annotation_images(slide, [box(200, 200), box(1500, 900)], tmp_path,
                                 progress=lambda d, t, m: seen.append((d, t)))
        assert seen[0] == (0, 2) and seen[-1] == (2, 2)


class TestNaming:
    @pytest.mark.parametrize("raw,expected", [
        ("PanIN-1a", "PanIN-1a"),
        ("grade 2/3", "grade 2-3"),
        ("   ", "unlabelled"),
        ("", "unlabelled"),
    ])
    def test_forbidden_characters_are_replaced(self, raw, expected):
        assert _safe(raw) == expected

    def test_every_windows_reserved_character_is_handled(self):
        cleaned = _safe('a<b>c:d"e/f\\g|h?i*j')
        assert not any(c in cleaned for c in '<>:"/\\|?*')

    def test_unique_numbers_collisions(self, tmp_path):
        used = set()
        first = _unique(tmp_path, "a.jpg", used)
        second = _unique(tmp_path, "a.jpg", used)
        assert first.name == "a.jpg" and second.name == "a-2.jpg"

    def test_a_class_with_a_slash_still_exports(self, slide, tmp_path):
        """Class names are user-typed; a slash would otherwise mean a subfolder."""
        report = export_annotation_images(slide, [box(200, 200, cls="PanIN 2/3")],
                                          tmp_path)
        assert len(report.written) == 1
        assert report.written[0].path.is_file()


class TestSheet:
    @pytest.fixture
    def sheet(self, qtbot, slide, tmp_path):
        from pathlearn.ui.sheets.export_images import ExportImagesSheet

        store = AnnotationStore()
        store.bind(tmp_path / "case-01.svs", slide.dimensions.height)
        store.add(box(200, 200, cls="PanIN-1a"))
        store.add(box(1500, 900, cls="PanIN-3"))
        widget = ExportImagesSheet(slide, store)
        qtbot.addWidget(widget)
        yield widget
        widget.reject()

    def test_it_counts_the_checked_annotations(self, sheet):
        assert sheet.export_button.text() == "Export 2 Image(s)"
        assert "2 checked" in sheet.status.text()

    def test_unchecked_ones_are_called_out(self, sheet):
        sheet.store.set_selected(sheet.store.annotations[0].id, False)
        sheet._update_summary()
        assert "1 unchecked" in sheet.status.text()
        assert sheet.export_button.text() == "Export 1 Image(s)"

    def test_the_default_folder_sits_beside_the_slide(self, sheet):
        assert "case-01-annotations" in sheet.folder_edit.text()

    def test_settings_reflect_the_controls(self, sheet):
        sheet.margin.setValue(32)
        sheet.quality.setValue(80)
        sheet.mask_outside.setChecked(True)
        settings = sheet.settings()
        assert settings.margin == 32
        assert settings.quality == 80
        assert settings.mask_outside

    def test_automatic_resolution_is_the_default(self, sheet):
        assert sheet.settings().level is None

    def test_export_is_blocked_without_a_folder(self, sheet):
        sheet.folder_edit.setText("")
        assert not sheet.export_button.isEnabled()
