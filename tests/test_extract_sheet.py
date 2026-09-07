"""Extract Patches sheet: state, preview, and the background run.

Drives the dialog headlessly with a stand-in extractor, so no model files or
GPU are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt

from pathlearn.data.bank import PatchBank
from pathlearn.extractors.descriptor import ExtractorDescriptor
from pathlearn.extractors.identity import ExtractorIdentity
from pathlearn.extractors.registry import ExtractorRegistry
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.store import AnnotationStore
from pathlearn.ui.sheets.extract_patches import ExtractPatchesSheet
from synthetic_slide import write_synthetic_slide
from test_extractors import write_extractor

WIDTH, HEIGHT = 4096, 3072


class StubExtractor:
    """Mimics OnnxFeatureExtractor without a model file."""

    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.identity = descriptor.identity
        self.feature_dim = descriptor.feature_dim
        self.input_size = descriptor.input_size
        self.provider = "CUDAExecutionProvider"

    def extract(self, patches):
        batch = np.asarray(patches)
        if batch.ndim == 3:
            batch = batch[None]
        return np.tile(batch.reshape(batch.shape[0], -1).mean(axis=1)[:, None],
                       (1, self.feature_dim)).astype(np.float32)

    def extract_one(self, patch):
        return self.extract(patch)[0]

    def close(self):
        pass


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("s") / "sheet.tif"
    write_synthetic_slide(path, WIDTH, HEIGHT, levels=4)
    with SlideImage(path) as s:
        yield s


@pytest.fixture
def registry(tmp_path, monkeypatch):
    write_extractor(tmp_path, "stub", dim=32)
    reg = ExtractorRegistry([tmp_path])
    monkeypatch.setattr(reg, "open",
                        lambda identity, **kw: StubExtractor(reg.by_identity(str(identity))))
    yield reg


@pytest.fixture
def store(tmp_path, slide):
    s = AnnotationStore()
    s.bind(tmp_path / "slide.svs", slide.dimensions.height)
    for i, (name, x) in enumerate((("Acinar", 400), ("Islets", 1800))):
        s.add(Annotation(
            points=[Point(x, 400), Point(x + 900, 400),
                    Point(x + 900, 1300), Point(x, 1300)],
            classification=name, color=AnnotationColor.default()))
    return s


@pytest.fixture
def bank(tmp_path):
    with PatchBank(tmp_path / "b.db") as b:
        yield b


@pytest.fixture
def sheet(qtbot, slide, store, registry, bank):
    dialog = ExtractPatchesSheet(slide, store, registry, bank)
    qtbot.addWidget(dialog)
    dialog._update_preview()
    yield dialog
    # Close before the bank fixture tears down, so the debounced preview timer
    # cannot fire against a closed connection.
    dialog.reject()


class TestInitialState:
    def test_lists_installed_extractors(self, sheet):
        assert sheet.extractor_combo.count() == 1
        assert sheet.current_descriptor is not None

    def test_patch_size_defaults_to_model_input(self, sheet):
        """Sampling at the model's input size avoids a silent resize."""
        assert sheet.patch_size.value() == sheet.current_descriptor.input_width
        assert sheet.stride.value() == sheet.current_descriptor.input_width

    def test_all_classes_checked_by_default(self, sheet):
        assert sheet.selected_classes == {"Acinar", "Islets"}

    def test_every_pyramid_level_offered(self, sheet, slide):
        assert sheet.level_combo.count() == slide.level_count

    def test_preview_is_positive(self, sheet):
        assert sheet._estimated > 0
        assert "patches" in sheet.summary.text()

    def test_extract_enabled(self, sheet):
        assert sheet.extract_button.isEnabled()


class TestPreview:
    def test_deselecting_classes_reduces_count(self, sheet):
        both = sheet._estimated
        sheet.class_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        sheet._update_preview()
        assert 0 < sheet._estimated < both

    def test_no_classes_disables_extract(self, sheet):
        sheet._set_all(Qt.CheckState.Unchecked)
        sheet._update_preview()
        assert sheet._estimated == 0
        assert not sheet.extract_button.isEnabled()

    def test_halving_stride_quadruples_count(self, sheet):
        """The reason the preview exists at all."""
        base = sheet._estimated
        sheet.stride.setValue(sheet.stride.value() // 2)
        sheet._update_preview()
        assert 3.0 < sheet._estimated / base < 5.0

    def test_overlap_is_flagged(self, sheet):
        sheet.stride.setValue(sheet.patch_size.value() // 2)
        sheet._update_preview()
        assert "Overlapping" in sheet.overlap_note.text()

    def test_coarser_level_reduces_count(self, sheet):
        base = sheet._estimated
        sheet.level_combo.setCurrentIndex(2)
        sheet._update_preview()
        assert sheet._estimated < base

    def test_preview_matches_actual(self, sheet, bank, qtbot):
        expected = sheet._estimated
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.report.considered == expected


class TestRun:
    def test_extraction_populates_the_bank(self, sheet, bank, qtbot):
        sheet.max_white.setValue(1.0)
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.report.saved > 0
        assert bank.stats().total == sheet.report.saved

    def test_both_classes_are_recorded(self, sheet, bank, qtbot):
        sheet.max_white.setValue(1.0)
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert set(bank.stats().by_class) == {"Acinar", "Islets"}

    def test_identity_is_stamped(self, sheet, bank, qtbot):
        sheet.max_white.setValue(1.0)
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert bank.extractor_identities == ["onnx:stub:r1"]

    def test_controls_are_re_enabled_after(self, sheet, qtbot):
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.patch_size.isEnabled()
        assert sheet.extract_button.text() == "Extract"
        assert not sheet.progress.isVisible()

    def test_progress_reaches_the_end(self, sheet, qtbot):
        seen: list[int] = []
        sheet.progress.valueChanged.connect(seen.append)
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert seen and max(seen) == sheet.progress.maximum()

    def test_white_filter_is_applied(self, sheet, bank, qtbot):
        sheet.max_white.setValue(0.0)
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert sheet.report.skipped_white > 0

    def test_second_run_appends(self, sheet, bank, qtbot):
        sheet.max_white.setValue(1.0)
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        first = bank.stats().total
        sheet._start()
        qtbot.waitUntil(lambda: sheet.task is None, timeout=60_000)
        assert bank.stats().total == 2 * first


class TestEmptyRegistry:
    def test_extract_disabled_without_extractors(self, qtbot, slide, store, bank, tmp_path):
        empty = ExtractorRegistry([tmp_path / "nothing"])
        dialog = ExtractPatchesSheet(slide, store, empty, bank)
        qtbot.addWidget(dialog)
        assert not dialog.extract_button.isEnabled()
        assert "Install a model" in dialog.extractor_note.text()
