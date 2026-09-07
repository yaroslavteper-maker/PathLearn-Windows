"""Panels derived from the annotation store must follow it, not just the slide.

The bug this covers: importing a GeoJSON into a slide that had no sidecar
filled the sidebar and the canvas, but the Geometry panel kept the state it
had at slide-open — describe button disabled, reading "Describe 0 Checked
Annotation(s)". The annotations were there; the panel had never been told.

These drive ``MainWindow`` rather than the panel alone on purpose: the panel's
own refresh was already correct, and the defect was entirely in the wiring.
"""

from __future__ import annotations

import pytest

from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from synthetic_slide import write_synthetic_slide


class _Settings:
    """Stand-in for QSettings so a test never touches the user's registry."""

    def __init__(self, *_args) -> None:
        self._values: dict[str, object] = {}

    def value(self, key, default=None):
        return self._values.get(key, default)

    def setValue(self, key, value) -> None:
        self._values[key] = value


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    import pathlearn.data.bank as bank_module
    import pathlearn.data.geometry_bank as geometry_module
    import pathlearn.ui.main_window as main_window

    monkeypatch.setattr(main_window, "QSettings", _Settings)
    monkeypatch.setattr(bank_module, "default_bank_path",
                        lambda: tmp_path / "bank.db")
    monkeypatch.setattr(geometry_module, "default_geometry_bank_path",
                        lambda: tmp_path / "geometry.json")

    widget = main_window.MainWindow()
    qtbot.addWidget(widget)
    yield widget
    widget.close()


@pytest.fixture
def slide_path(tmp_path_factory):
    """A slide with no sidecar beside it — the case that exposed the bug."""
    path = tmp_path_factory.mktemp("wiring") / "unannotated.tif"
    write_synthetic_slide(path, 2048, 1536, levels=3)
    return path


def polygons(count, height):
    return [Annotation(points=[Point(i * 200 + 20, 20), Point(i * 200 + 180, 20),
                               Point(i * 200 + 180, min(400, height - 20)),
                               Point(i * 200 + 20, min(400, height - 20))],
                       classification="PaNIN-2", color=AnnotationColor.default())
            for i in range(count)]


class TestGeometryPanelFollowsTheStore:
    def test_starts_empty_on_a_slide_without_a_sidecar(self, window, slide_path):
        window.open_slide(slide_path)
        assert not window.geometry_panel.describe_button.isEnabled()
        assert "Describe 0" in window.geometry_panel.describe_button.text()

    def test_import_enables_describe(self, window, slide_path):
        window.open_slide(slide_path)
        window.store.import_merge(polygons(3, window.slide.dimensions.height),
                                  replace=True)
        button = window.geometry_panel.describe_button
        assert button.isEnabled()
        assert "Describe 3 Checked Annotation(s)" == button.text()

    def test_appending_an_import_updates_the_count(self, window, slide_path):
        window.open_slide(slide_path)
        height = window.slide.dimensions.height
        window.store.import_merge(polygons(2, height), replace=True)
        window.store.import_merge(polygons(3, height), replace=False)
        assert "Describe 5" in window.geometry_panel.describe_button.text()

    def test_drawing_one_updates_the_panel(self, window, slide_path):
        window.open_slide(slide_path)
        window.store.add(polygons(1, window.slide.dimensions.height)[0])
        assert window.geometry_panel.describe_button.isEnabled()
        assert "Describe 1" in window.geometry_panel.describe_button.text()

    def test_unticking_is_reflected(self, window, slide_path):
        window.open_slide(slide_path)
        window.store.import_merge(polygons(3, window.slide.dimensions.height),
                                  replace=True)
        window.store.set_selected(window.store.annotations[0].id, False)
        text = window.geometry_panel.describe_button.text()
        assert "Describe 2 Checked Annotation(s)" in text
        assert "(1 unchecked)" in text

    def test_deleting_everything_disables_describe(self, window, slide_path):
        window.open_slide(slide_path)
        window.store.import_merge(polygons(3, window.slide.dimensions.height),
                                  replace=True)
        window.store.remove_all()
        assert not window.geometry_panel.describe_button.isEnabled()
        assert "Describe 0" in window.geometry_panel.describe_button.text()

    def test_the_legacy_visible_flag_does_not_change_what_is_described(
            self, window, slide_path):
        """Only Use gates analysis. ``isVisible`` is carried, never consulted."""
        window.open_slide(slide_path)
        window.store.import_merge(polygons(3, window.slide.dimensions.height),
                                  replace=True)
        for annotation in list(window.store.annotations):
            annotation.is_visible = False
        assert "Describe 3 Checked Annotation(s)" in \
            window.geometry_panel.describe_button.text()

    def test_closing_the_slide_clears_the_panel(self, window, slide_path):
        window.open_slide(slide_path)
        window.store.import_merge(polygons(3, window.slide.dimensions.height),
                                  replace=True)
        window.close_slide()
        assert not window.geometry_panel.describe_button.isEnabled()


class TestSidecarRoundTripThroughTheWindow:
    def test_reopening_finds_the_imported_annotations(self, window, slide_path):
        window.open_slide(slide_path)
        window.store.import_merge(polygons(4, window.slide.dimensions.height),
                                  replace=True)
        window.close_slide()
        window.open_slide(slide_path)
        assert len(window.store.in_use()) == 4
        assert "Describe 4" in window.geometry_panel.describe_button.text()


class TestExtractorsDialog:
    """Open Folder and the execution-provider report.

    Both exist for the same reason: the failure modes here are silent. A fresh
    install has no extractors folder at all, and a CPU-only onnxruntime runs
    ~20x slower without ever erroring.
    """

    def test_open_folder_creates_it_when_absent(self, window, tmp_path, monkeypatch):
        import pathlearn.ui.main_window as main_window

        target = tmp_path / "Extractors"
        window.registry.directories = [target]
        opened = []
        monkeypatch.setattr(main_window.QDesktopServices, "openUrl",
                            lambda url: opened.append(url.toLocalFile()))
        assert not target.exists()
        assert window.open_extractors_folder() == target
        assert target.is_dir()
        assert opened and "Extractors" in opened[0]

    def test_open_folder_is_happy_when_it_exists(self, window, tmp_path, monkeypatch):
        import pathlearn.ui.main_window as main_window

        target = tmp_path / "Extractors"
        target.mkdir()
        (target / "keep.txt").write_text("x", encoding="utf-8")
        window.registry.directories = [target]
        monkeypatch.setattr(main_window.QDesktopServices, "openUrl", lambda url: None)
        window.open_extractors_folder()
        assert (target / "keep.txt").exists(), "must not clobber existing contents"

    def test_provider_summary_names_the_active_provider(self, window):
        summary = window._provider_summary()
        assert "Execution provider:" in summary
        assert "ExecutionProvider" in summary

    def test_cpu_only_is_called_out(self, window, monkeypatch):
        import pathlearn.extractors.onnx_extractor as onnx_extractor

        monkeypatch.setattr(onnx_extractor, "available_providers",
                            lambda: ["CPUExecutionProvider"])
        monkeypatch.setattr(onnx_extractor, "best_provider",
                            lambda *a, **k: "CPUExecutionProvider")
        summary = window._provider_summary()
        assert "CPU only" in summary
        assert "onnxruntime-gpu" in summary

    def test_gpu_is_not_nagged_about(self, window, monkeypatch):
        import pathlearn.extractors.onnx_extractor as onnx_extractor

        monkeypatch.setattr(onnx_extractor, "available_providers",
                            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        monkeypatch.setattr(onnx_extractor, "best_provider",
                            lambda *a, **k: "CUDAExecutionProvider")
        summary = window._provider_summary()
        assert "CUDAExecutionProvider" in summary
        assert "CPU only" not in summary

    def test_a_broken_runtime_is_reported_not_raised(self, window, monkeypatch):
        import pathlearn.extractors.onnx_extractor as onnx_extractor

        def boom():
            raise RuntimeError("cuDNN not found")

        monkeypatch.setattr(onnx_extractor, "available_providers", boom)
        summary = window._provider_summary()
        assert "cuDNN not found" in summary
