"""Auto-saving and reloading the heatmap around a real slide open/close."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_slide import write_synthetic_slide

from pathlearn.models.prediction import (PatchPrediction, PredictionSet,
                                         auto_sidecar_path, sidecar_path)

SIZE = 224


class _Settings:
    """Stand-in for QSettings so a test never touches the user's registry."""

    def __init__(self, *_args) -> None:
        self._values: dict = {}

    def value(self, key, default=None):
        return self._values.get(key, default)

    def setValue(self, key, value) -> None:
        self._values[key] = value


def tile(col, label="PanIN-2", confidence=0.9):
    return PatchPrediction(x=col * SIZE, y=0, size_level0=SIZE, label=label,
                           probabilities=np.array([confidence, 0.0],
                                                  dtype=np.float32))


def a_set(count=4, slide="", **kw):
    return PredictionSet(predictions=[tile(i) for i in range(count)],
                         class_labels=["PanIN-2", "Acinar"],
                         slide_path=slide,
                         extractor_identity="onnx:uni2-h:r1", **kw)


@pytest.fixture(scope="session")
def slide_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("slides") / "case-01.tiff"
    write_synthetic_slide(path, width=2048, height=1536)
    return path


@pytest.fixture
def slide(slide_file, tmp_path):
    """A private copy, so each test owns its sidecars."""
    import shutil

    target = tmp_path / slide_file.name
    shutil.copy(slide_file, target)
    return target


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    import pathlearn.data.bank as bank_module
    import pathlearn.data.geometry_bank as geometry_module
    import pathlearn.ui.main_window as main_window

    monkeypatch.setattr(main_window, "QSettings", _Settings)
    monkeypatch.setattr(bank_module, "default_bank_path",
                        lambda: tmp_path / "live" / "bank.db")
    monkeypatch.setattr(geometry_module, "default_geometry_bank_path",
                        lambda: tmp_path / "live" / "geometry_bank.json")
    (tmp_path / "live").mkdir(exist_ok=True)

    widget = main_window.MainWindow()
    qtbot.addWidget(widget)
    yield widget
    widget.close()


class TestSaving:
    def test_it_writes_beside_the_slide(self, window, slide):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        assert window._save_heatmap_now() is True
        assert auto_sidecar_path(slide).exists()

    def test_it_stamps_the_slide_it_belongs_to(self, window, slide):
        """A sidecar copied elsewhere must still say what it was computed on."""
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set(slide=""))
        window._save_heatmap_now()
        stored = PredictionSet.load(auto_sidecar_path(slide))
        assert stored.slide_path == str(slide)

    def test_nothing_is_written_without_a_slide(self, window):
        window.heatmap_panel.set_predictions(a_set())
        assert window._save_heatmap_now() is False

    def test_nothing_is_written_without_predictions(self, window, slide):
        window.open_slide(slide)
        assert window._save_heatmap_now() is False
        assert not auto_sidecar_path(slide).exists()

    def test_turning_it_off_stops_new_saves(self, window, slide):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window.heatmap_panel.set_autosave(False)
        assert window._save_heatmap_now() is False
        assert not auto_sidecar_path(slide).exists()

    def test_turning_it_off_leaves_an_existing_file_alone(self, window, slide):
        """Stopping the saves must not be a delete in disguise."""
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window._save_heatmap_now()
        window.heatmap_panel.autosave_checkbox.setChecked(False)
        assert auto_sidecar_path(slide).exists()

    def test_a_read_only_target_does_not_raise(self, window, slide,
                                               monkeypatch):
        """A failed save is a status message, never a crash mid-workflow."""
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())

        def explode(self, path):
            raise OSError("read-only volume")

        monkeypatch.setattr(PredictionSet, "save", explode)
        assert window._save_heatmap_now() is False


class TestTheDebounce:
    def test_a_view_change_schedules_rather_than_writes(self, window, slide):
        """Dragging the confidence slider must not write on every tick."""
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        auto_sidecar_path(slide).unlink(missing_ok=True)
        window.heatmap_panel.threshold.setValue(50)
        assert window._heatmap_save_timer.isActive()
        assert not auto_sidecar_path(slide).exists()

    def test_the_pending_save_lands(self, window, slide, qtbot):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        auto_sidecar_path(slide).unlink(missing_ok=True)
        window.heatmap_panel.threshold.setValue(50)
        window._heatmap_save_timer.setInterval(10)
        window._heatmap_save_timer.start()
        qtbot.waitUntil(lambda: auto_sidecar_path(slide).exists(), timeout=2000)

    def test_nothing_is_scheduled_when_it_is_switched_off(self, window, slide):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window.heatmap_panel.set_autosave(False)
        window.heatmap_panel.threshold.setValue(50)
        assert not window._heatmap_save_timer.isActive()

    def test_closing_the_slide_flushes_a_pending_save(self, window, slide):
        """Otherwise the window forgets which slide it belonged to."""
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        auto_sidecar_path(slide).unlink(missing_ok=True)
        window.heatmap_panel.threshold.setValue(50)
        window.close_slide()
        assert auto_sidecar_path(slide).exists()


class TestLoading:
    def test_reopening_brings_the_heatmap_back(self, window, slide):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window._save_heatmap_now()
        window.close_slide()
        assert window.heatmap_panel.predictions is None

        window.open_slide(slide)
        assert window.heatmap_panel.predictions is not None
        assert len(window.heatmap_panel.predictions) == 4

    def test_the_view_state_comes_back_with_it(self, window, slide):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window.heatmap_panel.threshold.setValue(60)
        window.heatmap_panel.predictions.hidden.add("Acinar")
        window._save_heatmap_now()
        window.close_slide()

        window.open_slide(slide)
        restored = window.heatmap_panel.predictions
        assert restored.min_confidence == pytest.approx(0.6)
        assert restored.hidden == {"Acinar"}
        assert window.heatmap_panel.threshold.value() == 60

    def test_the_canvas_gets_it_too(self, window, slide):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window._save_heatmap_now()
        window.close_slide()
        window.open_slide(slide)
        assert window.canvas.predictions is not None

    def test_a_slide_with_no_sidecar_opens_clean(self, window, slide):
        window.open_slide(slide)
        assert window.heatmap_panel.predictions is None

    def test_a_hand_saved_sidecar_is_picked_up(self, window, slide):
        """Someone who used Save… before this existed should not lose it."""
        a_set(count=7).save(sidecar_path(slide))
        window.open_slide(slide)
        assert len(window.heatmap_panel.predictions) == 7

    def test_a_corrupt_sidecar_does_not_stop_the_slide_opening(self, window,
                                                               slide):
        auto_sidecar_path(slide).write_bytes(b"\x1f\x8b not really gzip")
        window.open_slide(slide)
        assert window.slide is not None
        assert window.heatmap_panel.predictions is None

    def test_a_truncated_gzip_is_survivable(self, window, slide):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window._save_heatmap_now()
        target = auto_sidecar_path(slide)
        target.write_bytes(target.read_bytes()[:40])
        window.close_slide()
        window.open_slide(slide)
        assert window.slide is not None
        assert window.heatmap_panel.predictions is None

    def test_opening_a_second_slide_does_not_keep_the_first_heatmap(
            self, window, slide, tmp_path):
        import shutil

        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window._save_heatmap_now()

        other = tmp_path / "case-02.tiff"
        shutil.copy(slide, other)
        window.open_slide(other)
        assert window.heatmap_panel.predictions is None


class TestThePreference:
    def test_it_is_on_by_default(self, window):
        assert window.heatmap_panel.autosave

    def test_toggling_it_is_remembered(self, window):
        window.heatmap_panel.autosave_checkbox.setChecked(False)
        assert window.settings.value("saveHeatmapBesideSlide") is False

    def test_setting_it_quietly_does_not_write_the_setting(self, window):
        """Restoring a stored value must not look like a user action."""
        window.heatmap_panel.set_autosave(False)
        assert window.settings.value("saveHeatmapBesideSlide", "unset") == \
            "unset"

    def test_turning_it_back_on_saves_at_once(self, window, slide):
        window.open_slide(slide)
        window.heatmap_panel.set_predictions(a_set())
        window.heatmap_panel.set_autosave(False)
        window.heatmap_panel.autosave_checkbox.setChecked(True)
        assert auto_sidecar_path(slide).exists()
