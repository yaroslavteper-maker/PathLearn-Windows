"""Stitching predicted tiles into annotations, one per connected region."""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.models.annotation import AnnotationColor
from pathlearn.models.prediction import PatchPrediction, PredictionSet
from pathlearn.models.store import AnnotationStore
from pathlearn.pipeline.stitch import (StitchSettings, infer_cell_size, rasterise,
                                       stitch_class, trace_outline)

SIZE = 224
CELL_AREA = SIZE * SIZE


def tile(col, row, label="PanIN-2", size=SIZE, confidence=0.9, origin=(0, 0)):
    # PatchPrediction.confidence is max(probabilities), so the winning class
    # must carry exactly the value asked for — [c, 1-c] would report 1-c
    # whenever c < 0.5, which is precisely the case worth testing.
    probabilities = np.array([confidence, 0.0], dtype=np.float32)
    return PatchPrediction(x=origin[0] + col * size, y=origin[1] + row * size,
                           size_level0=size, label=label,
                           probabilities=probabilities, patch_size_level=size)


def prediction_set(tiles, labels=("PanIN-2", "Acinar")):
    return PredictionSet(predictions=list(tiles), class_labels=list(labels),
                         colors={"PanIN-2": AnnotationColor(200, 60, 60)})


def cells(*pairs, **kw):
    """(row, col) pairs -> tiles."""
    return [tile(col, row, **kw) for row, col in pairs]


class TestAreaIsExact:
    """Marching squares chamfers corners; tiles are squares. Pin the difference."""

    @pytest.mark.parametrize("name,pairs,expected", [
        ("single", [(0, 0)], 1),
        ("2x2", [(0, 0), (0, 1), (1, 0), (1, 1)], 4),
        ("1x4 bar", [(0, 0), (0, 1), (0, 2), (0, 3)], 4),
        ("L", [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)], 5),
        ("plus", [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], 5),
        ("ring", [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)], 9),
    ])
    def test_traced_area_matches_the_tiles(self, name, pairs, expected):
        report = stitch_class(prediction_set(cells(*pairs)), "PanIN-2",
                              StitchSettings(min_cells=1))
        area = sum(a.area_in_slide_pixels for a in report.annotations)
        assert area == pytest.approx(expected * CELL_AREA), name

    def test_an_l_is_not_its_bounding_box(self):
        report = stitch_class(prediction_set(cells((0, 0), (1, 0), (2, 0),
                                                   (2, 1), (2, 2))),
                              "PanIN-2", StitchSettings(min_cells=1))
        annotation = report.annotations[0]
        _, _, width, height = annotation.bounding_box
        assert annotation.area_in_slide_pixels < width * height

    def test_the_outline_starts_at_the_tile_origin(self):
        report = stitch_class(prediction_set(cells((0, 0), (0, 1), (1, 0), (1, 1))),
                              "PanIN-2", StitchSettings(min_cells=1))
        x, y, width, height = report.annotations[0].bounding_box
        assert (x, y) == (0, 0)
        assert (width, height) == (2 * SIZE, 2 * SIZE)


class TestSeparateRegions:
    def test_two_clumps_become_two_annotations(self):
        tiles = cells((0, 0), (0, 1), (1, 0), (1, 1)) + cells((0, 10), (0, 11))
        report = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=1))
        assert len(report.annotations) == 2
        assert {a.classification for a in report.annotations} == {"PanIN-2"}

    def test_they_do_not_overlap(self):
        tiles = cells((0, 0), (0, 1)) + cells((0, 10), (0, 11))
        report = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=1))
        first, second = report.annotations
        assert first.bounding_box[0] + first.bounding_box[2] < second.bounding_box[0]

    def test_each_gets_its_own_name(self):
        tiles = cells((0, 0)) + cells((0, 10))
        report = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=1))
        assert len({a.name for a in report.annotations}) == 2

    def test_diagonal_touch_is_two_regions_by_default(self):
        report = stitch_class(prediction_set(cells((0, 0), (1, 1))), "PanIN-2",
                              StitchSettings(min_cells=1))
        assert len(report.annotations) == 2

    def test_diagonal_touch_joins_at_connectivity_eight(self):
        report = stitch_class(prediction_set(cells((0, 0), (1, 1))), "PanIN-2",
                              StitchSettings(min_cells=1, connectivity=8))
        assert len(report.annotations) == 1


class TestFiltering:
    def test_only_the_chosen_class_is_used(self):
        tiles = cells((0, 0), (0, 1)) + cells((5, 5), label="Acinar")
        report = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=1))
        assert report.tiles_used == 2
        assert all(a.classification == "PanIN-2" for a in report.annotations)

    def test_low_confidence_tiles_are_dropped(self):
        tiles = cells((0, 0)) + cells((0, 1), confidence=0.3)
        report = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=1, min_confidence=0.5))
        assert report.tiles_used == 1
        assert len(report.annotations) == 1

    def test_confidence_can_split_a_region(self):
        """A weak tile in the middle breaks one region into two."""
        tiles = (cells((0, 0)) + cells((0, 1), confidence=0.2) + cells((0, 2)))
        joined = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=1))
        split = stitch_class(prediction_set(tiles), "PanIN-2",
                             StitchSettings(min_cells=1, min_confidence=0.5))
        assert len(joined.annotations) == 1
        assert len(split.annotations) == 2

    def test_small_regions_are_dropped_and_counted(self):
        tiles = cells((0, 0), (0, 1), (1, 0), (1, 1)) + cells((0, 10))
        report = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=2))
        assert len(report.annotations) == 1
        assert report.too_small == 1
        assert "too small" in report.summary()

    def test_a_class_with_no_tiles(self):
        report = stitch_class(prediction_set(cells((0, 0))), "Nothing")
        assert report.annotations == []
        assert "No tiles" in report.summary()


class TestGrid:
    def test_the_cell_size_comes_from_the_tile_pitch(self):
        assert infer_cell_size(cells((0, 0), (0, 1), (0, 2))) == SIZE

    def test_overlapping_tiles_use_the_stride(self):
        """Stride below patch size: the pitch is the grid, not the tile."""
        half = [tile(0, 0), PatchPrediction(x=SIZE // 2, y=0, size_level0=SIZE,
                                            label="PanIN-2",
                                            probabilities=np.array([0.9, 0.0],
                                                                   dtype=np.float32))]
        assert infer_cell_size(half) == SIZE // 2

    def test_a_single_tile_falls_back_to_its_size(self):
        assert infer_cell_size(cells((0, 0))) == SIZE

    def test_rasterise_marks_the_covered_cells(self):
        mask, ox, oy = rasterise(cells((0, 0), (1, 1)), SIZE)
        assert mask.shape == (2, 2)
        assert mask[0, 0] and mask[1, 1]
        assert not mask[0, 1] and not mask[1, 0]
        assert (ox, oy) == (0, 0)

    def test_the_origin_follows_the_tiles(self):
        _, ox, oy = rasterise(cells((3, 4)), SIZE)
        assert (ox, oy) == (4 * SIZE, 3 * SIZE)

    def test_overlapping_tiles_still_cover_one_region(self):
        overlap = [tile(0, 0),
                   PatchPrediction(x=SIZE // 2, y=0, size_level0=SIZE,
                                   label="PanIN-2",
                                   probabilities=np.array([0.9, 0.0],
                                                          dtype=np.float32))]
        report = stitch_class(prediction_set(overlap), "PanIN-2",
                              StitchSettings(min_cells=1))
        assert len(report.annotations) == 1


class TestSimplification:
    def test_it_is_off_by_default(self):
        exact = stitch_class(prediction_set(cells((0, 0), (1, 0), (2, 0),
                                                  (2, 1), (2, 2))),
                             "PanIN-2", StitchSettings(min_cells=1))
        assert exact.annotations[0].area_in_slide_pixels == pytest.approx(5 * CELL_AREA)

    def test_smoothing_moves_the_boundary(self):
        """Documented trade: it removes steps by moving the outline."""
        pairs = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        exact = stitch_class(prediction_set(cells(*pairs)), "PanIN-2",
                             StitchSettings(min_cells=1, simplify_tolerance=0.0))
        smoothed = stitch_class(prediction_set(cells(*pairs)), "PanIN-2",
                                StitchSettings(min_cells=1, simplify_tolerance=0.9))
        assert (smoothed.annotations[0].area_in_slide_pixels
                < exact.annotations[0].area_in_slide_pixels)

    def test_smoothing_never_destroys_a_region(self):
        report = stitch_class(prediction_set(cells((0, 0))), "PanIN-2",
                              StitchSettings(min_cells=1, simplify_tolerance=3.0))
        assert len(report.annotations) == 1


class TestOutlines:
    def test_a_rectangle_traces_to_four_corners(self):
        outline = trace_outline(np.ones((2, 3), dtype=bool), SIZE, 0, 0, 0.0)
        assert len(outline) == 4

    def test_the_ring_is_stored_open(self):
        outline = trace_outline(np.ones((2, 2), dtype=bool), SIZE, 0, 0, 0.0)
        assert outline[0] != outline[-1]

    def test_an_empty_component_has_no_outline(self):
        assert trace_outline(np.zeros((3, 3), dtype=bool), SIZE, 0, 0, 0.0) is None

    def test_the_outer_boundary_wins_over_a_hole(self):
        """A ring traces its outside, not the hole in the middle."""
        ring = np.ones((3, 3), dtype=bool)
        ring[1, 1] = False
        outline = trace_outline(ring, SIZE, 0, 0, 0.0)
        xs = [p.x for p in outline]
        ys = [p.y for p in outline]
        assert min(xs) == 0 and max(xs) == 3 * SIZE
        assert min(ys) == 0 and max(ys) == 3 * SIZE


class TestAnnotationsProduced:
    def test_they_carry_the_class_and_its_colour(self):
        report = stitch_class(prediction_set(cells((0, 0), (0, 1))), "PanIN-2",
                              StitchSettings(min_cells=1))
        annotation = report.annotations[0]
        assert annotation.classification == "PanIN-2"
        assert annotation.color == AnnotationColor(200, 60, 60)

    def test_they_are_usable_immediately(self):
        report = stitch_class(prediction_set(cells((0, 0), (0, 1))), "PanIN-2",
                              StitchSettings(min_cells=1))
        assert report.annotations[0].is_selected
        assert not report.annotations[0].is_subtractive

    def test_the_name_says_where_they_came_from(self):
        report = stitch_class(prediction_set(cells((0, 0))), "PanIN-2",
                              StitchSettings(min_cells=1))
        assert "stitched" in report.annotations[0].name

    def test_they_can_be_described_by_the_geometry_pipeline(self, tmp_path):
        """The whole point: stitched regions feed the shape descriptor."""
        from pathlearn.core.shape import describe_shape

        report = stitch_class(prediction_set(cells((0, 0), (0, 1), (1, 0), (1, 1))),
                              "PanIN-2", StitchSettings(min_cells=1))
        result = describe_shape(report.annotations[0], mpp=0.25)
        assert result is not None
        assert np.all(np.isfinite(result.features))


class TestSheet:
    """The sheet stitches on a worker thread, so every check waits for it.

    It used to run inline on the UI thread and froze the app on a real
    whole-slide prediction.
    """

    def settle(self, qtbot, sheet, timeout_ms=20_000):
        """Run any pending debounce, then wait for the worker to finish."""
        if sheet._debounce.isActive():
            sheet._debounce.stop()
            sheet._start_preview()
        waited = 0
        while sheet.task is not None and waited < timeout_ms:
            qtbot.wait(20)
            waited += 20
        assert sheet.task is None, "the stitch worker never finished"

    @pytest.fixture
    def sheet(self, qtbot, tmp_path):
        from pathlearn.ui.sheets.stitch_regions import StitchRegionsSheet

        store = AnnotationStore()
        store.bind(tmp_path / "s.svs", 4096)
        tiles = (cells((0, 0), (0, 1), (1, 0), (1, 1))
                 + cells((0, 10), (0, 11))
                 + cells((5, 5), label="Acinar"))
        widget = StitchRegionsSheet(prediction_set(tiles), store)
        qtbot.addWidget(widget)
        yield widget
        widget.reject()

    def test_classes_are_offered_by_tile_count(self, sheet):
        assert sheet.class_combo.count() == 2
        assert sheet.class_combo.itemData(0) == "PanIN-2"

    def test_the_preview_counts_the_regions(self, qtbot, sheet):
        self.settle(qtbot, sheet)
        assert "2 region(s)" in sheet.preview.text()
        assert "2 Annotation(s)" in sheet.create_button.text()

    def test_raising_min_cells_drops_the_small_one(self, qtbot, sheet):
        self.settle(qtbot, sheet)
        sheet.min_cells.setValue(3)
        self.settle(qtbot, sheet)
        assert "1 region(s)" in sheet.preview.text()

    def test_connectivity_is_offered(self, sheet):
        assert sheet.connectivity.itemData(0) == 4
        assert sheet.connectivity.itemData(1) == 8

    def test_creating_adds_them_to_the_store(self, qtbot, sheet):
        self.settle(qtbot, sheet)
        before = len(sheet.store.annotations)
        sheet._create()
        assert len(sheet.store.annotations) == before + 2
        assert sheet.added == 2

    def test_the_staircase_caution_is_shown(self, sheet):
        assert "staircase" in sheet.caution.text()

    def test_smoothing_defaults_to_off(self, sheet):
        assert sheet.settings().simplify_tolerance == 0.0

    def test_editing_a_setting_invalidates_the_preview(self, qtbot, sheet):
        """Create must never add regions that differ from what was shown."""
        self.settle(qtbot, sheet)
        assert sheet.report is not None
        sheet.min_cells.setValue(3)
        assert sheet.report is None
        assert not sheet.create_button.isEnabled()

    def test_creating_with_a_stale_preview_recomputes_instead(self, qtbot, sheet):
        self.settle(qtbot, sheet)
        sheet.min_cells.setValue(3)
        sheet._debounce.stop()          # pretend the debounce has not fired yet
        sheet._create()                 # must re-run, not add the old regions
        assert sheet.added == 0
        self.settle(qtbot, sheet)
        assert "1 region(s)" in sheet.preview.text()

    def test_controls_are_locked_while_it_runs(self, sheet):
        sheet._set_running(True)
        assert sheet.stop_button.isEnabled()
        assert not sheet.preview_button.isEnabled()
        assert not sheet.class_combo.isEnabled()
        sheet._set_running(False)
        assert not sheet.stop_button.isEnabled()
        assert sheet.preview_button.isEnabled()

    def test_stopping_when_idle_is_harmless(self, qtbot, sheet):
        self.settle(qtbot, sheet)
        sheet._stop()
        assert sheet.task is None


class TestTheHangIsFixed:
    """Regression tests for a freeze on real, whole-slide predictions."""

    def test_a_stray_origin_cannot_collapse_the_cell_size(self):
        """The cause of a 2.4 GB raster: one odd pair of origins.

        Multi-pass prediction at different levels produces origins that are not
        all on one pitch. Taking the minimum gap turned a single 1 px pair into
        a cell size of 1.
        """
        odd = [tile(0, 0), tile(1, 0),
               PatchPrediction(x=SIZE + 1, y=0, size_level0=SIZE, label="PanIN-2",
                               probabilities=np.array([0.9, 0.0], dtype=np.float32)),
               tile(150, 250)]
        assert infer_cell_size(odd) == SIZE

    def test_an_impossible_grid_is_refused_not_allocated(self):
        from pathlearn.pipeline.stitch import StitchError

        far = [tile(0, 0), tile(400, 400)]
        with pytest.raises(StitchError, match="too large"):
            rasterise(far, 1)

    def test_the_refusal_says_what_to_do(self):
        from pathlearn.pipeline.stitch import StitchError

        try:
            rasterise([tile(0, 0), tile(400, 400)], 1)
        except StitchError as exc:
            assert "cell size" in str(exc)

    def test_many_small_regions_are_not_quadratic(self):
        """Each region is traced within its own box, not over the whole grid.

        Tracing every component across the full raster made this O(regions x
        slide) — 6,000 regions took nine seconds on the UI thread.
        """
        import time

        scattered = [tile(c, r) for r in range(0, 120, 3) for c in range(0, 90, 3)]
        start = time.perf_counter()
        report = stitch_class(prediction_set(scattered), "PanIN-2",
                              StitchSettings(min_cells=1))
        elapsed = time.perf_counter() - start
        assert len(report.annotations) == len(scattered)
        assert elapsed < 3.0, f"took {elapsed:.1f}s for {len(scattered)} regions"

    def test_a_whole_slide_worth_of_tiles_is_quick(self):
        import time

        tiles = [tile(c, r) for r in range(150) for c in range(120)]
        start = time.perf_counter()
        report = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=1))
        elapsed = time.perf_counter() - start
        assert len(report.annotations) == 1
        assert elapsed < 3.0, f"took {elapsed:.1f}s for {len(tiles)} tiles"

    def test_it_can_be_cancelled(self):
        tiles = [tile(c, r) for r in range(0, 60, 3) for c in range(0, 60, 3)]
        report = stitch_class(prediction_set(tiles), "PanIN-2",
                              StitchSettings(min_cells=1),
                              should_cancel=lambda: True)
        assert report.cancelled
        assert "Stopped early" in report.summary()
        assert "No tiles" not in report.summary(), (
            "cancelled must not read as nothing matched")
