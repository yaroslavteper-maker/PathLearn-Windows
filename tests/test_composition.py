"""Breaking predicted tissue down into a percentage per class."""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.models.annotation import AnnotationColor
from pathlearn.models.prediction import PatchPrediction, PredictionSet
from pathlearn.pipeline.composition import (CompositionError, composition,
                                            format_area, _to_mm2)

SIZE = 224


def tile(col, row, label="PanIN-2", size=SIZE, confidence=0.9):
    # confidence is max(probabilities), so the winning class must carry the
    # value asked for exactly — [c, 1-c] would report 1-c whenever c < 0.5.
    return PatchPrediction(x=col * size, y=row * size, size_level0=size,
                           label=label,
                           probabilities=np.array([confidence, 0.0],
                                                  dtype=np.float32),
                           patch_size_level=size)


def a_set(tiles, labels=("PanIN-2", "Acinar"), min_confidence=0.0):
    return PredictionSet(predictions=list(tiles), class_labels=list(labels),
                         colors={"PanIN-2": AnnotationColor(200, 60, 60)},
                         slide_path="E:/case-01.svs",
                         min_confidence=min_confidence)


class TestShares:
    def test_an_even_split_is_fifty_fifty(self):
        tiles = [tile(0, 0), tile(1, 0),
                 tile(2, 0, label="Acinar"), tile(3, 0, label="Acinar")]
        report = composition(a_set(tiles))
        assert report.share_for("PanIN-2").area_percent == pytest.approx(50.0)
        assert report.share_for("Acinar").area_percent == pytest.approx(50.0)

    def test_the_percentages_sum_to_a_hundred(self):
        tiles = ([tile(c, 0) for c in range(7)]
                 + [tile(c, 1, label="Acinar") for c in range(3)]
                 + [tile(c, 2, label="Normal duct") for c in range(2)])
        report = composition(a_set(tiles))
        assert sum(s.area_percent for s in report.shares) == pytest.approx(100.0)
        assert sum(s.tile_percent for s in report.shares) == pytest.approx(100.0)

    def test_tile_share_and_area_share_agree_on_a_uniform_grid(self):
        """No overlap, one patch size: the two measures must not diverge."""
        tiles = [tile(c, 0) for c in range(5)] + [tile(c, 1, label="Acinar")
                                                 for c in range(3)]
        for share in composition(a_set(tiles)).shares:
            assert share.area_percent == pytest.approx(share.tile_percent)

    def test_the_largest_class_comes_first(self):
        tiles = [tile(0, 0, label="Acinar")] + [tile(c, 1) for c in range(4)]
        report = composition(a_set(tiles))
        assert [s.label for s in report.shares] == ["PanIN-2", "Acinar"]

    def test_mean_confidence_is_per_class(self):
        tiles = [tile(0, 0, confidence=0.6), tile(1, 0, confidence=1.0),
                 tile(2, 0, label="Acinar", confidence=0.8)]
        report = composition(a_set(tiles))
        assert report.share_for("PanIN-2").mean_confidence == pytest.approx(0.8)
        assert report.share_for("Acinar").mean_confidence == pytest.approx(0.8)

    def test_one_class_is_a_hundred_percent(self):
        report = composition(a_set([tile(0, 0), tile(1, 0)]))
        assert len(report.shares) == 1
        assert report.shares[0].area_percent == pytest.approx(100.0)


class TestArea:
    def test_area_is_cells_times_cell_area(self):
        report = composition(a_set([tile(0, 0), tile(1, 0)]))
        assert report.total_area_px == pytest.approx(2 * SIZE * SIZE)

    def test_millimetres_need_a_pixel_size(self):
        report = composition(a_set([tile(0, 0)]))
        assert report.total_area_mm2 is None
        assert report.shares[0].area_mm2 is None
        assert "px²" in report.area_text()

    def test_millimetres_when_the_slide_reports_one(self):
        report = composition(a_set([tile(0, 0)]), mpp=0.25)
        # 224 px * 0.25 um = 56 um a side -> 3136 um^2 -> 0.003136 mm^2.
        assert report.total_area_mm2 == pytest.approx(0.003136)

    def test_the_conversion_is_area_not_length(self):
        """Doubling the pixel size quadruples the area, not doubles it."""
        assert _to_mm2(100.0, 0.5) == pytest.approx(4 * _to_mm2(100.0, 0.25))

    def test_class_areas_sum_to_the_total(self):
        tiles = [tile(c, 0) for c in range(3)] + [tile(c, 1, label="Acinar")
                                                 for c in range(5)]
        report = composition(a_set(tiles))
        assert sum(s.area_px for s in report.shares) == pytest.approx(
            report.total_area_px)


class TestFormattingAnArea:
    """A small region must not render as "0.00 mm2" — that reads as none."""

    @pytest.mark.parametrize("mm2,expected", [
        (12.3456, "12.35 mm²"),
        (1.0, "1.00 mm²"),
        (0.0376, "0.038 mm²"),
        (0.01, "0.010 mm²"),
        (0.003136, "3,136 µm²"),
        (0.0000001, "0 µm²"),
    ])
    def test_the_unit_follows_the_magnitude(self, mm2, expected):
        assert format_area(0.0, mm2) == expected

    def test_no_pixel_size_means_pixels(self):
        assert format_area(50176.0, None) == "50,176 px²"

    def test_a_small_area_never_reads_as_zero(self):
        for mm2 in (0.009, 0.0005, 0.00002):
            assert not format_area(0.0, mm2).startswith("0.00 ")


class TestOverlap:
    """Overlapping tiles must not be double-counted."""

    def half_stride(self):
        def at(x, label, confidence):
            return PatchPrediction(
                x=x, y=0, size_level0=SIZE, label=label,
                probabilities=np.array([confidence, 0.0], dtype=np.float32))
        return [at(0, "PanIN-2", 0.9), at(SIZE // 2, "Acinar", 0.6)]

    def test_the_more_confident_tile_owns_the_shared_cells(self):
        report = composition(a_set(self.half_stride()), cell_size=SIZE // 2)
        assert report.share_for("PanIN-2").cells == 4
        assert report.share_for("Acinar").cells == 2

    def test_overlap_does_not_inflate_the_total(self):
        report = composition(a_set(self.half_stride()), cell_size=SIZE // 2)
        assert report.total_cells == 6
        assert sum(s.cells for s in report.shares) == report.total_cells
        assert sum(s.area_percent for s in report.shares) == pytest.approx(100.0)

    def test_area_share_diverges_from_tile_share_under_overlap(self):
        """Half the tiles, two thirds of the tissue — that is the whole point."""
        report = composition(a_set(self.half_stride()), cell_size=SIZE // 2)
        winner = report.share_for("PanIN-2")
        assert winner.tile_percent == pytest.approx(50.0)
        assert winner.area_percent == pytest.approx(200 / 3)

    def test_a_big_tile_outweighs_a_small_one_by_area(self):
        big = PatchPrediction(x=0, y=0, size_level0=SIZE * 2, label="PanIN-2",
                              probabilities=np.array([0.9, 0.0], dtype=np.float32))
        small = PatchPrediction(x=SIZE * 2, y=0, size_level0=SIZE, label="Acinar",
                                probabilities=np.array([0.9, 0.0], dtype=np.float32))
        report = composition(a_set([big, small]), cell_size=SIZE)
        assert report.share_for("PanIN-2").tile_percent == pytest.approx(50.0)
        # 2x2 cells against 1 -> four fifths of the tissue.
        assert report.share_for("PanIN-2").area_percent == pytest.approx(80.0)


class TestTheDenominator:
    def test_low_confidence_tiles_are_excluded_and_counted(self):
        tiles = [tile(0, 0), tile(1, 0, label="Acinar", confidence=0.2)]
        report = composition(a_set(tiles), min_confidence=0.5)
        assert report.excluded_low_confidence == 1
        assert report.total_tiles == 1
        assert report.share_for("Acinar") is None
        assert report.share_for("PanIN-2").area_percent == pytest.approx(100.0)

    def test_the_sets_own_threshold_is_the_default(self):
        """The table must match the heatmap the user is looking at."""
        tiles = [tile(0, 0), tile(1, 0, confidence=0.3)]
        report = composition(a_set(tiles, min_confidence=0.5))
        assert report.total_tiles == 1
        assert report.min_confidence == pytest.approx(0.5)

    def test_hidden_classes_are_still_counted(self):
        """Hiding is a drawing choice; it must not move a quantitative result."""
        tiles = [tile(0, 0), tile(1, 0, label="Acinar")]
        predictions = a_set(tiles)
        visible = composition(predictions)
        predictions.hidden.add("Acinar")
        assert composition(predictions).shares == visible.shares

    def test_an_empty_set(self):
        report = composition(a_set([]))
        assert report.is_empty
        assert "No predictions" in report.summary()

    def test_everything_below_the_threshold_says_so(self):
        report = composition(a_set([tile(0, 0, confidence=0.1)]),
                             min_confidence=0.9)
        assert report.is_empty
        assert report.excluded_low_confidence == 1
        assert "below it" in report.summary()

    def test_a_nonsense_grid_is_refused(self):
        far = PatchPrediction(x=0, y=0, size_level0=10 ** 9, label="PanIN-2",
                              probabilities=np.array([0.9, 0.0], dtype=np.float32))
        with pytest.raises(CompositionError):
            composition(a_set([far]), cell_size=1)


class TestReporting:
    def report(self):
        tiles = [tile(c, 0) for c in range(3)] + [tile(c, 1, label="Acinar")
                                                 for c in range(1)]
        return composition(a_set(tiles), mpp=0.25)

    def test_the_summary_names_the_classes_and_the_area(self):
        text = self.report().summary()
        assert "PanIN-2 75.0%" in text
        assert "mm²" in text

    def test_the_summary_mentions_excluded_tiles(self):
        tiles = [tile(0, 0), tile(1, 0, confidence=0.1)]
        assert "excluded" in composition(a_set(tiles),
                                         min_confidence=0.5).summary()

    def test_the_csv_has_a_row_per_class_and_a_total(self):
        lines = self.report().to_csv().splitlines()
        assert lines[0].startswith("Class,Tiles")
        assert lines[1].startswith("PanIN-2,3,")
        assert lines[2].startswith("Acinar,1,")
        assert lines[3].startswith("Total,4,")

    def test_the_csv_records_what_the_denominator_was(self):
        """A bare percentage column outlives the memory of what it was of."""
        text = self.report().to_csv()
        assert "# Denominator,Predicted tissue only" in text
        assert "# Confidence threshold,0.00" in text
        assert "# Microns per pixel,0.2500" in text

    def test_an_unknown_pixel_size_says_unknown_rather_than_guessing(self):
        assert "# Microns per pixel,unknown" in composition(
            a_set([tile(0, 0)])).to_csv()

    def test_a_label_with_a_comma_is_quoted(self):
        tiles = [tile(0, 0, label="PanIN-1a, low grade")]
        assert '"PanIN-1a, low grade"' in composition(a_set(tiles)).to_csv()
