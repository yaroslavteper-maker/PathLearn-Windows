"""Class breakdown of the annotations you drew."""

from __future__ import annotations

import pytest

from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.pipeline.annotation_stats import (MAX_OVERLAP_CHECK, breakdown,
                                                 _centroid)


def square(x, y, side, label="PanIN-2", subtractive=False, colour=None):
    return Annotation(
        points=[Point(x, y), Point(x + side, y),
                Point(x + side, y + side), Point(x, y + side)],
        classification=label,
        color=colour or AnnotationColor(200, 60, 60),
        is_subtractive=subtractive)


class TestCounts:
    def test_an_even_split(self):
        report = breakdown([square(0, 0, 10), square(100, 0, 10),
                            square(200, 0, 10, label="Acinar"),
                            square(300, 0, 10, label="Acinar")])
        assert report.tally_for("PanIN-2").count_percent == pytest.approx(50.0)
        assert report.tally_for("Acinar").count_percent == pytest.approx(50.0)

    def test_the_counts_sum_to_a_hundred(self):
        report = breakdown([square(i * 100, 0, 10, label=f"C{i % 3}")
                            for i in range(10)])
        assert sum(t.count_percent for t in report.tallies) == \
            pytest.approx(100.0)
        assert report.total_count == 10

    def test_an_unlabelled_region_gets_a_name(self):
        """It still has to appear somewhere, not vanish from the total."""
        report = breakdown([square(0, 0, 10, label="")])
        assert report.tally_for("Unlabeled").count == 1

    def test_no_annotations(self):
        report = breakdown([])
        assert report.is_empty
        assert "No annotations" in report.summary()


class TestCountVersusArea:
    """The two answers differ, and both are always reported."""

    def test_many_small_against_one_large(self):
        # Nine 10x10 ducts (900 px²) and one 100x100 lesion (10,000 px²).
        annotations = [square(i * 20, 0, 10, label="PanIN-1a")
                       for i in range(9)]
        annotations.append(square(0, 500, 100, label="PanIN-3"))
        report = breakdown(annotations)

        small = report.tally_for("PanIN-1a")
        assert small.count_percent == pytest.approx(90.0)
        assert small.area_percent == pytest.approx(900 / 10900 * 100)
        assert small.area_percent < 10.0

    def test_the_areas_sum_to_a_hundred(self):
        report = breakdown([square(0, 0, 10), square(100, 0, 20,
                                                     label="Acinar")])
        assert sum(t.area_percent for t in report.tallies) == \
            pytest.approx(100.0)

    def test_mean_size_is_per_class(self):
        report = breakdown([square(0, 0, 10), square(100, 0, 30)])
        # (100 + 900) / 2
        assert report.tally_for("PanIN-2").mean_area_px == pytest.approx(500.0)

    def test_the_largest_by_area_comes_first(self):
        report = breakdown([square(0, 0, 10, label="Small"),
                            square(100, 0, 10, label="Small"),
                            square(200, 0, 100, label="Big")])
        assert [t.label for t in report.tallies] == ["Big", "Small"]


class TestArea:
    def test_area_is_the_polygon_area(self):
        assert breakdown([square(0, 0, 10)]).total_area_px == \
            pytest.approx(100.0)

    def test_millimetres_need_a_pixel_size(self):
        report = breakdown([square(0, 0, 10)])
        assert report.total_area_mm2 is None
        assert report.tallies[0].area_mm2 is None

    def test_millimetres_when_given_one(self):
        report = breakdown([square(0, 0, 1000)], mpp=0.5)
        # 1000 px * 0.5 um = 500 um a side -> 250,000 um^2 -> 0.25 mm^2.
        assert report.total_area_mm2 == pytest.approx(0.25)
        assert report.tallies[0].mean_area_mm2 == pytest.approx(0.25)


class TestSubtractivePolygons:
    """Holes are not regions: never counted, always deducted."""

    def test_a_hole_is_not_counted_as_a_region(self):
        report = breakdown([square(0, 0, 100),
                            square(10, 10, 20, label="Lumen",
                                   subtractive=True)])
        assert report.total_count == 1
        assert report.tally_for("Lumen") is None
        assert report.subtractive_count == 1

    def test_a_hole_is_deducted_from_the_enclosing_class(self):
        report = breakdown([square(0, 0, 100),
                            square(10, 10, 20, subtractive=True)])
        assert report.total_area_px == pytest.approx(10_000 - 400)
        assert report.tally_for("PanIN-2").carved_px == pytest.approx(400.0)

    def test_it_deducts_from_the_innermost_region(self):
        """A hole in a lesion in a block carves the lesion, not the block."""
        report = breakdown([
            square(0, 0, 1000, label="Block"),
            square(100, 100, 200, label="Lesion"),
            square(150, 150, 50, label="Lumen", subtractive=True)])
        assert report.tally_for("Lesion").carved_px == pytest.approx(2500.0)
        assert report.tally_for("Block").carved_px == 0.0

    def test_a_hole_inside_nothing_deducts_from_nothing(self):
        """Guessing which class it meant would be inventing data."""
        report = breakdown([square(0, 0, 100),
                            square(5000, 5000, 20, subtractive=True)])
        assert report.orphan_subtractive == 1
        assert report.carved_px == 0.0
        assert report.total_area_px == pytest.approx(10_000)

    def test_a_hole_bigger_than_its_parent_clamps_at_zero(self):
        """A tracing mistake must not produce a negative area."""
        # Centred on the parent, so it really does enclose it rather than
        # merely being large and elsewhere.
        report = breakdown([square(0, 0, 10),
                            square(5 - 250, 5 - 250, 500, subtractive=True)])
        assert report.orphan_subtractive == 0
        assert report.tally_for("PanIN-2").area_px == 0.0
        assert report.tally_for("PanIN-2").area_percent == 0.0

    def test_only_holes_is_not_a_breakdown(self):
        report = breakdown([square(0, 0, 10, subtractive=True)])
        assert report.is_empty
        assert report.orphan_subtractive == 1

    def test_the_summary_mentions_them(self):
        report = breakdown([square(0, 0, 100),
                            square(10, 10, 20, subtractive=True)])
        assert "subtractive" in report.summary()


class TestOverlapWarning:
    def test_separate_regions_do_not_warn(self):
        assert breakdown([square(0, 0, 10),
                          square(500, 500, 10)]).overlapping_pairs == 0

    def test_touching_edges_do_not_count_as_overlap(self):
        """Two regions sharing a border are adjacent, not overlapping."""
        assert breakdown([square(0, 0, 10),
                          square(10, 0, 10)]).overlapping_pairs == 0

    def test_genuinely_overlapping_regions_warn(self):
        assert breakdown([square(0, 0, 100),
                          square(50, 50, 100)]).overlapping_pairs == 1

    def test_the_check_is_skipped_when_there_are_too_many(self):
        """O(n^2) on a stitched slide is not worth a footnote."""
        many = [square(0, 0, 100) for _ in range(MAX_OVERLAP_CHECK + 1)]
        report = breakdown(many)
        assert report.overlapping_pairs == 0
        assert report.total_count == MAX_OVERLAP_CHECK + 1


class TestCentroid:
    def test_a_square(self):
        assert _centroid(square(0, 0, 10)) == pytest.approx((5.0, 5.0))

    def test_a_degenerate_polygon_does_not_divide_by_zero(self):
        line = Annotation(points=[Point(0, 0), Point(10, 0)],
                          classification="x", color=AnnotationColor.default())
        assert _centroid(line) == pytest.approx((5.0, 0.0))

    def test_an_empty_polygon(self):
        empty = Annotation(points=[], classification="x",
                           color=AnnotationColor.default())
        assert _centroid(empty) == (0.0, 0.0)


class TestReporting:
    def a_report(self):
        return breakdown([square(0, 0, 10), square(100, 0, 10),
                          square(200, 0, 30, label="Acinar"),
                          square(20, 2, 4, subtractive=True)], mpp=0.5,
                         scope="all annotations")

    def test_the_csv_has_a_row_per_class_and_a_total(self):
        lines = self.a_report().to_csv().splitlines()
        assert lines[0].startswith("Class,Annotations,Count %")
        assert lines[3].startswith("Total,3,100.00,")

    def test_the_csv_records_the_scope_and_the_source(self):
        text = self.a_report().to_csv()
        assert "# Scope,all annotations" in text
        assert "not model predictions" in text
        assert "# Subtractive polygons excluded,1" in text

    def test_the_csv_explains_why_the_two_percentages_differ(self):
        assert "Count % versus Area %" in self.a_report().to_csv()

    def test_a_label_with_a_comma_is_quoted(self):
        report = breakdown([square(0, 0, 10, label="PanIN-1a, low")])
        assert '"PanIN-1a, low"' in report.to_csv()

    def test_the_scope_is_carried_through(self):
        report = breakdown([square(0, 0, 10)],
                           scope="annotations checked under Use")
        assert "# Scope,annotations checked under Use" in report.to_csv()
