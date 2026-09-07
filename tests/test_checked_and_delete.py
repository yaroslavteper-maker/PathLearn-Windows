"""Checked-annotation gating, source recording, and patch deletion."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from PySide6.QtCore import Qt

from pathlearn.data.bank import Patch, PatchBank
from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.store import AnnotationStore
from pathlearn.ui.sheets.patch_browser import PatchBrowser
from synthetic_slide import write_synthetic_slide

DIM = 8


def patch(slide="a.svs", classification="Acinar", x=0, y=0, **kw):
    defaults = dict(slide_path=f"E:/{slide}", slide_name=slide,
                    annotation_id=uuid.uuid4(), classification=classification,
                    patch_x=x, patch_y=y, patch_level=0, patch_size_level=224,
                    features=np.zeros(DIM, dtype=np.float32),
                    extractor_identity="onnx:test:r1", white_fraction=0.1,
                    nucleus_count=12)
    defaults.update(kw)
    return Patch(**defaults)


@pytest.fixture
def bank(tmp_path):
    with PatchBank(tmp_path / "b.db") as b:
        yield b


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("s") / "chk.tif"
    write_synthetic_slide(path, 4096, 3072, levels=4)
    with SlideImage(path) as s:
        yield s


@pytest.fixture
def store(tmp_path, slide):
    s = AnnotationStore()
    s.bind(tmp_path / "slide.svs", slide.dimensions.height)
    for name, x in (("A", 400), ("B", 1800)):
        s.add(Annotation(points=[Point(x, 400), Point(x + 800, 400),
                                 Point(x + 800, 1200), Point(x, 1200)],
                         classification=name, color=AnnotationColor.default()))
    return s


class TestCheckedGating:
    def test_all_checked_by_default(self, store):
        assert len(store.in_use()) == 2

    def test_unchecking_excludes_from_use(self, store):
        first = store.annotations[0]
        store.set_selected(first.id, False)
        in_use = store.in_use()
        assert len(in_use) == 1
        assert first.id not in {a.id for a in in_use}

    def test_subtractive_excluded_unless_asked(self, store):
        store.add(Annotation(points=[Point(0, 0), Point(100, 0), Point(100, 100)],
                             classification="hole",
                             color=AnnotationColor.default(), is_subtractive=True))
        assert len(store.in_use()) == 2
        assert len(store.in_use(include_subtractive=True)) == 3

    def test_subtractive_in_use_respects_the_checkbox(self, store):
        hole = Annotation(points=[Point(0, 0), Point(100, 0), Point(100, 100)],
                          classification="hole", color=AnnotationColor.default(),
                          is_subtractive=True)
        store.add(hole)
        assert len(store.subtractive_in_use) == 1
        store.set_selected(hole.id, False)
        assert store.subtractive_in_use == []

    def test_extract_sheet_only_offers_checked(self, qtbot, slide, store, bank, tmp_path):
        from pathlearn.extractors.registry import ExtractorRegistry
        from pathlearn.ui.sheets.extract_patches import ExtractPatchesSheet
        from test_extractors import write_extractor

        write_extractor(tmp_path, "stub", dim=DIM)
        registry = ExtractorRegistry([tmp_path])
        store.set_selected(store.annotations[0].id, False)

        sheet = ExtractPatchesSheet(slide, store, registry, bank)
        qtbot.addWidget(sheet)
        assert sheet.selected_classes == {"B"}
        sheet._update_preview()
        assert {a.classification for a in sheet._annotations()} == {"B"}
        sheet.reject()

    def test_geometry_panel_only_describes_checked(self, qtbot, slide, store, tmp_path):
        from pathlearn.data.geometry_bank import GeometryBank
        from pathlearn.ui.panels.geometry_panel import GeometryPanel

        panel = GeometryPanel(GeometryBank(tmp_path / "g.json"))
        qtbot.addWidget(panel)
        store.set_selected(store.annotations[0].id, False)
        panel.set_slide(slide, store)
        assert "1 Checked" in panel.describe_button.text()
        assert "1 unchecked" in panel.describe_button.text()
        panel.shutdown()


class TestSourceIsRecorded:
    def test_patch_records_slide_and_annotation(self, bank):
        annotation = uuid.uuid4()
        bank.add(patch(slide="9729.svs", annotation_id=annotation))
        stored = bank.fetch()[0]
        assert stored.slide_name == "9729.svs"
        assert stored.slide_path == "E:/9729.svs"
        assert stored.annotation_id == annotation

    def test_slides_lists_source_files_with_counts(self, bank):
        bank.add_many([patch(slide="a.svs") for _ in range(3)]
                      + [patch(slide="b.svs")])
        rows = {name: count for _, name, count in bank.slides()}
        assert rows == {"a.svs": 3, "b.svs": 1}

    def test_slides_is_ordered_by_size(self, bank):
        bank.add_many([patch(slide="small.svs")]
                      + [patch(slide="big.svs") for _ in range(5)])
        assert bank.slides()[0][1] == "big.svs"

    def test_slides_on_empty_bank(self, bank):
        assert bank.slides() == []


class TestDeletion:
    def test_delete_one_patch_by_id(self, bank):
        rows = [patch(x=i) for i in range(5)]
        bank.add_many(rows)
        assert bank.delete_ids([rows[2].id]) == 1
        assert len(bank.fetch()) == 4
        assert rows[2].id not in {p.id for p in bank.fetch()}

    def test_delete_several_ids(self, bank):
        rows = [patch(x=i) for i in range(5)]
        bank.add_many(rows)
        assert bank.delete_ids([r.id for r in rows[:3]]) == 3
        assert len(bank.fetch()) == 2

    def test_delete_ids_ignores_unknown(self, bank):
        bank.add(patch())
        assert bank.delete_ids([uuid.uuid4()]) == 0
        assert len(bank.fetch()) == 1

    def test_delete_empty_list(self, bank):
        bank.add(patch())
        assert bank.delete_ids([]) == 0

    def test_delete_beyond_the_parameter_limit(self, bank):
        """SQLite caps host parameters per statement, so this is chunked."""
        rows = [patch(x=i) for i in range(1200)]
        bank.add_many(rows)
        assert bank.delete_ids([r.id for r in rows]) == 1200
        assert bank.stats().is_empty

    def test_delete_all_from_one_slide(self, bank):
        bank.add_many([patch(slide="a.svs") for _ in range(4)]
                      + [patch(slide="b.svs") for _ in range(2)])
        assert bank.delete_where(slide_path="E:/a.svs") == 4
        assert {p.slide_name for p in bank.fetch()} == {"b.svs"}


class TestPatchBrowser:
    @pytest.fixture
    def browser(self, qtbot, bank):
        bank.add_many([patch(slide="a.svs", classification="Acinar", x=i * 10)
                       for i in range(6)]
                      + [patch(slide="b.svs", classification="PanIN-3", x=i * 10)
                         for i in range(4)])
        widget = PatchBrowser(bank)
        qtbot.addWidget(widget)
        yield widget
        widget.reject()

    def test_lists_every_patch(self, browser):
        assert browser.table.rowCount() == 10

    def test_shows_the_source_slide(self, browser):
        names = {browser.table.item(r, 0).text() for r in range(browser.table.rowCount())}
        assert names == {"a.svs", "b.svs"}

    def test_tooltip_carries_the_full_path_and_annotation(self, browser):
        tooltip = browser.table.item(0, 0).toolTip()
        assert "E:/" in tooltip and "annotation" in tooltip

    def test_filter_by_slide(self, browser):
        index = browser.slide_filter.findData("b.svs")
        browser.slide_filter.setCurrentIndex(index)
        assert browser.table.rowCount() == 4

    def test_filter_by_class(self, browser):
        index = browser.class_filter.findData("Acinar")
        browser.class_filter.setCurrentIndex(index)
        assert browser.table.rowCount() == 6

    def test_delete_selected_removes_only_those_rows(self, browser, bank):
        browser.table.selectRow(0)
        browser.table.selectRow(1)
        ids = browser._selected_ids()
        assert len(ids) >= 1
        removed = bank.delete_ids(ids)
        browser.reload()
        assert bank.stats().total == 10 - removed

    def test_delete_button_reflects_selection(self, browser):
        assert not browser.delete_selected.isEnabled()
        browser.table.selectRow(0)
        assert browser.delete_selected.isEnabled()
        assert "(1)" in browser.delete_selected.text()

    def test_empty_after_deleting_everything(self, browser, bank):
        bank.clear()
        browser.reload()
        assert browser.table.rowCount() == 0
        assert "0 of 0" in browser.status.text()


class TestSelectionIsSeparateFromVisibility:
    """The bug: macOS sidecars carry isVisible=false freely.

    An annotation hidden to see the tissue beneath was saved that way, so
    treating visibility as inclusion silently dropped most of a slide's
    annotations from analysis the moment it was loaded.

    The per-annotation Show control is gone now — every annotation is drawn,
    and ``is_visible`` survives only so a file round-trips unchanged. These
    pin that it is inert: nothing in the app may read it as a gate again.
    """

    def test_hidden_annotations_are_still_used(self, store):
        for annotation in store.annotations:
            annotation.is_visible = False
        assert len(store.in_use()) == 2

    def test_unselecting_excludes_without_hiding(self, store):
        first = store.annotations[0]
        store.set_selected(first.id, False)
        assert len(store.in_use()) == 1
        assert first.is_visible is True

    def test_legacy_sidecar_loads_everything_as_selected(self, tmp_path, slide):
        """A file written before `selected` existed must not exclude anything."""
        from pathlearn.io import geojson
        from pathlearn.models.store import AnnotationStore

        source = AnnotationStore()
        source.bind(tmp_path / "legacy.svs", slide.dimensions.height)
        for i in range(3):
            source.add(Annotation(points=[Point(i * 50, 0), Point(i * 50 + 40, 0),
                                          Point(i * 50 + 40, 40)],
                                  classification="A", color=AnnotationColor.default(),
                                  is_visible=False))
        # Strip the key, as a macOS-written file would not have it.
        import json
        path = tmp_path / "legacy.geojson"
        doc = json.loads(path.read_text(encoding="utf-8"))
        for feature in doc["features"]:
            feature["properties"].pop("selected", None)
        path.write_text(json.dumps(doc), encoding="utf-8")

        reloaded = AnnotationStore()
        reloaded.bind(tmp_path / "legacy.svs", slide.dimensions.height)
        # The flag survives the round trip so the file is not rewritten with
        # information removed — but it gates nothing.
        assert all(a.is_visible is False for a in reloaded.annotations)
        assert len(reloaded.in_use()) == 3

    def test_selection_round_trips(self, tmp_path, slide):
        from pathlearn.models.store import AnnotationStore
        store = AnnotationStore()
        store.bind(tmp_path / "rt.svs", slide.dimensions.height)
        a = Annotation(points=[Point(0, 0), Point(40, 0), Point(40, 40)],
                       classification="A", color=AnnotationColor.default())
        store.add(a)
        store.set_selected(a.id, False)

        reloaded = AnnotationStore()
        reloaded.bind(tmp_path / "rt.svs", slide.dimensions.height)
        assert reloaded.annotations[0].is_selected is False
        assert reloaded.in_use() == []

    def test_select_all_bulk_fixes_a_slide(self, store):
        for annotation in store.annotations:
            store.set_selected(annotation.id, False)
        assert store.in_use() == []
        assert store.select_all(True) == 2
        assert len(store.in_use()) == 2

    def test_select_all_reports_no_change(self, store):
        assert store.select_all(True) == 0

    def test_second_slide_offers_its_annotations(self, tmp_path, slide):
        """The reported bug, end to end: describe slide A, then open slide B."""
        from pathlearn.models.store import AnnotationStore

        first = AnnotationStore()
        first.bind(tmp_path / "a.svs", slide.dimensions.height)
        first.add(Annotation(points=[Point(0, 0), Point(40, 0), Point(40, 40)],
                             classification="A", color=AnnotationColor.default()))

        second = AnnotationStore()
        second.bind(tmp_path / "b.svs", slide.dimensions.height)
        for i in range(4):
            second.add(Annotation(points=[Point(i * 50, 0), Point(i * 50 + 40, 0),
                                          Point(i * 50 + 40, 40)],
                                  classification="B",
                                  color=AnnotationColor.default(),
                                  is_visible=(i == 0)))
        assert len(second.in_use()) == 4


class TestSidebarBulkUse:
    """`Use All` is the repair for a slide whose annotations arrived unticked."""

    @pytest.fixture
    def sidebar(self, qtbot, store):
        from pathlearn.models.classification import ClassificationProfile
        from pathlearn.ui.panels.annotation_sidebar import AnnotationSidebar

        widget = AnnotationSidebar(store, ClassificationProfile.default())
        qtbot.addWidget(widget)
        widget.refresh()
        return widget

    def test_use_none_then_use_all(self, sidebar, store):
        sidebar.use_none_button.click()
        assert store.in_use() == []
        sidebar.use_all_button.click()
        assert len(store.in_use()) == 2

    def test_bulk_use_leaves_the_legacy_flag_alone(self, sidebar, store):
        for annotation in store.annotations:
            annotation.is_visible = False
        sidebar.use_all_button.click()
        assert all(a.is_visible is False for a in store.annotations)
        assert len(store.in_use()) == 2

    def test_the_table_follows(self, sidebar, store):
        sidebar.use_none_button.click()
        states = [sidebar.tree.topLevelItem(r).checkState(0)
                  for r in range(sidebar.tree.topLevelItemCount())]
        assert states == [Qt.CheckState.Unchecked] * 2

    def test_repaint_is_requested(self, sidebar, qtbot):
        with qtbot.waitSignal(sidebar.annotations_changed, timeout=1000):
            sidebar.use_none_button.click()


class TestDeleteCheckedBatch:
    """Delete acts on the Use ticks, so the batch is what the table shows."""

    @pytest.fixture
    def sidebar(self, qtbot, store):
        from pathlearn.models.classification import ClassificationProfile
        from pathlearn.ui.panels.annotation_sidebar import AnnotationSidebar

        widget = AnnotationSidebar(store, ClassificationProfile.default())
        qtbot.addWidget(widget)
        widget.refresh()
        return widget

    def test_button_counts_what_will_go(self, sidebar, store):
        assert "Delete Checked (2)" == sidebar.delete_checked_button.text()
        store.set_selected(store.annotations[0].id, False)
        sidebar.refresh()
        assert "Delete Checked (1)" == sidebar.delete_checked_button.text()

    def test_disabled_when_nothing_is_checked(self, sidebar, store):
        store.select_all(False)
        sidebar.refresh()
        assert not sidebar.delete_checked_button.isEnabled()

    def test_deletes_only_the_checked_ones(self, sidebar, store):
        keep = store.annotations[0]
        store.set_selected(keep.id, False)
        removed = sidebar.delete_checked_now(store.in_use(include_subtractive=True))
        assert removed == 1
        assert [a.id for a in store.annotations] == [keep.id]

    def test_deletes_everything_when_everything_is_checked(self, sidebar, store):
        assert sidebar.delete_checked_now(store.in_use(include_subtractive=True)) == 2
        assert store.annotations == []

    def test_checked_subtractive_regions_go_too(self, sidebar, store):
        hole = Annotation(points=[Point(0, 0), Point(100, 0), Point(100, 100)],
                          classification="hole", color=AnnotationColor.default(),
                          is_subtractive=True)
        store.add(hole)
        sidebar.refresh()
        assert "(3)" in sidebar.delete_checked_button.text()
        sidebar.delete_checked_now(store.in_use(include_subtractive=True))
        assert store.annotations == []

    def test_an_unchecked_subtractive_region_survives(self, sidebar, store):
        hole = Annotation(points=[Point(0, 0), Point(100, 0), Point(100, 100)],
                          classification="hole", color=AnnotationColor.default(),
                          is_subtractive=True)
        store.add(hole)
        store.set_selected(hole.id, False)
        sidebar.delete_checked_now(store.in_use(include_subtractive=True))
        assert [a.id for a in store.annotations] == [hole.id]

    def test_the_table_and_canvas_follow(self, qtbot, sidebar, store):
        with qtbot.waitSignal(sidebar.annotations_changed, timeout=1000):
            sidebar.delete_checked_now(store.in_use(include_subtractive=True))
        assert sidebar.tree.topLevelItemCount() == 0

    def test_the_sidecar_is_written_once(self, sidebar, store, tmp_path, monkeypatch):
        saves = []
        monkeypatch.setattr(type(store), "save",
                            lambda self: saves.append(len(self.annotations)))
        sidebar.delete_checked_now(store.in_use(include_subtractive=True))
        assert saves == [0]

    def test_row_delete_still_targets_one(self, sidebar, store):
        """The single-row Delete button is unchanged; the batch is separate."""
        first = store.annotations[0]
        store.selected_id = first.id
        store.remove(first.id)
        assert len(store.annotations) == 1


class TestRemoveMany:
    def test_ignores_unknown_ids(self, store):
        assert store.remove_many([uuid.uuid4()]) == 0
        assert len(store.annotations) == 2

    def test_empty_batch_is_a_no_op(self, store):
        assert store.remove_many([]) == 0

    def test_clears_the_selection_when_it_is_deleted(self, store):
        first = store.annotations[0]
        store.selected_id = first.id
        store.remove_many([first.id])
        assert store.selected_id is None

    def test_keeps_a_selection_that_survives(self, store):
        first, second = store.annotations
        store.selected_id = second.id
        store.remove_many([first.id])
        assert store.selected_id == second.id


class TestShowColumnIsGone:
    """The per-annotation Show checkbox was removed; these pin what replaced it."""

    @pytest.fixture
    def sidebar(self, qtbot, store):
        from pathlearn.models.classification import ClassificationProfile
        from pathlearn.ui.panels.annotation_sidebar import AnnotationSidebar

        widget = AnnotationSidebar(store, ClassificationProfile.default())
        qtbot.addWidget(widget)
        widget.refresh()
        return widget

    def test_four_columns(self, sidebar):
        assert sidebar.tree.columnCount() == 4
        headers = [sidebar.tree.headerItem().text(c) for c in range(4)]
        assert headers == ["Use", "Class", "Name", "Area"]

    def test_only_the_use_column_is_checkable(self, sidebar, store):
        item = sidebar.tree.topLevelItem(0)
        assert item.checkState(0) == Qt.CheckState.Checked
        # Column 1 now carries the class name and its colour swatch.
        assert item.text(1) == store.annotations[0].classification

    def test_ticking_still_drives_selection(self, sidebar, store):
        item = sidebar.tree.topLevelItem(0)
        item.setCheckState(0, Qt.CheckState.Unchecked)
        assert store.annotations[0].is_selected is False

    def test_the_count_label_drops_the_shown_tally(self, sidebar):
        text = sidebar.count_label.text()
        assert "in use" in text
        assert "shown" not in text

    def test_the_store_no_longer_offers_visibility_controls(self, store):
        assert not hasattr(store, "set_visibility")
        assert not hasattr(store, "visible_count")

    def test_a_hidden_annotation_is_still_painted(self, qtbot, slide, store):
        """Regression for real slide 9738, which carries isVisible=false on 12/13.

        Paints onto a pixmap and counts non-background pixels, because the only
        thing that matters here is whether the polygon reaches the canvas.
        """
        from PySide6.QtGui import QPainter, QPixmap
        from pathlearn.ui.canvas import SlideCanvas

        canvas = SlideCanvas(store)
        qtbot.addWidget(canvas)
        canvas.resize(400, 300)
        canvas.set_slide(slide)
        canvas.zoom_to_fit()

        def painted_pixels() -> int:
            pixmap = QPixmap(400, 300)
            pixmap.fill()                       # white
            painter = QPainter(pixmap)
            canvas._paint_annotations(painter)
            painter.end()
            image = pixmap.toImage()
            return sum(1 for y in range(0, 300, 4) for x in range(0, 400, 4)
                       if image.pixelColor(x, y).rgb() != 0xFFFFFFFF)

        before = painted_pixels()
        assert before > 0, "fixture draws nothing; the test would prove nothing"
        for annotation in store.annotations:
            annotation.is_visible = False
        assert painted_pixels() == before
        canvas.shutdown()


class TestClassColourSwatch:
    """The colour beside the class picker is the control for changing it."""

    @pytest.fixture
    def sidebar(self, qtbot, store):
        from pathlearn.models.classification import ClassificationProfile
        from pathlearn.ui.panels.annotation_sidebar import AnnotationSidebar

        widget = AnnotationSidebar(store, ClassificationProfile.default())
        qtbot.addWidget(widget)
        widget.refresh()
        return widget

    def test_the_swatch_shows_the_current_class_colour(self, sidebar):
        cls = sidebar.current_class
        style = sidebar.color_button.styleSheet()
        assert f"rgb({cls.color.r}, {cls.color.g}, {cls.color.b})" in style

    def test_it_follows_the_class_picker(self, sidebar):
        first = sidebar.color_button.styleSheet()
        other = next(i for i in range(sidebar.class_combo.count())
                     if sidebar.profile.classes[i].color
                     != sidebar.current_class.color)
        sidebar.class_combo.setCurrentIndex(other)
        assert sidebar.color_button.styleSheet() != first

    def test_the_tooltip_names_the_class(self, sidebar):
        assert sidebar.current_class.name in sidebar.color_button.toolTip()

    def test_setting_a_colour_updates_the_class(self, sidebar):
        name = sidebar.current_class.name
        sidebar.set_class_color(name, AnnotationColor(1, 2, 3))
        assert sidebar.profile.by_name(name).color == AnnotationColor(1, 2, 3)

    def test_the_swatch_repaints_after_a_change(self, sidebar):
        sidebar.set_class_color(sidebar.current_class.name,
                                AnnotationColor(1, 2, 3))
        assert "rgb(1, 2, 3)" in sidebar.color_button.styleSheet()

    def test_existing_annotations_are_recoloured(self, sidebar, store):
        """The fixture's annotations use ad-hoc classes, so adopt a real one."""
        target = sidebar.current_class.name
        store.annotations[0].classification = target
        changed = sidebar.set_class_color(target, AnnotationColor(9, 9, 9))
        assert changed == 1
        assert store.annotations[0].color == AnnotationColor(9, 9, 9)

    def test_other_classes_are_left_alone(self, sidebar, store):
        target = sidebar.current_class.name
        store.annotations[0].classification = target
        before = store.annotations[1].color
        sidebar.set_class_color(target, AnnotationColor(9, 9, 9))
        assert store.annotations[1].color == before

    def test_a_class_outside_the_profile_is_not_recoloured(self, sidebar, store):
        """Only a class the profile defines can be recoloured."""
        assert sidebar.profile.by_name(store.annotations[0].classification) is None
        assert sidebar.set_class_color(store.annotations[0].classification,
                                       AnnotationColor(9, 9, 9)) == 0

    def test_new_annotations_take_the_new_colour(self, sidebar, store):
        name = sidebar.current_class.name
        sidebar.set_class_color(name, AnnotationColor(4, 5, 6))
        assert store.current_color == AnnotationColor(4, 5, 6)

    def test_an_unknown_class_changes_nothing(self, sidebar, store):
        assert sidebar.set_class_color("NotAClass", AnnotationColor(0, 0, 0)) == 0

    def test_a_repaint_is_requested(self, qtbot, sidebar):
        with qtbot.waitSignal(sidebar.annotations_changed, timeout=1000):
            sidebar.set_class_color(sidebar.current_class.name,
                                    AnnotationColor(7, 7, 7))

    def test_an_empty_profile_disables_the_swatch(self, qtbot, store):
        from pathlearn.models.classification import ClassificationProfile
        from pathlearn.ui.panels.annotation_sidebar import AnnotationSidebar

        empty = ClassificationProfile.default()
        empty.classes = []
        widget = AnnotationSidebar(store, empty)
        qtbot.addWidget(widget)
        assert not widget.color_button.isEnabled()
