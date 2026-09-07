"""Clicking a region on the slide selects it and reveals it in the list."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.classification import ClassificationProfile
from pathlearn.models.store import AnnotationStore
from pathlearn.ui.canvas import PAN_CLICK_SLOP, SlideCanvas, Tool
from pathlearn.ui.panels.annotation_sidebar import AnnotationSidebar
from synthetic_slide import write_synthetic_slide

WIDTH, HEIGHT = 4096, 3072


def press(canvas, x, y, button=Qt.MouseButton.LeftButton):
    canvas.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(x, y), QPointF(x, y),
        button, button, Qt.KeyboardModifier.NoModifier))


def release(canvas, x, y, button=Qt.MouseButton.LeftButton):
    canvas.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(x, y), QPointF(x, y),
        button, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))


def click(canvas, x, y, **kw):
    press(canvas, x, y, **kw)
    release(canvas, x, y, **kw)


def double_click(canvas, x, y):
    canvas.mouseDoubleClickEvent(QMouseEvent(
        QEvent.Type.MouseButtonDblClick, QPointF(x, y), QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))


def box(store, x, y, side, label="PanIN-2"):
    annotation = Annotation(
        points=[Point(x, y), Point(x + side, y),
                Point(x + side, y + side), Point(x, y + side)],
        classification=label, color=AnnotationColor(200, 60, 60))
    store.add(annotation)
    # store.add selects what it adds, which would make every "did not select"
    # assertion below pass for the wrong reason.
    store.selected_id = None
    return annotation


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("slides") / "click.tif"
    write_synthetic_slide(path, WIDTH, HEIGHT, levels=4)
    with SlideImage(path) as s:
        yield s


@pytest.fixture
def canvas(qtbot, slide):
    widget = SlideCanvas(AnnotationStore())
    qtbot.addWidget(widget)
    widget.resize(800, 600)
    widget.set_slide(slide)
    widget.tool = Tool.PAN
    yield widget
    widget.shutdown()


def screen_of(canvas, annotation):
    """Widget coordinates of an annotation's centre."""
    x, y, w, h = annotation.bounding_box
    return canvas.slide_to_screen(x + w / 2, y + h / 2)


class TestClickingSelects:
    def test_a_single_click_selects_the_region_under_it(self, canvas):
        annotation = box(canvas.store, 100, 100, 800)
        point = screen_of(canvas, annotation)
        click(canvas, point.x(), point.y())
        assert canvas.store.selected_id == annotation.id

    def test_it_emits_so_the_window_can_follow(self, canvas, qtbot):
        annotation = box(canvas.store, 100, 100, 800)
        point = screen_of(canvas, annotation)
        with qtbot.waitSignal(canvas.selection_changed, timeout=500) as caught:
            click(canvas, point.x(), point.y())
        assert caught.args[0] == annotation.id

    def test_clicking_empty_slide_clears_the_selection(self, canvas):
        annotation = box(canvas.store, 100, 100, 400)
        canvas.store.selected_id = annotation.id
        point = canvas.slide_to_screen(WIDTH - 50, HEIGHT - 50)
        click(canvas, point.x(), point.y())
        assert canvas.store.selected_id is None

    def test_the_topmost_region_wins_when_they_overlap(self, canvas):
        box(canvas.store, 100, 100, 2000)
        small = box(canvas.store, 200, 200, 300, label="Acinar")
        point = screen_of(canvas, small)
        click(canvas, point.x(), point.y())
        assert canvas.store.selected_id == canvas.store.hit_test(
            *[c for c in (350, 350)]).id

    def test_double_click_still_selects(self, canvas):
        """The old gesture must keep working, not toggle the new one off."""
        annotation = box(canvas.store, 100, 100, 800)
        point = screen_of(canvas, annotation)
        click(canvas, point.x(), point.y())
        double_click(canvas, point.x(), point.y())
        assert canvas.store.selected_id == annotation.id


class TestPanningStillWorks:
    def test_a_drag_pans_and_does_not_select(self, canvas):
        annotation = box(canvas.store, 100, 100, 2000)
        point = screen_of(canvas, annotation)
        before = QPointF(canvas.center)
        press(canvas, point.x(), point.y())
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(point.x() + 120, point.y() + 90),
            QPointF(point.x() + 120, point.y() + 90),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier))
        release(canvas, point.x() + 120, point.y() + 90)
        assert canvas.store.selected_id is None
        assert QPointF(canvas.center) != before

    def test_a_tremor_within_the_slop_still_counts_as_a_click(self, canvas):
        annotation = box(canvas.store, 100, 100, 2000)
        point = screen_of(canvas, annotation)
        press(canvas, point.x(), point.y())
        release(canvas, point.x() + PAN_CLICK_SLOP - 1, point.y())
        assert canvas.store.selected_id == annotation.id

    def test_just_beyond_the_slop_is_a_drag(self, canvas):
        box(canvas.store, 100, 100, 2000)
        point = canvas.slide_to_screen(500, 500)
        press(canvas, point.x(), point.y())
        release(canvas, point.x() + PAN_CLICK_SLOP + 3, point.y())
        assert canvas.store.selected_id is None

    def test_the_middle_button_pans_without_selecting(self, canvas):
        annotation = box(canvas.store, 100, 100, 2000)
        point = screen_of(canvas, annotation)
        press(canvas, point.x(), point.y(), button=Qt.MouseButton.MiddleButton)
        release(canvas, point.x(), point.y(),
                button=Qt.MouseButton.MiddleButton)
        assert canvas.store.selected_id is None

    def test_a_drawing_tool_does_not_select(self, canvas):
        """In Lasso a click starts a trace; it must not also select."""
        annotation = box(canvas.store, 100, 100, 2000)
        canvas.tool = Tool.LASSO
        point = screen_of(canvas, annotation)
        click(canvas, point.x(), point.y())
        assert canvas.store.selected_id is None
        canvas.cancel_draft()


class TestTheListFollows:
    def sidebar(self, qtbot, count=60):
        store = AnnotationStore()
        for i in range(count):
            box(store, i * 30, 0, 20, label=f"C{i % 3}")
        widget = AnnotationSidebar(store, ClassificationProfile.default())
        qtbot.addWidget(widget)
        widget.resize(320, 200)
        widget.refresh()
        return widget, store

    def test_it_highlights_the_row(self, qtbot):
        sidebar, store = self.sidebar(qtbot)
        target = store.annotations[40]
        assert sidebar.select_annotation(target.id) is True
        current = sidebar.tree.currentItem()
        assert current.data(0, Qt.ItemDataRole.UserRole) == target.id
        assert current.isSelected()

    def test_it_scrolls_the_row_into_view(self, qtbot):
        """The reason this method exists: row 40 of 60 is off the bottom."""
        sidebar, store = self.sidebar(qtbot)
        sidebar.show()
        target = store.annotations[55]
        sidebar.select_annotation(target.id)
        item = sidebar.tree.currentItem()
        visible = sidebar.tree.viewport().rect()
        assert visible.intersects(sidebar.tree.visualItemRect(item))

    def test_selecting_nothing_clears_the_row(self, qtbot):
        sidebar, store = self.sidebar(qtbot)
        sidebar.select_annotation(store.annotations[3].id)
        assert sidebar.select_annotation(None) is False
        assert sidebar.tree.currentItem() is None

    def test_an_unknown_id_is_not_an_error(self, qtbot):
        import uuid

        sidebar, _ = self.sidebar(qtbot)
        assert sidebar.select_annotation(uuid.uuid4()) is False

    def test_revealing_does_not_look_like_the_user_clicking(self, qtbot):
        """Otherwise the sidebar would write the selection straight back."""
        sidebar, store = self.sidebar(qtbot)
        received = []
        sidebar.annotations_changed.connect(lambda: received.append(1))
        sidebar.select_annotation(store.annotations[10].id)
        assert received == []
