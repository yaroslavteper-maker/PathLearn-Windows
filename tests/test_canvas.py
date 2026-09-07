"""Canvas view-transform tests.

These pin the slide<->screen mapping, which is where a Y mirror would sneak
back in.  They need a QApplication but never show a window.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from pathlearn.io.slide import SlideImage
from pathlearn.models.annotation import Annotation, AnnotationColor, Point
from pathlearn.models.store import AnnotationStore
from pathlearn.ui.canvas import MAX_ZOOM, MIN_ZOOM, SlideCanvas, Tool
from synthetic_slide import CORNER_COLORS, corner_centre, write_synthetic_slide


def qtbot_click(canvas, x, y, kind):
    """Deliver a left-button press or release at widget coordinates."""
    held = (Qt.MouseButton.LeftButton if kind is QEvent.Type.MouseButtonPress
            else Qt.MouseButton.NoButton)
    event = QMouseEvent(kind, QPointF(x, y), QPointF(x, y),
                        Qt.MouseButton.LeftButton, held,
                        Qt.KeyboardModifier.NoModifier)
    if kind is QEvent.Type.MouseButtonPress:
        canvas.mousePressEvent(event)
    else:
        canvas.mouseReleaseEvent(event)


WIDTH, HEIGHT = 4096, 3072


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    path = tmp_path_factory.mktemp("slides") / "canvas.tif"
    write_synthetic_slide(path, WIDTH, HEIGHT, levels=4)
    with SlideImage(path) as s:
        yield s


@pytest.fixture
def canvas(qtbot, slide):
    widget = SlideCanvas(AnnotationStore())
    qtbot.addWidget(widget)
    widget.resize(800, 600)
    widget.set_slide(slide)
    yield widget
    widget.shutdown()


class TestTransform:
    def test_round_trip(self, canvas):
        for x, y in [(0, 0), (WIDTH, HEIGHT), (1234.5, 987.25), (-500, 20_000)]:
            back = canvas.screen_to_slide(canvas.slide_to_screen(x, y))
            assert back.x() == pytest.approx(x, abs=1e-6)
            assert back.y() == pytest.approx(y, abs=1e-6)

    def test_centre_maps_to_widget_centre(self, canvas):
        canvas.center = QPointF(1000, 800)
        screen = canvas.slide_to_screen(1000, 800)
        assert screen.x() == pytest.approx(canvas.width() / 2)
        assert screen.y() == pytest.approx(canvas.height() / 2)

    def test_y_is_not_mirrored(self, canvas):
        """Larger slide Y must map to larger screen Y.  This is the whole point."""
        canvas.center = QPointF(WIDTH / 2, HEIGHT / 2)
        top = canvas.slide_to_screen(WIDTH / 2, 0)
        bottom = canvas.slide_to_screen(WIDTH / 2, HEIGHT)
        assert top.y() < bottom.y()

    def test_x_is_not_mirrored(self, canvas):
        left = canvas.slide_to_screen(0, HEIGHT / 2)
        right = canvas.slide_to_screen(WIDTH, HEIGHT / 2)
        assert left.x() < right.x()

    def test_scale_is_isotropic(self, canvas):
        """One slide pixel must cover the same screen distance in x and y."""
        origin = canvas.slide_to_screen(0, 0)
        dx = canvas.slide_to_screen(100, 0).x() - origin.x()
        dy = canvas.slide_to_screen(0, 100).y() - origin.y()
        assert dx == pytest.approx(dy)

    def test_visible_rect_covers_widget(self, canvas):
        rect = canvas.visible_slide_rect()
        top_left = canvas.screen_to_slide(QPointF(0, 0))
        assert rect.left() == pytest.approx(top_left.x())
        assert rect.top() == pytest.approx(top_left.y())
        assert rect.width() == pytest.approx(canvas.width() / canvas.zoom)


class TestZoom:
    def test_fit_shows_whole_slide(self, canvas):
        canvas.zoom_to_fit()
        rect = canvas.visible_slide_rect()
        assert rect.left() <= 0 and rect.top() <= 0
        assert rect.right() >= WIDTH and rect.bottom() >= HEIGHT

    def test_zoom_is_clamped(self, canvas):
        canvas.zoom_by(1e9)
        assert canvas.zoom == pytest.approx(MAX_ZOOM)
        canvas.zoom_by(1e-12)
        assert canvas.zoom == pytest.approx(MIN_ZOOM)

    def test_zoom_anchor_stays_put(self, canvas):
        """Wheel-zoom must pin the slide point under the pointer."""
        anchor = QPointF(200, 150)
        before = canvas.screen_to_slide(anchor)
        canvas.zoom_by(2.0, anchor=anchor)
        after = canvas.screen_to_slide(anchor)
        assert after.x() == pytest.approx(before.x(), abs=1e-6)
        assert after.y() == pytest.approx(before.y(), abs=1e-6)

    def test_level_follows_zoom(self, canvas):
        canvas.zoom = 1.0
        assert canvas.current_level == 0
        canvas.zoom = 1 / 8
        assert canvas.current_level == 3

    def test_zoom_to_annotation_centres_it(self, canvas):
        ann = Annotation(points=[Point(1000, 800), Point(1400, 800), Point(1400, 1200),
                                 Point(1000, 1200)],
                         classification="PaNIN-2", color=AnnotationColor(240, 180, 30))
        canvas.zoom_to_annotation(ann)
        assert canvas.center.x() == pytest.approx(1200)
        assert canvas.center.y() == pytest.approx(1000)


class TestCornerAgreement:
    """Screen position of each corner marker must match its colour's corner."""

    @pytest.mark.parametrize("corner", list(CORNER_COLORS))
    def test_corner_lands_in_right_quadrant(self, canvas, corner):
        canvas.zoom_to_fit()
        cx, cy = corner_centre(corner, WIDTH, HEIGHT)
        screen = canvas.slide_to_screen(cx, cy)
        mid_x, mid_y = canvas.width() / 2, canvas.height() / 2

        expect_left = corner.endswith("left")
        expect_top = corner.startswith("top")
        assert (screen.x() < mid_x) is expect_left, f"{corner} on wrong side (x)"
        assert (screen.y() < mid_y) is expect_top, f"{corner} on wrong side (y)"


class TestTools:
    def test_polygon_draft_commits(self, canvas):
        canvas.set_tool(Tool.POLYGON)
        canvas._draft = [Point(10, 10), Point(200, 10), Point(200, 200)]
        canvas.commit_draft()
        assert len(canvas.store.annotations) == 1
        assert canvas.store.annotations[0].points[0] == Point(10, 10)

    def test_too_few_points_is_rejected(self, canvas):
        canvas.set_tool(Tool.POLYGON)
        canvas._draft = [Point(10, 10), Point(200, 10)]
        canvas.commit_draft()
        assert canvas.store.annotations == []
        assert not canvas.has_draft

    def test_switching_tool_cancels_draft(self, canvas):
        canvas.set_tool(Tool.POLYGON)
        canvas._draft = [Point(10, 10), Point(200, 10)]
        canvas.set_tool(Tool.LASSO)
        assert not canvas.has_draft

    def test_undo_draft_point(self, canvas):
        canvas.set_tool(Tool.POLYGON)
        canvas._draft = [Point(10, 10), Point(200, 10), Point(200, 200)]
        canvas.undo_draft_point()
        assert len(canvas._draft) == 2

    def test_committed_annotation_uses_current_class(self, canvas):
        canvas.store.current_label = "PaNIN-3"
        canvas.store.current_color = AnnotationColor(220, 60, 60)
        canvas.set_tool(Tool.POLYGON)
        canvas._draft = [Point(0, 0), Point(100, 0), Point(100, 100)]
        canvas.commit_draft()
        ann = canvas.store.annotations[-1]
        assert ann.classification == "PaNIN-3"
        assert ann.color == AnnotationColor(220, 60, 60)


class TestLassoClickToClose:
    """Click once to start tracing, click again to close the loop.

    Driven through real mouse events rather than by poking ``_draft``: the
    whole feature is which press/move/release sequence means what, and that is
    exactly what setting the attribute directly would skip.
    """

    def press(self, canvas, x, y):
        qtbot_click(canvas, x, y, QEvent.Type.MouseButtonPress)

    def release(self, canvas, x, y):
        qtbot_click(canvas, x, y, QEvent.Type.MouseButtonRelease)

    def move(self, canvas, x, y, buttons=Qt.MouseButton.NoButton):
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
                            Qt.MouseButton.NoButton, buttons,
                            Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(event)

    def trace(self, canvas, points, buttons=Qt.MouseButton.NoButton):
        for x, y in points:
            self.move(canvas, x, y, buttons)

    # -- the drag lasso must keep working unchanged ------------------------

    def test_drag_still_commits_on_release(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.trace(canvas, [(160, 100), (160, 160), (100, 160)],
                   Qt.MouseButton.LeftButton)
        self.release(canvas, 100, 160)
        assert len(canvas.store.annotations) == 1
        assert not canvas.has_draft

    def test_a_drag_does_not_leave_it_tracing(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.trace(canvas, [(160, 100), (160, 160)], Qt.MouseButton.LeftButton)
        self.release(canvas, 160, 160)
        assert not canvas._lasso_sticky

    # -- click to start ----------------------------------------------------

    def test_a_click_starts_a_hands_free_trace(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        assert canvas._lasso_sticky
        assert canvas.has_draft
        assert canvas.store.annotations == []

    def test_tiny_jitter_still_counts_as_a_click(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.trace(canvas, [(102, 101)], Qt.MouseButton.LeftButton)
        self.release(canvas, 102, 101)
        assert canvas._lasso_sticky

    def test_the_outline_follows_the_cursor_with_no_button_held(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        before = len(canvas._draft)
        self.trace(canvas, [(140, 100), (180, 140), (140, 180)])
        assert len(canvas._draft) > before

    # -- click again to close ---------------------------------------------

    def test_the_second_click_closes_the_loop(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        self.trace(canvas, [(200, 100), (200, 200), (100, 200)])
        self.press(canvas, 100, 200)
        assert len(canvas.store.annotations) == 1
        assert not canvas.has_draft
        assert not canvas._lasso_sticky

    def test_the_stored_ring_starts_where_the_trace_started(self, canvas):
        canvas.set_tool(Tool.LASSO)
        start = canvas.screen_to_slide(QPointF(100, 100))
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        self.trace(canvas, [(200, 100), (200, 200), (100, 200)])
        self.press(canvas, 100, 200)
        first = canvas.store.annotations[0].points[0]
        assert first.x == pytest.approx(start.x(), abs=1.0)
        assert first.y == pytest.approx(start.y(), abs=1.0)

    def test_the_ring_is_stored_open_and_closes_implicitly(self, canvas):
        """No duplicate closing vertex; the polygon closes on its first point."""
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        self.trace(canvas, [(200, 100), (200, 200), (100, 200)])
        self.press(canvas, 100, 200)
        points = canvas.store.annotations[0].points
        assert points[0] != points[-1]

    def test_closing_starts_no_new_draft(self, canvas):
        """The closing press must not double as the start of the next region."""
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        self.trace(canvas, [(200, 100), (200, 200), (100, 200)])
        self.press(canvas, 100, 200)
        self.release(canvas, 100, 200)
        assert not canvas.has_draft
        assert len(canvas.store.annotations) == 1

    def test_two_regions_in_a_row(self, canvas):
        canvas.set_tool(Tool.LASSO)
        for origin in (100, 300):
            self.press(canvas, origin, 100)
            self.release(canvas, origin, 100)
            self.trace(canvas, [(origin + 80, 100), (origin + 80, 180), (origin, 180)])
            self.press(canvas, origin, 180)
            self.release(canvas, origin, 180)
        assert len(canvas.store.annotations) == 2

    # -- getting out of it -------------------------------------------------

    def test_escape_cancels_the_trace(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        self.trace(canvas, [(200, 100), (200, 200)])
        canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                       Qt.KeyboardModifier.NoModifier))
        assert not canvas.has_draft
        assert not canvas._lasso_sticky
        assert canvas.store.annotations == []

    def test_enter_closes_the_trace(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        self.trace(canvas, [(200, 100), (200, 200), (100, 200)])
        canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                                       Qt.KeyboardModifier.NoModifier))
        assert len(canvas.store.annotations) == 1

    def test_switching_tool_abandons_the_trace(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        canvas.set_tool(Tool.PAN)
        assert not canvas._lasso_sticky
        assert not canvas.has_draft

    def test_closing_too_early_is_refused(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        self.press(canvas, 101, 100)
        assert canvas.store.annotations == []
        assert not canvas.has_draft


class TestDoubleClickCloses:
    """Both drawing tools close on a double-click.

    These go through ``qtbot`` rather than calling the handlers directly,
    because the point being tested is Qt's own delivery: the second click of a
    double-click arrives as MouseButtonDblClick, *not* as a second press. A
    hand-built event sequence would just re-assert my assumption about that.
    """

    def point(self, canvas, x, y):
        return QPoint(int(x), int(y))

    def test_polygon_closes_on_double_click(self, qtbot, canvas):
        canvas.set_tool(Tool.POLYGON)
        for x, y in [(100, 100), (300, 100), (300, 300)]:
            qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton,
                             pos=self.point(canvas, x, y))
        qtbot.mouseDClick(canvas, Qt.MouseButton.LeftButton,
                          pos=self.point(canvas, 100, 300))
        assert len(canvas.store.annotations) == 1
        assert not canvas.has_draft

    def test_polygon_keeps_the_double_clicked_vertex(self, qtbot, canvas):
        canvas.set_tool(Tool.POLYGON)
        for x, y in [(100, 100), (300, 100)]:
            qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton,
                             pos=self.point(canvas, x, y))
        qtbot.mouseDClick(canvas, Qt.MouseButton.LeftButton,
                          pos=self.point(canvas, 300, 300))
        # The double-clicked position has to become a vertex, whether or not a
        # press preceded the double-click; otherwise this is two points and is
        # rejected.
        assert len(canvas.store.annotations) == 1
        assert len(canvas.store.annotations[0].points) == 3

    def test_polygon_still_closes_on_the_first_vertex(self, qtbot, canvas):
        canvas.set_tool(Tool.POLYGON)
        for x, y in [(100, 100), (300, 100), (300, 300)]:
            qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton,
                             pos=self.point(canvas, x, y))
        qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton,
                         pos=self.point(canvas, 104, 103))
        assert len(canvas.store.annotations) == 1

    def test_polygon_double_click_too_early_is_refused(self, qtbot, canvas):
        canvas.set_tool(Tool.POLYGON)
        qtbot.mouseDClick(canvas, Qt.MouseButton.LeftButton,
                          pos=self.point(canvas, 100, 100))
        assert canvas.store.annotations == []
        assert not canvas.has_draft

    def test_lasso_closes_on_a_fast_double_click(self, qtbot, canvas):
        """The gap a slow-click-only implementation leaves open.

        Pointer motion is delivered by calling the handler: QTest.mouseMove
        does not reach an unshown widget, so a qtbot move here would silently
        contribute nothing and the test would pass for the wrong reason.
        """
        canvas.set_tool(Tool.LASSO)
        qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton,
                         pos=self.point(canvas, 100, 100))
        assert canvas._lasso_sticky
        for x, y in [(300, 100), (300, 300), (100, 300)]:
            canvas.mouseMoveEvent(QMouseEvent(
                QEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
                Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier))
        assert len(canvas._draft) >= 3, "moves did not reach the draft"
        qtbot.mouseDClick(canvas, Qt.MouseButton.LeftButton,
                          pos=self.point(canvas, 100, 300))
        assert len(canvas.store.annotations) == 1
        assert not canvas._lasso_sticky

    def test_double_click_in_pan_still_selects(self, qtbot, canvas):
        canvas.set_tool(Tool.PAN)
        canvas.zoom_to_fit()
        ann = Annotation(points=[Point(0, 0), Point(WIDTH, 0),
                                 Point(WIDTH, HEIGHT), Point(0, HEIGHT)],
                         classification="A", color=AnnotationColor.default())
        canvas.store.add(ann)
        canvas.store.selected_id = None
        qtbot.mouseDClick(canvas, Qt.MouseButton.LeftButton,
                          pos=self.point(canvas, canvas.width() // 2,
                                         canvas.height() // 2))
        assert canvas.store.selected_id == ann.id

    def test_double_click_does_not_start_a_stray_draft(self, qtbot, canvas):
        canvas.set_tool(Tool.POLYGON)
        for x, y in [(100, 100), (300, 100), (300, 300)]:
            qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton,
                             pos=self.point(canvas, x, y))
        qtbot.mouseDClick(canvas, Qt.MouseButton.LeftButton,
                          pos=self.point(canvas, 100, 300))
        assert not canvas.has_draft


class TestRealQtDoubleClickOrdering:
    """Qt delivers a real double-click as press, release, dblclick, release.

    QTest.mouseDClick sends only the dblclick, so that ordering — the one an
    actual user produces — has to be built by hand. This is where a duplicate
    vertex would appear: the press places one, and the dblclick must not place
    a second at the same spot.
    """

    def press(self, canvas, x, y):
        qtbot_click(canvas, x, y, QEvent.Type.MouseButtonPress)

    def release(self, canvas, x, y):
        qtbot_click(canvas, x, y, QEvent.Type.MouseButtonRelease)

    def dblclick(self, canvas, x, y):
        canvas.mouseDoubleClickEvent(QMouseEvent(
            QEvent.Type.MouseButtonDblClick, QPointF(x, y), QPointF(x, y),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier))

    def test_polygon_does_not_duplicate_the_final_vertex(self, canvas):
        canvas.set_tool(Tool.POLYGON)
        for x, y in [(100, 100), (300, 100)]:
            self.press(canvas, x, y)
            self.release(canvas, x, y)
        self.press(canvas, 300, 300)        # click 1 of the pair: places it
        self.release(canvas, 300, 300)
        self.dblclick(canvas, 300, 300)     # click 2: closes, adds nothing
        points = canvas.store.annotations[0].points
        assert len(points) == 3
        assert points[-1] != points[-2]

    def test_polygon_ring_has_no_repeated_closing_point(self, canvas):
        canvas.set_tool(Tool.POLYGON)
        for x, y in [(100, 100), (300, 100), (300, 300)]:
            self.press(canvas, x, y)
            self.release(canvas, x, y)
        self.dblclick(canvas, 300, 300)
        points = canvas.store.annotations[0].points
        assert points[0] != points[-1], "ring must stay open; consumers close it"

    def test_lasso_closes_on_the_press_and_the_dblclick_is_harmless(self, canvas):
        canvas.set_tool(Tool.LASSO)
        self.press(canvas, 100, 100)
        self.release(canvas, 100, 100)
        for x, y in [(300, 100), (300, 300), (100, 300)]:
            canvas.mouseMoveEvent(QMouseEvent(
                QEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
                Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier))
        self.press(canvas, 100, 300)        # this already commits
        self.release(canvas, 100, 300)
        self.dblclick(canvas, 100, 300)     # must not start or commit anything
        assert len(canvas.store.annotations) == 1
        assert not canvas.has_draft
