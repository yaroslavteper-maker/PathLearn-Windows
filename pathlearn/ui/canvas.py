"""The slide canvas — tiled pan/zoom plus the annotation drawing tools.

Replaces ``Views/SlideScrollView.swift`` + ``SlideCanvasView.swift`` +
``AnnotationOverlayView.swift``.

COORDINATES
===========
The slide is drawn **upright**.  There is exactly one transform between slide
space (level-0 px, top-left origin, Y down) and widget space::

    screen = (slide - center) * zoom + widget_centre

``zoom`` is screen pixels per level-0 slide pixel, so ``zoom == 1`` shows the
slide at native resolution and ``zoom == 0.01`` is zoomed far out.  There is no
Y mirror anywhere — see :mod:`pathlearn.coords` for why the macOS build had one
and why we do not.
"""

from __future__ import annotations

import enum
import math
import uuid

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (QColor, QCursor, QFont, QKeyEvent, QMouseEvent, QPainter,
                           QPainterPath, QPen, QPolygonF, QWheelEvent)
from PySide6.QtWidgets import QWidget

from ..io.slide import SlideImage
from ..models.annotation import Annotation, AnnotationColor, Point
from ..models.store import AnnotationStore
from .tiles import TILE_SIZE, TileKey, TileManager

#: Zoom limits, matching the macOS NSScrollView magnification range.
MIN_ZOOM = 0.005
MAX_ZOOM = 20.0
#: Minimum pointer travel (screen px) before a lasso records another vertex.
LASSO_MIN_STEP = 3.0
#: Screen radius within which a polygon click snaps closed onto the first vertex.
POLYGON_CLOSE_RADIUS = 12.0
#: Pointer travel (screen px) below which a lasso press counts as a *click*
#: rather than the start of a drag, and so begins a hands-free trace.
LASSO_CLICK_SLOP = 6.0

#: A press and release within this many screen pixels is a click, not a
#: pan. Generous enough to survive the hand tremor of a real click on a
#: trackpad, small enough that a deliberate nudge of the slide still pans.
PAN_CLICK_SLOP = 4.0


class Tool(enum.Enum):
    PAN = "pan"
    LASSO = "lasso"
    POLYGON = "polygon"


class SlideCanvas(QWidget):
    """Renders the open slide and hosts the annotation tools."""

    #: Emitted whenever the view transform changes (for the status bar).
    view_changed = Signal()
    #: Emitted with a status message worth showing the user.
    status_message = Signal(str)
    #: Emitted when the selected annotation changes (id or None).
    selection_changed = Signal(object)

    def __init__(self, store: AnnotationStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)

        self.store = store
        self.slide: SlideImage | None = None
        self.tiles = TileManager(parent=self)
        self.tiles.tile_ready.connect(self.update)

        self.zoom: float = 1.0
        self.center = QPointF(0.0, 0.0)
        self.tool = Tool.PAN
        self.show_annotations = True
        self.show_heatmap = True
        self.predictions = None
        self._heatmap_drawn = 0

        self._panning = False
        self._pan_anchor = QPoint()
        self._pan_center_at_anchor = QPointF()
        self._draft: list[Point] = []          # in-progress polygon/lasso, slide coords
        self._drawing_lasso = False
        # Hands-free lasso: the outline follows the cursor with no button
        # held, and the next click closes it. Entered by clicking rather
        # than dragging, so the drag lasso keeps working unchanged.
        self._lasso_sticky = False
        self._lasso_press_pos: QPointF | None = None
        self._lasso_travel = 0.0
        self._cursor_slide: QPointF | None = None

    # -- slide lifecycle --------------------------------------------------

    def set_slide(self, slide: SlideImage | None) -> None:
        self.slide = slide
        self.tiles.set_slide(slide)
        self.cancel_draft()
        # Predictions belong to one slide; carrying them over would paint a
        # heatmap from the wrong tissue.
        self.predictions = None
        if slide is not None:
            self.zoom_to_fit()
        self.update()
        self.view_changed.emit()

    def shutdown(self) -> None:
        self.tiles.shutdown()

    # -- view transform ---------------------------------------------------

    def slide_to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(
            (x - self.center.x()) * self.zoom + self.width() / 2.0,
            (y - self.center.y()) * self.zoom + self.height() / 2.0,
        )

    def screen_to_slide(self, pos: QPointF | QPoint) -> QPointF:
        px, py = float(pos.x()), float(pos.y())
        return QPointF(
            (px - self.width() / 2.0) / self.zoom + self.center.x(),
            (py - self.height() / 2.0) / self.zoom + self.center.y(),
        )

    def visible_slide_rect(self) -> QRectF:
        top_left = self.screen_to_slide(QPointF(0, 0))
        bottom_right = self.screen_to_slide(QPointF(self.width(), self.height()))
        return QRectF(top_left, bottom_right).normalized()

    def zoom_to_fit(self) -> None:
        if self.slide is None or self.width() == 0 or self.height() == 0:
            return
        dims = self.slide.dimensions
        fit = min(self.width() / dims.width, self.height() / dims.height)
        self.zoom = _clamp(fit * 0.98, MIN_ZOOM, MAX_ZOOM)
        self.center = QPointF(dims.width / 2.0, dims.height / 2.0)
        self.update()
        self.view_changed.emit()

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        """Multiply zoom, keeping the slide point under *anchor* pinned."""
        if self.slide is None:
            return
        new_zoom = _clamp(self.zoom * factor, MIN_ZOOM, MAX_ZOOM)
        if new_zoom == self.zoom:
            return
        if anchor is None:
            self.zoom = new_zoom
        else:
            before = self.screen_to_slide(anchor)
            self.zoom = new_zoom
            after = self.screen_to_slide(anchor)
            self.center += before - after
        self._clamp_center()
        self.update()
        self.view_changed.emit()

    def zoom_to_annotation(self, annotation: Annotation) -> None:
        if self.slide is None or not annotation.points:
            return
        x, y, w, h = annotation.bounding_box
        if w <= 0 or h <= 0:
            return
        self.center = QPointF(x + w / 2.0, y + h / 2.0)
        fit = min(self.width() / w, self.height() / h)
        self.zoom = _clamp(fit * 0.8, MIN_ZOOM, MAX_ZOOM)
        self._clamp_center()
        self.update()
        self.view_changed.emit()

    def _clamp_center(self) -> None:
        """Keep at least a sliver of slide on screen at all times."""
        if self.slide is None:
            return
        dims = self.slide.dimensions
        margin_x = self.width() / (2.0 * self.zoom)
        margin_y = self.height() / (2.0 * self.zoom)
        self.center = QPointF(
            _clamp(self.center.x(), -margin_x * 0.5, dims.width + margin_x * 0.5),
            _clamp(self.center.y(), -margin_y * 0.5, dims.height + margin_y * 0.5),
        )

    @property
    def current_level(self) -> int:
        """Pyramid level appropriate to the current zoom."""
        if self.slide is None:
            return 0
        return self.slide.best_level_for_downsample(1.0 / max(self.zoom, 1e-9))

    @property
    def magnification_text(self) -> str:
        mpp = self.slide.mpp_x if self.slide else None
        if not mpp:
            return f"{self.zoom * 100:.1f}%"
        # 1 screen px covers (1/zoom) slide px; a 40x objective is ~0.25 um/px.
        return f"{0.25 / (mpp / self.zoom):.1f}x"

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 28, 30))
        if self.slide is None:
            self._paint_placeholder(painter)
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self._paint_tiles(painter)
        # The heatmap sits between slide and annotations so that region
        # outlines stay readable over a dense overlay.
        if self.show_heatmap and self.predictions is not None:
            self._paint_heatmap(painter)
        if self.show_annotations:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._paint_annotations(painter)
            self._paint_draft(painter)

    def _paint_heatmap(self, painter: QPainter) -> None:
        """Alpha-blended per-class rectangles over the predicted tiles.

        Only tiles intersecting the viewport are drawn: a whole-slide run can
        produce tens of thousands, and painting them all would make panning
        crawl. Alpha scales with confidence so a marginal call looks marginal.
        """
        predictions = self.predictions
        visible_rect = self.visible_slide_rect()
        painter.setPen(Qt.PenStyle.NoPen)

        drawn = 0
        for prediction in predictions.visible():
            size = prediction.size_level0
            if (prediction.x + size < visible_rect.left()
                    or prediction.x > visible_rect.right()
                    or prediction.y + size < visible_rect.top()
                    or prediction.y > visible_rect.bottom()):
                continue

            rgb = predictions.color_for(prediction.label)
            color = QColor(rgb.r, rgb.g, rgb.b)
            # 60..200 keeps a low-confidence tile visible but clearly weaker.
            color.setAlpha(int(60 + 140 * min(max(prediction.confidence, 0.0), 1.0)))
            painter.setBrush(color)

            top_left = self.slide_to_screen(prediction.x, prediction.y)
            edge = size * self.zoom
            painter.drawRect(QRectF(top_left.x(), top_left.y(), edge, edge))
            drawn += 1
        self._heatmap_drawn = drawn

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor(140, 140, 145))
        font = QFont()
        font.setPointSize(13)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                         "No slide open\n\nFile ▸ Open Slide…")

    def _paint_tiles(self, painter: QPainter) -> None:
        assert self.slide is not None
        level = self.current_level
        downsample = self.slide.level_downsamples[level]
        level_dims = self.slide.level_dimensions[level]
        visible = self.visible_slide_rect()

        # Visible rect in this level's pixels, expanded to whole tiles.
        first_col = max(0, int(visible.left() / downsample) // TILE_SIZE)
        last_col = min((level_dims.width - 1) // TILE_SIZE,
                       int(visible.right() / downsample) // TILE_SIZE)
        first_row = max(0, int(visible.top() / downsample) // TILE_SIZE)
        last_row = min((level_dims.height - 1) // TILE_SIZE,
                       int(visible.bottom() / downsample) // TILE_SIZE)

        for row in range(first_row, last_row + 1):
            for col in range(first_col, last_col + 1):
                key = TileKey(level, col, row)
                target = self._tile_target_rect(key, downsample)
                pixmap = self.tiles.tile(key)
                if pixmap is not None:
                    painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
                else:
                    # Nothing decoded yet: show a coarser level so panning never
                    # flashes empty, then let tile_ready repaint over it.
                    self._paint_fallback(painter, key, target, level)

    def _tile_target_rect(self, key: TileKey, downsample: float) -> QRectF:
        """Where a tile lands on screen, in widget coordinates."""
        x0 = key.col * TILE_SIZE * downsample
        y0 = key.row * TILE_SIZE * downsample
        top_left = self.slide_to_screen(x0, y0)
        size = TILE_SIZE * downsample * self.zoom
        return QRectF(top_left.x(), top_left.y(), size, size)

    def _paint_fallback(self, painter: QPainter, key: TileKey,
                        target: QRectF, level: int) -> None:
        """Blit whatever a coarser cached level already has for this area."""
        assert self.slide is not None
        for coarser in range(level + 1, self.slide.level_count):
            ratio = self.slide.level_downsamples[coarser] / self.slide.level_downsamples[level]
            if ratio <= 0:
                continue
            src_col = int(key.col / ratio)
            src_row = int(key.row / ratio)
            pixmap = self.tiles.cached_only(TileKey(coarser, src_col, src_row))
            if pixmap is None:
                continue
            # The sub-rectangle of the coarse tile covering this fine tile.
            span = TILE_SIZE / ratio
            sx = (key.col - src_col * ratio) * span
            sy = (key.row - src_row * ratio) * span
            source = QRectF(sx, sy, span, span).intersected(QRectF(pixmap.rect()))
            if not source.isEmpty():
                painter.drawPixmap(target, pixmap, source)
            return

    def _paint_annotations(self, painter: QPainter) -> None:
        selected_id = self.store.selected_id
        for ann in self.store.annotations:
            # Every annotation is drawn. Per-annotation visibility used to
            # be a checkbox; it hid the fact that a region existed, which
            # is worse than showing it faintly. View > Show Annotations
            # is the one remaining switch, and it covers all of them.
            if len(ann.points) < 3:
                continue
            polygon = QPolygonF([self.slide_to_screen(p.x, p.y) for p in ann.points])
            color = QColor(ann.color.r, ann.color.g, ann.color.b)
            is_selected = ann.id == selected_id

            fill = QColor(color)
            fill.setAlpha(70 if is_selected else 45)
            painter.setBrush(fill)

            pen = QPen(color, 3.0 if is_selected else 1.8)
            pen.setCosmetic(True)
            if not ann.is_selected:
                # Excluded from analysis: drawn as a faint outline with no
                # fill, so it is obvious at a glance which regions will not
                # contribute — quieter than hiding, which loses the fact.
                pen.setStyle(Qt.PenStyle.DotLine)
                pen.setWidthF(1.2)
                faded = QColor(color)
                faded.setAlpha(110)
                pen.setColor(faded)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            if ann.is_subtractive:
                # Subtractive polygons carve regions out; dash them so the
                # distinction is visible at a glance.
                pen.setStyle(Qt.PenStyle.DashLine)
                fill.setAlpha(20)
                painter.setBrush(fill)
            painter.setPen(pen)
            painter.drawPolygon(polygon)

            if is_selected:
                self._paint_vertices(painter, polygon, color)

    def _paint_vertices(self, painter: QPainter, polygon: QPolygonF, color: QColor) -> None:
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(color, 1.5))
        for pt in polygon:
            painter.drawEllipse(pt, 3.0, 3.0)

    def _paint_draft(self, painter: QPainter) -> None:
        if not self._draft:
            return
        color = QColor(self.store.current_color.r,
                       self.store.current_color.g,
                       self.store.current_color.b)
        screen_pts = [self.slide_to_screen(p.x, p.y) for p in self._draft]

        path = QPainterPath(screen_pts[0])
        for pt in screen_pts[1:]:
            path.lineTo(pt)
        rubber_band = (self.tool is Tool.POLYGON or self._lasso_sticky)
        if rubber_band and self._cursor_slide is not None:
            path.lineTo(self.slide_to_screen(self._cursor_slide.x(),
                                             self._cursor_slide.y()))
        if self._lasso_sticky and len(screen_pts) >= 2:
            # The loop always closes on the first point, so show that edge
            # rather than letting the user guess where it will land.
            path.lineTo(screen_pts[0])

        pen = QPen(color, 2.0)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        if self.tool is Tool.POLYGON:
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(color, 1.5))
            for pt in screen_pts:
                painter.drawEllipse(pt, 3.5, 3.5)

    # -- tools ------------------------------------------------------------

    def set_predictions(self, predictions) -> None:
        """Attach a PredictionSet to draw, or None to clear the overlay."""
        self.predictions = predictions
        self.update()

    def set_tool(self, tool: Tool) -> None:
        if tool is self.tool:
            return
        self.cancel_draft()
        self.tool = tool
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor if tool is Tool.PAN
                               else Qt.CursorShape.CrossCursor))
        self.update()

    def cancel_draft(self) -> None:
        self._reset_lasso_state()
        if self._draft:
            self._draft = []
            self.update()

    def _reset_lasso_state(self) -> None:
        """Clear the tracing flags. Every exit from a draft goes through here.

        ``_lasso_sticky`` has to be cleared on commit as well as on cancel:
        left set, the click that *starts* the next region would be read as the
        click that closes this one.
        """
        self._drawing_lasso = False
        self._lasso_sticky = False
        self._lasso_press_pos = None
        self._lasso_travel = 0.0

    def commit_draft(self) -> None:
        """Turn the in-progress outline into a stored annotation.

        The ring is stored open; every consumer closes it back to the first
        point — the canvas draws a polygon, the codec repeats the first
        coordinate — so committing *is* connecting the loop to its start.
        """
        if len(self._draft) < 3:
            self.status_message.emit("Need at least 3 points to close a region.")
            self.cancel_draft()
            return
        annotation = Annotation(
            points=list(self._draft),
            classification=self.store.current_label,
            color=self.store.current_color,
        )
        self._draft = []
        self._reset_lasso_state()
        self.store.add(annotation)
        self.selection_changed.emit(annotation.id)
        self.status_message.emit(
            f"Added {annotation.classification} ({annotation.area_short} px²)."
        )
        self.update()

    def undo_draft_point(self) -> None:
        if self._draft:
            self._draft.pop()
            self.update()

    # -- input ------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.slide is None:
            return
        pos = event.position()
        slide_pt = self.screen_to_slide(pos)

        middle_or_space_pan = (event.button() == Qt.MouseButton.MiddleButton
                               or event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if event.button() == Qt.MouseButton.LeftButton and (self.tool is Tool.PAN
                                                            or middle_or_space_pan):
            self._begin_pan(event.position().toPoint())
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._begin_pan(event.position().toPoint())
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.tool is Tool.LASSO:
            if self._lasso_sticky:
                # Second click: close the loop back to where it started.
                self.commit_draft()
                return
            self._drawing_lasso = True
            self._lasso_press_pos = QPointF(pos)
            self._lasso_travel = 0.0
            self._draft = [Point(slide_pt.x(), slide_pt.y())]
            self.update()
        elif self.tool is Tool.POLYGON:
            self._add_polygon_vertex(pos, slide_pt)

    def _begin_pan(self, pos: QPoint) -> None:
        self._panning = True
        self._pan_anchor = pos
        self._pan_center_at_anchor = QPointF(self.center)
        self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _add_polygon_vertex(self, screen_pos: QPointF, slide_pt: QPointF) -> None:
        if self._draft:
            first_screen = self.slide_to_screen(self._draft[0].x, self._draft[0].y)
            if (len(self._draft) >= 3
                    and _distance(first_screen, screen_pos) <= POLYGON_CLOSE_RADIUS):
                self.commit_draft()
                return
        self._draft.append(Point(slide_pt.x(), slide_pt.y()))
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.slide is None:
            return
        pos = event.position()
        self._cursor_slide = self.screen_to_slide(pos)

        if self._panning:
            delta = event.position().toPoint() - self._pan_anchor
            self.center = QPointF(
                self._pan_center_at_anchor.x() - delta.x() / self.zoom,
                self._pan_center_at_anchor.y() - delta.y() / self.zoom,
            )
            self._clamp_center()
            self.update()
            self.view_changed.emit()
            return

        if self._drawing_lasso and self._draft:
            if self._lasso_press_pos is not None:
                self._lasso_travel = max(self._lasso_travel,
                                         _distance(self._lasso_press_pos, pos))
            last_screen = self.slide_to_screen(self._draft[-1].x, self._draft[-1].y)
            if _distance(last_screen, pos) >= LASSO_MIN_STEP:
                self._draft.append(Point(self._cursor_slide.x(), self._cursor_slide.y()))
            self.update()      # repaint for the closing edge even when idle
            return

        if self.tool is Tool.POLYGON and self._draft:
            self.update()  # redraw the rubber-band edge
        self.view_changed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            anchor = self._pan_anchor
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor if self.tool is Tool.PAN
                                   else Qt.CursorShape.CrossCursor))
            # Pressed and released in the same spot: that was a click, not a
            # drag, so select what is under it. Every pan begins as a press in
            # Pan mode, which is why this has to be decided on release —
            # deciding on press would make selecting and panning the same
            # gesture and one of them would have to lose.
            if (self.tool is Tool.PAN
                    and event.button() == Qt.MouseButton.LeftButton
                    and anchor is not None
                    and _distance(QPointF(anchor),
                                  event.position()) <= PAN_CLICK_SLOP):
                self.select_at(event.position())
            return
        if self._drawing_lasso and event.button() == Qt.MouseButton.LeftButton:
            if not self._lasso_sticky and self._lasso_travel < LASSO_CLICK_SLOP:
                # Pressed and released without really moving: the user
                # clicked. Keep tracing without the button held.
                self._lasso_sticky = True
                self.status_message.emit(
                    "Tracing — move to draw, click again to close the loop, "
                    "Esc to cancel.")
                return
            self._drawing_lasso = False
            self.commit_draft()

    def select_at(self, screen_pos: QPointF) -> None:
        """Select whatever annotation is under *screen_pos*, or clear it."""
        slide_pt = self.screen_to_slide(screen_pos)
        hit = self.store.hit_test(slide_pt.x(), slide_pt.y())
        self.store.selected_id = hit.id if hit else None
        self.selection_changed.emit(self.store.selected_id)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Pan: select what is under the pointer. Drawing: close the outline.

        Qt delivers the *second* click of a double-click as this event instead
        of a second press, so the close paths in ``mousePressEvent`` are never
        reached by a quick double-click — both tools need it handled here or
        closing works only when the two clicks are slow enough to be separate.
        """
        if self.slide is None:
            return
        if self.tool is Tool.PAN:
            # Single click already selected this; repeating it keeps the two
            # gestures agreeing rather than toggling the selection back off.
            self.select_at(event.position())
            return
        if event.button() != Qt.MouseButton.LeftButton or not self._draft:
            return
        if self.tool is Tool.POLYGON:
            # Whether the first click of the pair already placed this vertex
            # depends on how the double-click reached us, so place it only if
            # the outline does not already end there. Adding it twice would
            # leave a zero-length edge in the stored ring.
            last = self.slide_to_screen(self._draft[-1].x, self._draft[-1].y)
            if _distance(last, event.position()) > POLYGON_CLOSE_RADIUS:
                slide_pt = self.screen_to_slide(event.position())
                self._draft.append(Point(slide_pt.x(), slide_pt.y()))
            self.commit_draft()
        elif self._lasso_sticky:
            self.commit_draft()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.slide is None:
            return
        degrees = event.angleDelta().y() / 8.0
        if degrees == 0:
            return
        # ~1.0015 per degree gives a smooth, trackpad-friendly ramp.
        self.zoom_by(math.pow(1.0015, degrees), anchor=event.position())
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.cancel_draft()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._draft and (self.tool is Tool.POLYGON or self._lasso_sticky):
                self.commit_draft()
        elif key == Qt.Key.Key_Backspace:
            if self._draft:
                self.undo_draft_point()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(1.25)
        elif key == Qt.Key.Key_Minus:
            self.zoom_by(1 / 1.25)
        elif key == Qt.Key.Key_0:
            self.zoom_to_fit()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._clamp_center()
        self.view_changed.emit()

    @property
    def has_draft(self) -> bool:
        return bool(self._draft)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _distance(a: QPointF, b: QPointF) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())
