"""The class breakdown of a prediction run, as a table and a proportion bar.

The bar is here because a table of percentages answers "how much" but not "is
one class dominant", and that is usually the first thing you want to see.  It
uses the same colours as the heatmap so the two read as one result.

The caveat under the table is not decoration.  These percentages are of the
tissue that was predicted over, and someone reading the number a month later
will not remember that unless it is written next to it — which is also why the
saved CSV carries the same provenance in its footer.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog,
                               QDialogButtonBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton,
                               QSizePolicy, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ...models.prediction import PredictionSet
from ...pipeline.composition import (CompositionReport, composition,
                                     format_area)

COLUMNS = ["Class", "Area %", "Area", "Tiles", "Tile %", "Mean confidence"]

#: A slice thinner than this cannot hold its own label legibly.
LABEL_FLOOR = 0.06


class ProportionBar(QWidget):
    """One horizontal bar, split by area share, in the heatmap's colours."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._slices: list[tuple[str, float, QColor]] = []

    def set_slices(self, slices: list[tuple[str, float, QColor]]) -> None:
        """(label, fraction, colour) — the generic entry point."""
        self._slices = list(slices)
        self.update()

    def set_report(self, report: CompositionReport,
                   predictions: PredictionSet) -> None:
        colour_of = predictions.color_for
        self.set_slices([
            (share.label, share.area_fraction,
             QColor(colour_of(share.label).r, colour_of(share.label).g,
                    colour_of(share.label).b))
            for share in report.shares])

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        height = self.height()
        if not self._slices or width <= 0:
            painter.fillRect(self.rect(), QBrush(QColor(60, 60, 60)))
            return

        x = 0.0
        for index, (label, fraction, colour) in enumerate(self._slices):
            # The last slice takes whatever is left, so rounding never leaves
            # a one-pixel gap at the right edge.
            span = (width - x) if index == len(self._slices) - 1 else fraction * width
            rectangle = (int(round(x)), 0, int(round(span)), height)
            painter.fillRect(*rectangle, QBrush(colour))
            if fraction >= LABEL_FLOOR:
                painter.setPen(QPen(_readable_on(colour)))
                painter.drawText(
                    int(round(x)), 0, int(round(span)), height,
                    Qt.AlignmentFlag.AlignCenter,
                    f"{label}  {fraction * 100:.0f}%")
            x += span
        painter.setPen(QPen(QColor(30, 30, 30)))
        painter.drawRect(0, 0, width - 1, height - 1)


def _readable_on(colour: QColor) -> QColor:
    """Black or white, whichever the eye can actually read on *colour*."""
    luminance = (0.299 * colour.red() + 0.587 * colour.green()
                 + 0.114 * colour.blue())
    return QColor(Qt.GlobalColor.black if luminance > 150 else Qt.GlobalColor.white)


class CompositionSheet(QDialog):
    """Shows what fraction of the predicted tissue each class occupies."""

    #: Asks the window to file this breakdown in the open project. The sheet
    #: does not own the project — the window does, and it is what knows the
    #: slide path this run belongs to.
    add_to_project_requested = Signal()

    def __init__(self, predictions: PredictionSet, *,
                 mpp: float | None = None,
                 project_name: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tissue Composition")
        self.resize(720, 520)
        self.predictions = predictions
        self.project_name = project_name
        self.report = composition(predictions, mpp=mpp)
        self.added_to_project = False

        layout = QVBoxLayout(self)

        self.headline = QLabel(self.report.summary())
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.headline)

        self.bar = ProportionBar()
        self.bar.set_report(self.report, predictions)
        layout.addWidget(self.bar)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(COLUMNS)):
            header.setSectionResizeMode(column,
                                        QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)
        self._fill_table()

        caveat = QLabel(self._caveat_text())
        caveat.setWordWrap(True)
        caveat.setStyleSheet("color: #b8860b;")
        layout.addWidget(caveat)

        row = QHBoxLayout()
        self.project_button = QPushButton(
            f"Add to “{project_name}”" if project_name else "Add to Project")
        self.project_button.setEnabled(bool(project_name)
                                       and not self.report.is_empty)
        self.project_button.setToolTip(
            "File this slide's breakdown in the open project, so it can be "
            "compared with the other slides in it."
            if project_name else
            "No project is open. Project ▸ New Project… to start one.")
        self.project_button.clicked.connect(self._add_to_project)
        row.addWidget(self.project_button)

        copy_button = QPushButton("Copy Table")
        copy_button.setToolTip("Copy the breakdown as CSV, ready to paste "
                               "into a spreadsheet.")
        copy_button.clicked.connect(self.copy_to_clipboard)
        row.addWidget(copy_button)

        save_button = QPushButton("Save CSV…")
        save_button.clicked.connect(self.save_csv)
        row.addWidget(save_button)
        row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        row.addWidget(buttons)
        layout.addLayout(row)

    # -- contents ---------------------------------------------------------

    def _caveat_text(self) -> str:
        text = ("Percentages are of the tissue that was predicted over — not "
                "of the whole slide. Tiles the sampler rejected as background "
                "were never classified and are not in the denominator.")
        if self.report.excluded_low_confidence:
            text += (f" {self.report.excluded_low_confidence} tile(s) below the "
                     f"{self.report.min_confidence:.2f} confidence threshold "
                     "are also excluded.")
        if self.report.mpp is None:
            text += (" This slide reports no pixel size, so areas are in "
                     "pixels rather than mm².")
        return text

    def _fill_table(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.report.shares))
        for row, share in enumerate(self.report.shares):
            area = format_area(share.area_px, share.area_mm2)
            values = [share.label, f"{share.area_percent:.1f}%", area,
                      f"{share.tiles:,}", f"{share.tile_percent:.1f}%",
                      f"{share.mean_confidence:.2f}"]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, column, item)
            colour = self.predictions.color_for(share.label)
            self.table.item(row, 0).setForeground(
                QBrush(QColor(colour.r, colour.g, colour.b)))
        self.table.setSortingEnabled(True)

    # -- actions ----------------------------------------------------------

    def _add_to_project(self) -> None:
        self.added_to_project = True
        self.add_to_project_requested.emit()

    def note(self, text: str) -> None:
        """Let the window report back into the sheet the user is looking at."""
        self.headline.setText(text)

    def copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.report.to_csv())
        self.headline.setText("Copied to the clipboard.")

    def save_csv(self) -> None:
        slide = self.predictions.slide_path
        stem = Path(slide).stem if slide else "predictions"
        default = Path.home() / f"{stem}-composition.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Composition", str(default), "CSV (*.csv)")
        if not path:
            return
        try:
            Path(path).write_text(self.report.to_csv(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.headline.setText(f"Saved {Path(path).name}.")
