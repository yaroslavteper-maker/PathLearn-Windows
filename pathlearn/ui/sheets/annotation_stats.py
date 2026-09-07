"""Class breakdown of the annotations you drew, by count and by area.

Two bars, not one, because the two answers genuinely differ: a slide can be
90% PanIN-1a by count and mostly grade 3 by area, and showing only one of
those is how the same slide gets described two opposite ways.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QDialog, QDialogButtonBox, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ...models.annotation import Annotation
from ...pipeline.annotation_stats import breakdown
from ...pipeline.composition import format_area
from .composition import ProportionBar

COLUMNS = ["Class", "Regions", "Count %", "Area", "Area %", "Mean size"]


class AnnotationStatsSheet(QDialog):
    """What fraction of the tracing each class accounts for."""

    #: Asks the window to file this breakdown in the open project. The sheet
    #: does not own the project, and does not know the slide it belongs to.
    add_to_project_requested = Signal()

    def __init__(self, annotations: list[Annotation], *,
                 in_use: list[Annotation] | None = None,
                 mpp: float | None = None,
                 project_name: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annotation Class Breakdown")
        self.resize(720, 560)
        self.all_annotations = list(annotations)
        self.in_use_annotations = list(in_use if in_use is not None
                                       else annotations)
        self.mpp = mpp
        self.project_name = project_name
        self.report = None
        self.added_to_project = False

        layout = QVBoxLayout(self)

        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.headline)

        self.use_only = QCheckBox("Only annotations checked under Use")
        self.use_only.setToolTip(
            "Restrict the breakdown to the annotations currently ticked, so "
            "it matches the set you are about to extract or delete.")
        self.use_only.setEnabled(
            len(self.in_use_annotations) != len(self.all_annotations))
        self.use_only.toggled.connect(lambda _: self.recalculate())
        layout.addWidget(self.use_only)

        layout.addWidget(_caption("By count — how many regions carry each class"))
        self.count_bar = ProportionBar()
        layout.addWidget(self.count_bar)

        layout.addWidget(_caption("By area — how much slide each class covers"))
        self.area_bar = ProportionBar()
        layout.addWidget(self.area_bar)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicator(
            0, Qt.SortOrder.AscendingOrder)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(COLUMNS)):
            header.setSectionResizeMode(column,
                                        QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        self.caveat = QLabel("")
        self.caveat.setWordWrap(True)
        self.caveat.setStyleSheet("color: #b8860b;")
        layout.addWidget(self.caveat)

        row = QHBoxLayout()
        self.project_button = QPushButton(
            f"Add to \u201c{project_name}\u201d" if project_name
            else "Add to Project")
        self.project_button.setEnabled(bool(project_name))
        self.project_button.setToolTip(
            "File this slide's annotation breakdown in the open project, as "
            "its own row alongside any predicted one."
            if project_name else
            "No project is open. Project \u25b8 New Project\u2026 to start one.")
        self.project_button.clicked.connect(self._add_to_project)
        row.addWidget(self.project_button)

        copy_button = QPushButton("Copy Table")
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

        self.recalculate()

    # -- contents ---------------------------------------------------------

    def current_annotations(self) -> list[Annotation]:
        return (self.in_use_annotations if self.use_only.isChecked()
                else self.all_annotations)

    def recalculate(self) -> None:
        scope = ("annotations checked under Use" if self.use_only.isChecked()
                 else "all annotations")
        self.report = breakdown(self.current_annotations(), mpp=self.mpp,
                                scope=scope)
        self.headline.setText(self.report.summary())
        self._fill_bars()
        self._fill_table()
        self.caveat.setText(self._caveat_text())
        self.caveat.setVisible(bool(self.caveat.text()))

    def _colour_for(self, label: str) -> QColor:
        for annotation in self.current_annotations():
            if (annotation.classification or "Unlabeled") == label:
                colour = annotation.color
                return QColor(colour.r, colour.g, colour.b)
        return QColor(120, 120, 120)

    def _fill_bars(self) -> None:
        self.count_bar.set_slices([
            (t.label, t.count_fraction, self._colour_for(t.label))
            for t in self.report.tallies])
        self.area_bar.set_slices([
            (t.label, t.area_fraction, self._colour_for(t.label))
            for t in self.report.tallies])

    def _fill_table(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.report.tallies))
        for row, tally in enumerate(self.report.tallies):
            values = [tally.label, f"{tally.count:,}",
                      f"{tally.count_percent:.1f}%",
                      format_area(tally.area_px, tally.area_mm2),
                      f"{tally.area_percent:.1f}%",
                      format_area(tally.mean_area_px, tally.mean_area_mm2)]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, column, item)
            if tally.carved_px:
                self.table.item(row, 3).setToolTip(
                    f"{format_area(tally.carved_px, None)} carved out by "
                    "subtractive polygons and already deducted.")
            self.table.item(row, 0).setForeground(
                QBrush(self._colour_for(tally.label)))
        self.table.setSortingEnabled(True)

    def _caveat_text(self) -> str:
        report = self.report
        parts = []
        if report.subtractive_count:
            parts.append(
                f"{report.subtractive_count} subtractive polygon(s) are "
                "excluded from the counts and their area deducted from the "
                "region enclosing each.")
        if report.orphan_subtractive:
            parts.append(
                f"{report.orphan_subtractive} subtractive polygon(s) sit "
                "inside no region, so they deduct from nothing — check they "
                "are where you meant them.")
        if report.overlapping_pairs:
            parts.append(
                f"{report.overlapping_pairs} pair(s) of regions may overlap. "
                "Area sums polygon areas, so any overlap is counted twice and "
                "the area percentages are upper bounds.")
        if report.mpp is None and not report.is_empty:
            parts.append("This slide reports no pixel size, so areas are in "
                         "pixels rather than mm².")
        return "  ".join(parts)

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
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Class Breakdown",
            str(Path.home() / "annotation-classes.csv"), "CSV (*.csv)")
        if not path:
            return
        try:
            Path(path).write_text(self.report.to_csv(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.headline.setText(f"Saved {Path(path).name}.")


def _caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("color: #999; font-size: 11px;")
    return label
