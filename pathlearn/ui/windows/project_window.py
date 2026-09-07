"""The project table: every slide's composition side by side.

Non-modal on purpose. The whole workflow is open a slide, predict, add, move
on — so the table has to stay visible while you work rather than blocking the
window it is summarising.

Blank cells are left blank. A blank means that slide's model had no such
class; a 0.00 means it looked and found none. Filling blanks with zeros would
turn "not measured" into "measured as absent", so the table never does it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ...project import ANNOTATED, Project, ProjectError

FIXED_COLUMNS = ["Slide", "Source", "Added", "Model", "Count",
                 "Total area"]


class ProjectWindow(QWidget):
    """Shows the accumulated per-slide breakdown and exports it."""

    #: Emitted when an entry is removed, so the window can re-save.
    changed = Signal()

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.project = project
        self.resize(940, 520)

        layout = QVBoxLayout(self)

        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.headline)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #d0762a;")
        layout.addWidget(self.warning)

        self.table = QTableWidget(0, len(FIXED_COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        # Qt's default sort indicator is DESCENDING on column 0, so without
        # this the cohort opens in reverse alphabetical order and looks
        # shuffled. Say which order we mean, and let the header show it.
        self.table.horizontalHeader().setSortIndicator(
            0, Qt.SortOrder.AscendingOrder)
        layout.addWidget(self.table, 1)

        note = QLabel(
            "A slide can carry one predicted row and one annotated row: they "
            "measure different things over different denominators, so compare "
            "within a source rather than across. A blank cell means the class "
            "was never on the table for that row — it is not a zero.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #bbb;")
        layout.addWidget(note)

        row = QHBoxLayout()
        self.remove_button = QPushButton("Remove Row")
        self.remove_button.clicked.connect(self.remove_selected)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        for text, slot in (("Copy Table", self.copy_table),
                           ("Export CSV…", self.export_csv),
                           ("Export Excel…", self.export_xlsx)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        row.addWidget(close_button)
        layout.addLayout(row)

        self.reload()

    # -- contents ---------------------------------------------------------

    def set_project(self, project: Project) -> None:
        self.project = project
        self.reload()

    def reload(self) -> None:
        project = self.project
        self.setWindowTitle(f"Project — {project.name}")
        self.headline.setText(project.summary())

        mixed = project.mixed_extractors()
        if len(mixed) > 1:
            self.warning.setText(
                "These entries came from different feature extractors ("
                + ", ".join(mixed) + "). Two feature spaces are not "
                "comparable — re-run the odd ones out before reading across "
                "the rows.")
            self.warning.setVisible(True)
        else:
            self.warning.setVisible(False)

        classes = project.class_union()
        headers = FIXED_COLUMNS + [f"{label} %" for label in classes]
        self.table.setSortingEnabled(False)
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(project.entries))

        for row, entry in enumerate(project.entries):
            fixed = [entry.slide_name, entry.source, entry.added,
                     entry.model_name or "—",
                     f"{entry.total_tiles:,} {entry.unit_name}",
                     entry.area_text()]
            for column, text in enumerate(fixed):
                item = QTableWidgetItem(text)
                if column >= 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if column == 0:
                    # Both, because a slide can carry one row per source and
                    # Remove must take the row you picked, not the slide.
                    item.setData(Qt.ItemDataRole.UserRole,
                                 (entry.slide_path, entry.source))
                    item.setToolTip(entry.slide_path)
                if column == 1 and entry.is_annotated:
                    item.setToolTip(
                        f"What you drew ({entry.scope or 'all annotations'}), "
                        "counted in regions.")
                elif column == 1:
                    item.setToolTip("What the model called it, counted in "
                                    "tiles.")
                self.table.setItem(row, column, item)

            for offset, label in enumerate(classes):
                percent = entry.percent_for(label)
                item = QTableWidgetItem("" if percent is None
                                        else f"{percent:.1f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                if percent is None:
                    item.setToolTip("This slide's model had no such class — "
                                    "not measured, not zero.")
                self.table.setItem(row, len(FIXED_COLUMNS) + offset, item)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(headers)):
            header.setSectionResizeMode(column,
                                        QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSortingEnabled(True)
        self.remove_button.setEnabled(bool(project.entries))

    # -- actions ----------------------------------------------------------

    def selected_slide(self) -> tuple[str, str] | None:
        """The (slide path, source) of the single selected row."""
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None
        item = self.table.item(rows.pop(), 0)
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def remove_selected(self) -> None:
        selection = self.selected_slide()
        if selection is None:
            QMessageBox.information(self, "Remove Row",
                                    "Select one row to remove.")
            return
        slide, source = selection
        name = Path(slide).name or slide
        what = ("the annotation breakdown" if source == ANNOTATED
                else "the predicted composition")
        confirm = QMessageBox.question(
            self, "Remove Row",
            f"Remove {what} for {name} from this project?\n\n"
            "Only this row goes — the slide, its annotations, its predictions "
            "and any other row for the same slide are untouched.")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.project.remove(slide, source)
        self.reload()
        self.changed.emit()

    def copy_table(self) -> None:
        QApplication.clipboard().setText(self.project.to_csv())
        self.headline.setText("Copied to the clipboard.")

    def export_csv(self) -> None:
        default = self._default_export(".csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Project as CSV", str(default), "CSV (*.csv)")
        if not path:
            return
        target = Path(path)
        try:
            target.write_text(self.project.to_csv(), encoding="utf-8")
            # The long form goes beside it: one row per slide and class is
            # what a stats package wants, and exporting only one shape means
            # someone reshapes it by hand later.
            long_target = target.with_name(target.stem + "-long.csv")
            long_target.write_text(self.project.to_csv(layout="long"),
                                   encoding="utf-8")
        except (OSError, ProjectError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.headline.setText(
            f"Exported {target.name} and {long_target.name}.")

    def export_xlsx(self) -> None:
        default = self._default_export(".xlsx")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Project as Excel", str(default), "Excel (*.xlsx)")
        if not path:
            return
        try:
            written = self.project.to_xlsx(path)
        except ProjectError as exc:
            QMessageBox.warning(self, "Excel export", str(exc))
            return
        self.headline.setText(f"Exported {written.name} "
                              "(Composition, Detail and About sheets).")

    def _default_export(self, suffix: str) -> Path:
        stem = self.project.name.replace("/", "-").replace("\\", "-")
        folder = (self.project.path.parent if self.project.path
                  else Path.home())
        return folder / f"{stem}{suffix}"
