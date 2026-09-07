"""Browse and delete patches — the row-level view of the bank.

Every patch already records where it came from: the slide path and name, the
annotation id, and its level-0 coordinates. Until now none of that was
visible. This shows it, and makes it actionable — delete one bad patch, or
everything that came from one slide.

Deletion is permanent and there is no undo, so both paths confirm with a count
and the export in the Patch Bank panel remains the recovery route.
"""

from __future__ import annotations

import uuid

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ...data.bank import PatchBank

#: Rows shown at once. A bank can hold tens of thousands, and populating a
#: QTableWidget with all of them is slow enough to feel broken.
PAGE_SIZE = 2000


class PatchBrowser(QDialog):
    """A filterable table of patches, with per-row and per-slide deletion."""

    def __init__(self, bank: PatchBank, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Browse Patches")
        self.resize(940, 620)
        self.bank = bank
        self.rows: list = []

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_filters())

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Slide", "Class", "X", "Y", "White", "Nuclei"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.table, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #bbb;")
        layout.addWidget(self.status)

        actions = QHBoxLayout()
        self.delete_selected = QPushButton("Delete Selected")
        self.delete_selected.clicked.connect(self._delete_selected)
        self.delete_slide = QPushButton("Delete All From Slide…")
        self.delete_slide.clicked.connect(self._delete_slide)
        actions.addWidget(self.delete_selected)
        actions.addWidget(self.delete_slide)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.reload()

    # -- construction -----------------------------------------------------

    def _build_filters(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Slide"))
        self.slide_filter = QComboBox()
        self.slide_filter.currentIndexChanged.connect(self.reload)
        row.addWidget(self.slide_filter, 2)

        row.addWidget(QLabel("Class"))
        self.class_filter = QComboBox()
        self.class_filter.currentIndexChanged.connect(self.reload)
        row.addWidget(self.class_filter, 2)

        row.addWidget(QLabel("Extractor"))
        self.extractor_filter = QComboBox()
        self.extractor_filter.currentIndexChanged.connect(self.reload)
        row.addWidget(self.extractor_filter, 2)
        return row

    def _refresh_filters(self) -> None:
        stats = self.bank.stats()
        for combo, values in ((self.slide_filter, sorted(stats.by_slide)),
                              (self.class_filter, sorted(stats.by_class)),
                              (self.extractor_filter, sorted(stats.by_extractor))):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All", userData=None)
            for value in values:
                combo.addItem(value, userData=value)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    # -- data -------------------------------------------------------------

    def reload(self) -> None:
        self._refresh_filters()
        filters = {}
        if self.class_filter.currentData():
            filters["classifications"] = [self.class_filter.currentData()]
        if self.extractor_filter.currentData():
            filters["extractor_identity"] = self.extractor_filter.currentData()

        rows = self.bank.fetch(**filters)
        slide = self.slide_filter.currentData()
        if slide:
            # fetch() filters on slide_path; the combo lists display names, so
            # match on the name to keep the two consistent.
            rows = [p for p in rows if p.slide_name == slide]

        self.rows = rows[:PAGE_SIZE]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.rows))
        for index, patch in enumerate(self.rows):
            cells = (patch.slide_name, patch.classification, str(patch.patch_x),
                     str(patch.patch_y), f"{patch.white_fraction:.2f}",
                     "—" if patch.nucleus_count < 0 else str(patch.nucleus_count))
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, patch.id)
                    item.setToolTip(f"{patch.slide_path}\n"
                                    f"annotation {patch.annotation_id}\n"
                                    f"{patch.extractor_identity}")
                self.table.setItem(index, column, item)
        self.table.setSortingEnabled(True)

        shown = len(self.rows)
        total = len(rows)
        text = f"{shown} of {total} matching patch(es)"
        if total > PAGE_SIZE:
            text += f" — showing the first {PAGE_SIZE}; narrow the filters to see the rest"
        self.status.setText(text)
        self._update_buttons()

    def _selected_ids(self) -> list[uuid.UUID]:
        ids = []
        for index in {i.row() for i in self.table.selectedIndexes()}:
            item = self.table.item(index, 0)
            if item is not None:
                value = item.data(Qt.ItemDataRole.UserRole)
                if value is not None:
                    ids.append(value)
        return ids

    def _update_buttons(self) -> None:
        count = len(self._selected_ids())
        self.delete_selected.setEnabled(count > 0)
        self.delete_selected.setText(
            f"Delete Selected ({count})" if count else "Delete Selected")
        self.delete_slide.setEnabled(bool(self.bank.slides()))

    # -- deletion ---------------------------------------------------------

    def _delete_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        if QMessageBox.warning(
            self, "Delete Patches",
            f"Delete {len(ids)} patch(es)?\n\nThis cannot be undone. Export the "
            f"bank first if you might want them back.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        removed = self.bank.delete_ids(ids)
        self.reload()
        self.status.setText(f"Deleted {removed} patch(es).")

    def _delete_slide(self) -> None:
        slides = self.bank.slides()
        if not slides:
            return

        # Offer the slide currently filtered to, else the largest.
        current = self.slide_filter.currentData()
        options = {f"{name}  ({count} patches)": path
                   for path, name, count in slides}
        preselect = 0
        for i, (path, name, _) in enumerate(slides):
            if name == current:
                preselect = i
                break

        from PySide6.QtWidgets import QInputDialog
        label, ok = QInputDialog.getItem(
            self, "Delete All From Slide",
            "Every patch extracted from this slide will be deleted:",
            list(options), preselect, False)
        if not ok:
            return

        path = options[label]
        count = next((c for p, _, c in slides if p == path), 0)
        if QMessageBox.warning(
            self, "Delete All From Slide",
            f"Delete all {count} patch(es) from this slide?\n\n{path}\n\n"
            f"This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return

        removed = self.bank.delete_where(slide_path=path)
        self.reload()
        self.status.setText(f"Deleted {removed} patch(es) from that slide.")
