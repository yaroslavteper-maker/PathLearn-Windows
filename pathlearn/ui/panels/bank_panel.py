"""Patch bank panel — from ``Views/MLBankPanel.swift``.

Shows what the bank holds, broken down by extractor and class, and offers the
`.bank` import/export that is the recovery path if the database is ever lost
(``04-DESIGN-DECISIONS.md`` §6).

The per-extractor breakdown is the important part of this view: a bank holding
two feature spaces is legal but cannot train a single model, and the user needs
to see that before the training sheet refuses.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...data.bank import PatchBank
from ..sheets.patch_browser import PatchBrowser


class BankPanel(QWidget):
    """Bank contents plus import/export/clear."""

    changed = Signal()

    def __init__(self, bank: PatchBank, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bank = bank

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.headline = QLabel()
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.headline)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Extractor / class", "Patches", "Dim"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree, 1)

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #d08a20;")
        self.warning.setVisible(False)
        layout.addWidget(self.warning)

        buttons = QHBoxLayout()
        for text, slot, tip in (
            ("Browse…", self.browse_patches,
             "See every patch with its source slide, and delete individually "
             "or by slide"),
            ("Export…", self.export_bank, "Write a .bank JSON snapshot — the recovery path"),
            ("Import…", self.import_bank, "Load a .bank file, from this build or macOS"),
            ("Clear…", self.clear_bank, "Delete every patch in the bank"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.path_label = QLabel(str(bank.path))
        self.path_label.setStyleSheet("color: #888; font-size: 11px;")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.refresh()

    # -- display ----------------------------------------------------------

    def refresh(self) -> None:
        stats = self.bank.stats()
        self.tree.clear()

        if stats.is_empty:
            self.headline.setText("Bank is empty")
            self.warning.setVisible(False)
            return

        self.headline.setText(
            f"{stats.total} patches · {len(stats.by_class)} classes · "
            f"{len(stats.by_slide)} slides")

        for identity, count in sorted(stats.by_extractor.items()):
            parent = QTreeWidgetItem([identity, str(count),
                                      str(stats.feature_dims.get(identity, "?"))])
            self.tree.addTopLevelItem(parent)
            parent.setExpanded(True)
            for name in sorted(stats.by_class):
                n = self.bank.count(extractor_identity=identity, classifications=[name])
                if n:
                    parent.addChild(QTreeWidgetItem([f"    {name}", str(n), ""]))

        for path, name, count in self.bank.slides():
            row = QTreeWidgetItem([f"slide: {name}", str(count), ""])
            row.setToolTip(0, path)
            self.tree.addTopLevelItem(row)

        if stats.is_mixed:
            self.warning.setText(
                "This bank holds more than one feature space. That is allowed, but "
                "a classifier is trained against exactly one extractor — training "
                "will ask you to choose.")
            self.warning.setVisible(True)
        else:
            self.warning.setVisible(False)

    # -- actions ----------------------------------------------------------

    def browse_patches(self) -> None:
        if self.bank.stats().is_empty:
            QMessageBox.information(self, "Browse", "The bank is empty.")
            return
        PatchBrowser(self.bank, self).exec()
        self.refresh()
        self.changed.emit()

    def export_bank(self) -> None:
        if self.bank.stats().is_empty:
            QMessageBox.information(self, "Export", "The bank is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Patch Bank", str(Path.home() / "patches.bank"),
            "Patch bank (*.bank);;All files (*)")
        if not path:
            return
        try:
            count = self.bank.export_bank(path)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export",
                                f"Wrote {count} patches to {Path(path).name}.")

    def import_bank(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Patch Bank", str(Path.home()),
            "Patch bank (*.bank);;JSON (*.json);;All files (*)")
        if not path:
            return

        replace = False
        if not self.bank.stats().is_empty:
            answer = QMessageBox.question(
                self, "Import Patch Bank",
                "Replace the current bank, or append to it?\n\n"
                "Yes = replace (existing patches are deleted)\n"
                "No = append",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                return
            replace = answer == QMessageBox.StandardButton.Yes

        try:
            count = self.bank.import_bank(path, replace=replace)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        self.refresh()
        self.changed.emit()
        QMessageBox.information(self, "Import",
                                f"Imported {count} patches from {Path(path).name}.")

    def clear_bank(self) -> None:
        stats = self.bank.stats()
        if stats.is_empty:
            return
        answer = QMessageBox.warning(
            self, "Clear Patch Bank",
            f"Delete all {stats.total} patches?\n\n"
            "This cannot be undone. Export first if you might want them back.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed = self.bank.clear()
        self.refresh()
        self.changed.emit()
        QMessageBox.information(self, "Clear", f"Deleted {removed} patches.")
