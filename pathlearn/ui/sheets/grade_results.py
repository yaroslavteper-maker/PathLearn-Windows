"""Per-annotation verdicts from a geometry model, and applying them.

Separate from the geometry panel because a verdict table needs room: one row
per annotation, with every class probability available, sortable by confidence
so the calls worth checking are the ones you look at first.

Applying is deliberately a second, explicit step. The model rewrites the class
of hand-traced annotations — the ground truth — so it must never happen as a
side effect of asking what the model thinks.
"""

from __future__ import annotations

import uuid

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ...models.classification import ClassificationProfile
from ...models.store import AnnotationStore
from ...pipeline.geometry_predict import Grade, GradeReport

#: Below this the call is closer to a coin-flip than a verdict.
LOW_CONFIDENCE = 0.5

COLUMNS = ["Annotation", "Annotated as", "Predicted", "Confidence", "Agrees"]


class GradeResultsSheet(QDialog):
    """Shows one row per annotation and can write the verdicts back."""

    def __init__(self, report: GradeReport, store: AnnotationStore,
                 profile: ClassificationProfile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Geometry Grades")
        self.resize(760, 560)
        self.report = report
        self.store = store
        self.profile = profile
        self.applied = 0

        layout = QVBoxLayout(self)

        self.summary = QLabel(report.summary())
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        caution = QLabel(
            "Agreement here is not accuracy: these annotations were most "
            "likely in the model's own training set, so this measures "
            "memorisation. The honest number is the leave-one-slide-out score "
            "from training.")
        caution.setWordWrap(True)
        caution.setStyleSheet("color: #b8860b;")
        layout.addWidget(caution)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        self.changes_only = QCheckBox("Show only verdicts that differ from the label")
        self.changes_only.toggled.connect(self.reload)
        layout.addWidget(self.changes_only)

        row = QHBoxLayout()
        self.apply_button = QPushButton("Apply Predicted Classes…")
        self.apply_button.setToolTip(
            "Reclassify each annotation to the model's verdict. This overwrites "
            "your tracing's class and saves the sidecar.")
        self.apply_button.clicked.connect(self._apply)
        row.addWidget(self.apply_button)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)

        self.reload()

    # -- table ------------------------------------------------------------

    def _rows(self) -> list[Grade]:
        if self.changes_only.isChecked():
            return [g for g in self.report.grades if g.changed]
        return list(self.report.grades)

    def reload(self) -> None:
        rows = self._rows()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for index, grade in enumerate(rows):
            self._fill_row(index, grade)
        self.table.setSortingEnabled(True)
        changed = sum(1 for g in self.report.grades if g.changed)
        self.apply_button.setEnabled(changed > 0)
        self.apply_button.setText(
            f"Apply Predicted Classes… ({changed})" if changed
            else "Apply Predicted Classes…")

    def _fill_row(self, index: int, grade: Grade) -> None:
        name = QTableWidgetItem(grade.display_name)
        name.setData(Qt.ItemDataRole.UserRole, grade.annotation_id)
        if grade.probabilities:
            name.setToolTip("\n".join(
                f"{label}: {p:.3f}" for label, p
                in sorted(grade.probabilities.items(), key=lambda kv: -kv[1])))
        self.table.setItem(index, 0, name)
        self.table.setItem(index, 1, QTableWidgetItem(grade.annotated_as))

        if grade.predicted is None:
            skipped = QTableWidgetItem("—")
            skipped.setToolTip(grade.skipped or "")
            self.table.setItem(index, 2, skipped)
            reason = QTableWidgetItem(grade.skipped or "not described")
            reason.setForeground(QColor("#888"))
            self.table.setItem(index, 3, reason)
            self.table.setItem(index, 4, QTableWidgetItem(""))
            return

        predicted = QTableWidgetItem(grade.predicted)
        if grade.changed:
            predicted.setForeground(QColor("#d04a20"))
        self.table.setItem(index, 2, predicted)

        # Numeric sort, not lexicographic — "0.9" must not sort below "0.85".
        confidence = QTableWidgetItem()
        confidence.setData(Qt.ItemDataRole.DisplayRole, round(grade.confidence, 3))
        if grade.confidence < LOW_CONFIDENCE:
            confidence.setForeground(QColor("#d04a20"))
            confidence.setToolTip(
                f"Below {LOW_CONFIDENCE:g}: closer to a coin-flip than a verdict.")
        self.table.setItem(index, 3, confidence)
        self.table.setItem(index, 4, QTableWidgetItem("yes" if grade.agrees else "no"))

    # -- applying ---------------------------------------------------------

    def _apply(self) -> None:
        changes = [g for g in self.report.grades if g.changed]
        if not changes:
            return
        if QMessageBox.warning(
            self, "Apply Predicted Classes",
            f"Reclassify {len(changes)} annotation(s) to the model's verdict?\n\n"
            "This overwrites the class you traced them as. There is no undo — "
            "the sidecar is saved immediately.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.applied = self.apply_now(changes)
        self.summary.setText(
            f"Applied {self.applied} verdict(s). " + self.report.summary())
        self.reload()

    def apply_now(self, changes: list[Grade]) -> int:
        """Write the verdicts to the store. Split out so tests skip the dialog."""
        applied = 0
        for grade in changes:
            annotation = self.store.by_id(grade.annotation_id)
            if annotation is None or grade.predicted is None:
                continue
            cls = self.profile.by_name(grade.predicted)
            # A predicted class the profile has never heard of keeps the
            # annotation's existing colour rather than defaulting to red, which
            # would read on the canvas as some other, wrong class.
            color = cls.color if cls is not None else annotation.color
            self.store.set_classification(grade.annotation_id, grade.predicted, color)
            applied += 1
        return applied

    def selected_annotation_id(self) -> uuid.UUID | None:
        items = self.table.selectedItems()
        if not items:
            return None
        return self.table.item(items[0].row(), 0).data(Qt.ItemDataRole.UserRole)
