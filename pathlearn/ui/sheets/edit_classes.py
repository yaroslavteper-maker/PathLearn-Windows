"""Rename or pool a trained model's classes, and save it as a new model.

The use case: a model trained to tell PanIN-1a from 1b from 2 from 3, used on a
slide where the question is only "is this PanIN at all". Pooling the four into
one answers that question with the model you already have.

Nothing is refitted. The weights are untouched and the pooled probability is
the exact sum of its parts, so this cannot manufacture accuracy — but it does
stop counting a PanIN-2-called-PanIN-3 as an error, because a model reporting
only "PanIN" was never asked to make that call.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ...models.classifier import CLASSIFIER_SUFFIX, ClassifierError, MLClassifier


class EditClassesSheet(QDialog):
    """One row per trained class; type the name it should report as."""

    def __init__(self, model: MLClassifier, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Model Classes")
        self.resize(640, 520)
        self.model = model
        self.saved_path: Path | None = None
        self.result_model: MLClassifier | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Give two classes the same name to pool them. The pooled "
            "probability is the sum of theirs — nothing is retrained."))

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Trained class", "Reports as"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_edited)
        layout.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.merge_edit = QLineEdit()
        self.merge_edit.setPlaceholderText("name for the selected rows, e.g. PanIN")
        merge = QPushButton("Merge Selected")
        merge.setToolTip("Give every selected row the same name, pooling them.")
        merge.clicked.connect(self._merge_selected)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.reload)
        row.addWidget(self.merge_edit, 1)
        row.addWidget(merge)
        row.addWidget(reset)
        layout.addLayout(row)

        self.preview = QLabel()
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #888;")
        layout.addWidget(self.note)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save as New Model…")
        self.save_button.clicked.connect(self._save)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        buttons.addWidget(box)
        layout.addLayout(buttons)

        self.reload()

    # -- table ------------------------------------------------------------

    def reload(self) -> None:
        """Start again from the model's own class names."""
        labels = self.model.training_labels
        current = self.model.class_labels
        groups = self.model.class_groups or list(range(len(labels)))

        self.table.blockSignals(True)
        self.table.setRowCount(len(labels))
        for row, label in enumerate(labels):
            trained = QTableWidgetItem(label)
            trained.setFlags(trained.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, trained)
            reports = current[groups[row]] if groups[row] < len(current) else label
            self.table.setItem(row, 1, QTableWidgetItem(reports))
        self.table.blockSignals(False)
        self._refresh_preview()

    def mapping(self) -> dict[str, str]:
        """Trained label -> the name it should report as."""
        out: dict[str, str] = {}
        for row in range(self.table.rowCount()):
            trained = self.table.item(row, 0).text()
            reports = (self.table.item(row, 1).text() or "").strip()
            out[trained] = reports or trained
        return out

    def _on_edited(self, _item) -> None:
        self._refresh_preview()

    def _merge_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if len(rows) < 2:
            self.note.setText("Select two or more rows to merge them.")
            return
        name = (self.merge_edit.text() or "").strip()
        if not name:
            # A sensible default: the longest common prefix, so PanIN-1a and
            # PanIN-2 offer "PanIN" without being told.
            name = _common_prefix([self.table.item(r, 0).text() for r in rows]) or "Merged"
            self.merge_edit.setText(name)
        self.table.blockSignals(True)
        for row in rows:
            self.table.item(row, 1).setText(name)
        self.table.blockSignals(False)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        mapping = self.mapping()
        outputs: list[str] = []
        for label in self.model.training_labels:
            target = mapping[label]
            if target not in outputs:
                outputs.append(target)

        pooled = len(self.model.training_labels) - len(outputs)
        self.preview.setText(
            f"<b>{len(outputs)} output class(es):</b> {', '.join(outputs)}")
        self.save_button.setEnabled(bool(outputs))

        if len(outputs) == 1:
            # Worth saying plainly: a one-class model scores 100% and has
            # told you nothing, because there is no call left to get wrong.
            self.note.setStyleSheet("color: #d0762a;")
            self.note.setText(
                "Everything pools into one class. Such a model always answers "
                "the same thing and scores 100% — it cannot be wrong because "
                "there is nothing left to decide. Useful only to mark regions "
                "as detected; meaningless as an accuracy figure.")
        elif pooled > 0:
            self.note.setStyleSheet("color: #888;")
            self.note.setText(
                f"{pooled} class(es) pooled away. Scores are recomputed from "
                f"the confusion matrix: confusions inside a pooled group stop "
                f"being errors, so accuracy rises legitimately. The model is "
                f"not retrained.")
        elif outputs != self.model.training_labels:
            self.note.setStyleSheet("color: #888;")
            self.note.setText("Renamed only — the model behaves identically.")
        else:
            self.note.setStyleSheet("color: #888;")
            self.note.setText("No changes yet.")

    # -- saving -----------------------------------------------------------

    def build(self) -> MLClassifier:
        """The regrouped model. Split out so tests skip the file dialog."""
        return self.model.with_class_mapping(self.mapping())

    def _save(self) -> None:
        try:
            regrouped = self.build()
        except ClassifierError as exc:
            QMessageBox.critical(self, "Cannot regroup", str(exc))
            return

        suggestion = "-".join(regrouped.class_labels[:2]) or "model"
        default = Path.home() / f"{suggestion.lower()}{CLASSIFIER_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Regrouped Model", str(default),
            f"Classifier (*{CLASSIFIER_SUFFIX})")
        if not path:
            return
        try:
            regrouped.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.saved_path = Path(path)
        self.result_model = regrouped
        self.accept()


def _common_prefix(labels: list[str]) -> str:
    """Longest shared prefix, trimmed of trailing punctuation.

    ``PanIN-1a``/``PanIN-2`` gives ``PanIN`` rather than ``PanIN-``.
    """
    if not labels:
        return ""
    prefix = labels[0]
    for label in labels[1:]:
        while prefix and not label.startswith(prefix):
            prefix = prefix[:-1]
    return prefix.rstrip("-_ .")
