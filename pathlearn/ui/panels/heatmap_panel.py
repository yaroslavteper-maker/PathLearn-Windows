"""Heatmap controls — per-class visibility and a confidence threshold.

From the rendering half of ``Views/MLModelPanel.swift``.  The threshold is the
useful control here: a per-patch run paints every tile it kept, and raising the
floor is how you tell a confident call from a marginal one without re-running.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from ...models.prediction import PredictionSet, sidecar_path


class HeatmapPanel(QWidget):
    """Controls the prediction overlay currently on the canvas."""

    changed = Signal()
    #: Asks the window to open the stitch sheet — the panel has the
    #: predictions but not the annotation store.
    stitch_requested = Signal()
    #: Asks the window for the composition sheet — the pixel size lives on
    #: the slide, which the panel does not hold.
    composition_requested = Signal()
    #: The panel owns the checkbox; the window owns the file and the setting.
    autosave_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.predictions: PredictionSet | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.headline = QLabel("No predictions.")
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.headline)

        self.autosave_checkbox = QCheckBox("Save heatmap beside the slide")
        self.autosave_checkbox.setChecked(True)
        self.autosave_checkbox.setToolTip(
            "Keep the heatmap in <slide>.predictions.json.gz next to the "
            "slide, and load it again when the slide is reopened. Turning "
            "this off stops new saves; it does not delete a file already "
            "written.")
        self.autosave_checkbox.toggled.connect(self.autosave_toggled.emit)

        self.show_checkbox = QCheckBox("Show heatmap")
        self.show_checkbox.setChecked(True)
        self.show_checkbox.toggled.connect(lambda _: self.changed.emit())
        layout.addWidget(self.show_checkbox)
        layout.addWidget(self.autosave_checkbox)

        layout.addWidget(QLabel("Classes:"))
        self.class_list = QListWidget()
        self.class_list.itemChanged.connect(self._on_class_toggled)
        layout.addWidget(self.class_list, 1)

        self.threshold_label = QLabel("Minimum confidence: 0.00")
        layout.addWidget(self.threshold_label)
        self.threshold = QSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(0, 100)
        self.threshold.setValue(0)
        self.threshold.valueChanged.connect(self._on_threshold)
        layout.addWidget(self.threshold)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #bbb;")
        layout.addWidget(self.detail)

        self.stitch_button = QPushButton("Stitch to Annotations…")
        self.stitch_button.setEnabled(False)
        self.stitch_button.setToolTip(
            "Join adjacent predicted tiles of one class into annotations, "
            "so they can be described by the geometry pipeline. Tiles that "
            "do not touch become separate annotations of the same class.")
        self.stitch_button.clicked.connect(self.stitch_requested.emit)
        layout.addWidget(self.stitch_button)

        self.composition_button = QPushButton("Tissue Composition…")
        self.composition_button.setEnabled(False)
        self.composition_button.setToolTip(
            "Break the predicted tissue down by class, as a percentage and an "
            "area. The denominator is the tissue that was predicted over, not "
            "the whole slide.")
        self.composition_button.clicked.connect(self.composition_requested.emit)
        layout.addWidget(self.composition_button)

        buttons = QHBoxLayout()
        for text, slot in (("Save…", self.save), ("Load…", self.load),
                           ("Clear", self.clear)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

    # -- state ------------------------------------------------------------

    @property
    def show_heatmap(self) -> bool:
        return self.show_checkbox.isChecked()

    @property
    def autosave(self) -> bool:
        return self.autosave_checkbox.isChecked()

    def set_autosave(self, enabled: bool) -> None:
        """Set the box without re-emitting — this restores a stored setting."""
        self.autosave_checkbox.blockSignals(True)
        self.autosave_checkbox.setChecked(enabled)
        self.autosave_checkbox.blockSignals(False)

    def set_predictions(self, predictions: PredictionSet | None) -> None:
        self.predictions = predictions
        usable = predictions is not None and not predictions.is_empty
        self.stitch_button.setEnabled(usable)
        self.composition_button.setEnabled(usable)
        self.class_list.blockSignals(True)
        self.class_list.clear()

        if predictions is None or predictions.is_empty:
            self.headline.setText("No predictions.")
            self.detail.setText("")
            self.class_list.blockSignals(False)
            return

        counts = predictions.counts
        for label in sorted(counts):
            color = predictions.color_for(label)
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(color.r, color.g, color.b))
            item = QListWidgetItem(QIcon(pixmap), f"{label}   ({counts[label]} tiles)")
            item.setData(Qt.ItemDataRole.UserRole, label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked if label in predictions.hidden
                               else Qt.CheckState.Checked)
            self.class_list.addItem(item)

        self.class_list.blockSignals(False)
        self.threshold.setValue(int(predictions.min_confidence * 100))
        self._refresh_text()

    def _refresh_text(self) -> None:
        if self.predictions is None or self.predictions.is_empty:
            return
        self.headline.setText(
            f"{len(self.predictions)} tiles · {len(self.predictions.counts)} classes")
        self.detail.setText(self.predictions.summary())

    def _on_class_toggled(self, item: QListWidgetItem) -> None:
        if self.predictions is None:
            return
        label = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.predictions.hidden.discard(label)
        else:
            self.predictions.hidden.add(label)
        self._refresh_text()
        self.changed.emit()

    def _on_threshold(self, value: int) -> None:
        self.threshold_label.setText(f"Minimum confidence: {value / 100:.2f}")
        if self.predictions is not None:
            self.predictions.min_confidence = value / 100.0
            self._refresh_text()
            self.changed.emit()

    # -- actions ----------------------------------------------------------

    def clear(self) -> None:
        self.set_predictions(None)
        self.changed.emit()

    def save(self) -> None:
        if self.predictions is None or self.predictions.is_empty:
            return
        default = (sidecar_path(self.predictions.slide_path)
                   if self.predictions.slide_path else Path.home() / "predictions.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Predictions", str(default), "Predictions (*.json)")
        if not path:
            return
        try:
            self.predictions.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.detail.setText(f"Saved {Path(path).name}.")

    def load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Predictions", str(Path.home()), "Predictions (*.json)")
        if not path:
            return
        try:
            predictions = PredictionSet.load(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.set_predictions(predictions)
        self.changed.emit()
