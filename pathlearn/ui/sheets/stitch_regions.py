"""Turn predicted tiles of one class into annotations you can analyse."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from ...models.prediction import PredictionSet
from ...models.store import AnnotationStore
from ...pipeline.stitch import StitchSettings, stitch_class
from ..workers import BackgroundTask

#: Wait this long after the last edit before recomputing. Spin boxes fire
#: on every step, and without a debounce a drag from 1 to 20 queues twenty
#: runs.
PREVIEW_DELAY_MS = 300


class StitchRegionsSheet(QDialog):
    """Pick a class, group its adjacent tiles, and add the result as annotations."""

    def __init__(self, predictions: PredictionSet, store: AnnotationStore,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stitch Predictions into Annotations")
        self.resize(600, 520)
        self.predictions = predictions
        self.store = store
        self.report = None
        self.added = 0
        self.task: BackgroundTask | None = None
        #: The settings the current report was produced with, so Create
        #: never adds annotations that do not match what is on screen.
        self._report_settings: StitchSettings | None = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(PREVIEW_DELAY_MS)
        self._debounce.timeout.connect(self._start_preview)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Adjacent tiles of the chosen class become one annotation. Tiles "
            "that do not touch become separate annotations of the same class."))
        layout.addWidget(self._build_options_box())

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.preview = QLabel()
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)

        self.caution = QLabel(
            "A stitched outline follows tile edges, so it is a staircase. The "
            "outline-shape descriptor reads that as high boundary complexity "
            "for reasons about the grid rather than the tissue — so do not mix "
            "stitched regions with hand traces in one training set without "
            "checking that first.")
        self.caution.setWordWrap(True)
        self.caution.setStyleSheet("color: #d0762a;")
        layout.addWidget(self.caution)
        layout.addStretch(1)

        row = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self._start_preview)
        self.create_button = QPushButton("Create Annotations")
        self.create_button.clicked.connect(self._create)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        row.addWidget(self.preview_button)
        row.addWidget(self.create_button)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)

        self._start_preview()

    # -- construction -----------------------------------------------------

    def _build_options_box(self) -> QWidget:
        box = QGroupBox("What to stitch")
        form = QFormLayout(box)

        self.class_combo = QComboBox()
        counts = self.predictions.counts
        for label in sorted(counts, key=lambda l: -counts[l]):
            self.class_combo.addItem(f"{label}   ({counts[label]} tiles)",
                                     userData=label)
        self.class_combo.currentIndexChanged.connect(self._schedule_preview)
        form.addRow("Class", self.class_combo)

        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setSingleStep(0.05)
        self.min_confidence.setValue(float(self.predictions.min_confidence))
        self.min_confidence.setToolTip(
            "Tiles below this are ignored. Raising it shrinks regions and can "
            "split one into several.")
        self.min_confidence.valueChanged.connect(self._schedule_preview)
        form.addRow("Min confidence", self.min_confidence)

        self.connectivity = QComboBox()
        self.connectivity.addItem("Edge-sharing only (4)", userData=4)
        self.connectivity.addItem("Also diagonal corners (8)", userData=8)
        self.connectivity.setToolTip(
            "Whether two tiles meeting only at a corner count as one region.")
        self.connectivity.currentIndexChanged.connect(self._schedule_preview)
        form.addRow("Adjacency", self.connectivity)

        self.min_cells = QSpinBox()
        self.min_cells.setRange(1, 1000)
        self.min_cells.setValue(2)
        self.min_cells.setToolTip("Regions smaller than this are discarded.")
        self.min_cells.valueChanged.connect(self._schedule_preview)
        form.addRow("Min tiles per region", self.min_cells)

        self.simplify = QDoubleSpinBox()
        self.simplify.setRange(0.0, 5.0)
        self.simplify.setSingleStep(0.1)
        self.simplify.setValue(0.0)
        self.simplify.setToolTip(
            "Smooths the staircase, in tile widths. 0 keeps the region exactly "
            "as the tiles define it. Above 0 the boundary moves: measured on an "
            "L-shaped region, 0.9 shaved the inner corner and lost 10% of the "
            "area.")
        self.simplify.valueChanged.connect(self._schedule_preview)
        form.addRow("Smooth staircase", self.simplify)
        return box

    # -- state ------------------------------------------------------------

    @property
    def label(self) -> str:
        return self.class_combo.currentData() or ""

    def settings(self) -> StitchSettings:
        return StitchSettings(min_confidence=self.min_confidence.value(),
                              connectivity=self.connectivity.currentData(),
                              min_cells=self.min_cells.value(),
                              simplify_tolerance=self.simplify.value())

    def _schedule_preview(self) -> None:
        """Restart the debounce so a spin-box drag computes once, not per step."""
        self.report = None
        self._report_settings = None
        self.create_button.setEnabled(False)
        self.preview.setText("…")
        self._debounce.start()

    # -- running ----------------------------------------------------------

    def _start_preview(self) -> None:
        """Stitch on a worker thread.

        On the UI thread this froze the window: a whole-slide prediction is
        tens of thousands of tiles, and even after the tracing fix that is far
        too much to do between keystrokes.
        """
        self._debounce.stop()
        if self.task is not None or not self.label:
            if not self.label:
                self.preview.setText("No predictions to stitch.")
                self.create_button.setEnabled(False)
            return

        wanted = self.settings()
        self._set_running(True)
        self.preview.setText("Stitching…")
        self.task = BackgroundTask(stitch_class, predictions=self.predictions,
                                   label=self.label, settings=wanted)
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(
            lambda report: self._on_finished(report, wanted))
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.preview.setText(message)

    def _on_finished(self, report, wanted: StitchSettings) -> None:
        self.task = None
        self._set_running(False)
        if report is None:
            return
        self.report = report
        self._report_settings = wanted
        self.preview.setText(report.summary())
        self.create_button.setEnabled(bool(report.annotations))
        self.create_button.setText(
            f"Create {len(report.annotations)} Annotation(s)"
            if report.annotations else "Create Annotations")

    def _on_failed(self, message: str) -> None:
        self.task = None
        self._set_running(False)
        self.report = None
        self._report_settings = None
        self.create_button.setEnabled(False)
        self.preview.setText(message)

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.progress.setRange(0, 0 if running else 1)
        self.stop_button.setEnabled(running)
        for widget in (self.preview_button, self.class_combo, self.min_confidence,
                       self.connectivity, self.min_cells, self.simplify):
            widget.setEnabled(not running)
        if running:
            self.create_button.setEnabled(False)

    def _stop(self) -> None:
        if self.task is not None and self.task.is_running:
            self.stop_button.setEnabled(False)
            self.preview.setText("Stopping…")
            self.task.cancel()

    def _create(self) -> None:
        # Only ever add what the preview actually showed. Without this a
        # changed setting mid-preview could add regions nobody saw.
        if (self.report is None or not self.report.annotations
                or self._report_settings != self.settings()):
            self._start_preview()
            return
        self.added = len(self.report.annotations)
        # add_batch saves the sidecar once rather than per annotation.
        self.store.add_batch(self.report.annotations)
        self.accept()

    def done(self, result: int) -> None:
        self._debounce.stop()
        if self.task is not None and self.task.is_running:
            self.task.cancel()
            self.task.wait(20_000)
        super().done(result)
