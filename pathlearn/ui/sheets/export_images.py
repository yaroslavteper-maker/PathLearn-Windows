"""Export the ticked annotations as JPEGs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ...io.slide import SlideImage
from ...models.store import AnnotationStore
from ...pipeline.export_images import (DEFAULT_MAX_EDGE, ExportSettings,
                                       export_annotation_images)
from ..workers import BackgroundTask


class ExportImagesSheet(QDialog):
    """Choose where and how, then write one JPEG per annotation."""

    def __init__(self, slide: SlideImage, store: AnnotationStore,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Annotations as JPEG")
        self.resize(620, 520)
        self.slide = slide
        self.store = store
        self.task: BackgroundTask | None = None
        self.report = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_destination_box())
        layout.addWidget(self._build_options_box())

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #bbb;")
        layout.addWidget(self.status)
        layout.addStretch(1)

        row = QHBoxLayout()
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._export)
        row.addWidget(self.export_button)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)

        self._update_summary()

    # -- construction -----------------------------------------------------

    def _build_destination_box(self) -> QWidget:
        box = QGroupBox("Destination")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.folder_edit = QLineEdit(str(self._default_folder()))
        self.folder_edit.textChanged.connect(self._update_summary)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)

        self.folder_per_class = QCheckBox("One subfolder per class")
        self.folder_per_class.setChecked(True)
        self.folder_per_class.setToolTip(
            "class/image.jpg — the layout most image classifiers expect, and "
            "usually the reason for exporting at all.")
        self.folder_per_class.toggled.connect(self._update_summary)
        layout.addWidget(self.folder_per_class)
        return box

    def _build_options_box(self) -> QWidget:
        box = QGroupBox("Images")
        form = QFormLayout(box)

        self.level_combo = QComboBox()
        self.level_combo.addItem("Automatic — finest that fits the size cap",
                                 userData=None)
        for level, downsample in enumerate(self.slide.level_downsamples):
            self.level_combo.addItem(f"Level {level} — {downsample:g}x", userData=level)
        self.level_combo.setToolTip(
            "Automatic keeps detail for small regions and avoids enormous "
            "files for large ones.")
        self.level_combo.currentIndexChanged.connect(self._update_summary)
        form.addRow("Resolution", self.level_combo)

        self.max_edge = QSpinBox()
        self.max_edge.setRange(64, 20_000)
        self.max_edge.setSingleStep(256)
        self.max_edge.setValue(DEFAULT_MAX_EDGE)
        self.max_edge.setToolTip(
            "Longest side of the written image. Anything larger is downscaled.")
        self.max_edge.valueChanged.connect(self._update_summary)
        form.addRow("Max edge (px)", self.max_edge)

        self.margin = QSpinBox()
        self.margin.setRange(0, 4096)
        self.margin.setSingleStep(32)
        self.margin.setToolTip(
            "Extra slide around the region, in level-0 pixels. Context often "
            "makes a duct readable that is ambiguous cropped tight.")
        form.addRow("Margin (px)", self.margin)

        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(92)
        form.addRow("JPEG quality", self.quality)

        self.draw_outline = QCheckBox("Draw the outline in the class colour")
        form.addRow("", self.draw_outline)

        self.mask_outside = QCheckBox("Blank everything outside the outline")
        self.mask_outside.setToolTip(
            "Fills the surrounding tissue with white. Off by default — the "
            "context is usually what makes the region interpretable.")
        form.addRow("", self.mask_outside)
        return box

    # -- state ------------------------------------------------------------

    def _default_folder(self) -> Path:
        return Path(self.slide.path).parent / f"{Path(self.slide.name).stem}-annotations"

    def _annotations(self):
        return self.store.in_use()

    def settings(self) -> ExportSettings:
        return ExportSettings(level=self.level_combo.currentData(),
                              margin=self.margin.value(),
                              max_edge=self.max_edge.value(),
                              quality=self.quality.value(),
                              mask_outside=self.mask_outside.isChecked(),
                              draw_outline=self.draw_outline.isChecked(),
                              folder_per_class=self.folder_per_class.isChecked())

    def _update_summary(self) -> None:
        annotations = self._annotations()
        classes = {a.classification for a in annotations}
        running = self.task is not None
        self.export_button.setEnabled(bool(annotations)
                                      and bool(self.folder_edit.text().strip())
                                      and not running)
        self.export_button.setText(f"Export {len(annotations)} Image(s)"
                                   if annotations else "Export")
        skipped = len(self.store.annotations) - len(annotations)
        text = (f"{len(annotations)} checked annotation(s) across "
                f"{len(classes)} class(es)")
        if skipped:
            text += f"; {skipped} unchecked and will not be exported"
        self.status.setStyleSheet("color: #bbb;")
        self.status.setText(text + ".")

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Export images into",
                                                self.folder_edit.text()
                                                or str(Path.home()))
        if path:
            self.folder_edit.setText(path)

    # -- running ----------------------------------------------------------

    def _export(self) -> None:
        annotations = self._annotations()
        if not annotations or self.task is not None:
            return
        folder = Path(self.folder_edit.text().strip())
        existing = list(folder.rglob("*.jpg")) if folder.is_dir() else []
        if existing and QMessageBox.question(
            self, "Export Images",
            f"{folder} already holds {len(existing)} JPEG(s).\n\n"
            "Existing files are never overwritten — new ones are numbered "
            "alongside them. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return

        self._set_running(True)
        self.task = BackgroundTask(export_annotation_images, slide=self.slide,
                                   annotations=annotations, out_dir=folder,
                                   settings=self.settings())
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(self._on_finished)
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.progress.setValue(0)
        for widget in (self.export_button, self.level_combo, self.max_edge,
                       self.margin, self.quality, self.draw_outline,
                       self.mask_outside, self.folder_per_class):
            widget.setEnabled(not running)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status.setText(message)

    def _on_finished(self, report) -> None:
        self.task = None
        self._set_running(False)
        if report is None:
            return
        self.report = report
        self._update_summary()
        self.status.setText(report.summary())
        if report.skipped:
            # Inline, not a modal: these are explanations, and there may be
            # many of them.
            first = report.skipped[:3]
            self.status.setText(
                report.summary() + "  " +
                "; ".join(f"{name}: {reason}" for name, reason in first))

    def _on_failed(self, message: str) -> None:
        self.task = None
        self._set_running(False)
        self._update_summary()
        self.status.setText(message)
        self.status.setStyleSheet("color: #d04a20;")

    def done(self, result: int) -> None:
        if self.task is not None and self.task.is_running:
            self.task.cancel()
            self.task.wait(20_000)
        super().done(result)
