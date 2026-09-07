"""Extract Patches sheet — from ``Views/ExtractPatchesSheet.swift``.

Chooses an extractor and the sampling geometry, previews how many patches that
implies, then runs extraction on a background thread with progress and cancel.

The live count preview matters more than it looks: patch count scales with the
*square* of stride, so halving the stride quadruples the work. Showing the real
in-polygon count (not a bounding-box estimate) before committing is the
difference between a 3-second run and a 40-minute one.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from ...data.bank import PatchBank
from ...extractors.registry import ExtractorRegistry
from ...io.slide import SlideImage
from ...models.store import AnnotationStore
from ...pipeline.extract import (ExtractionSettings, estimate_patch_count,
                                 extract_annotations)
from ..workers import BackgroundTask

#: Rough per-patch wall-clock by feature dim, measured on this machine (CUDA).
#: Only used to warn about long runs, so being approximate is fine.
_MS_PER_PATCH = {768: 10.0, 1024: 18.0, 1536: 35.0}
_DEFAULT_MS_PER_PATCH = 20.0


class ExtractPatchesSheet(QDialog):
    """Configure and run patch extraction for the open slide."""

    def __init__(self, slide: SlideImage, store: AnnotationStore,
                 registry: ExtractorRegistry, bank: PatchBank,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extract Patches")
        self.setMinimumWidth(560)

        self.slide = slide
        self.store = store
        self.registry = registry
        self.bank = bank
        self.task: BackgroundTask | None = None
        self.report = None

        # The button box is created first: populating the extractor combo fires
        # currentIndexChanged, whose handler enables/disables the Extract button.
        self.buttons = QDialogButtonBox()
        self.extract_button = self.buttons.addButton(
            "Extract", QDialogButtonBox.ButtonRole.AcceptRole)
        self.close_button = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_extractor_box())
        layout.addWidget(self._build_geometry_box())
        layout.addWidget(self._build_filter_box())
        layout.addWidget(self._build_classes_box(), 1)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #bbb;")
        layout.addWidget(self.summary)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.extract_button.clicked.connect(self._start)
        self.close_button.clicked.connect(self.reject)
        layout.addWidget(self.buttons)

        # Recomputing the exact count touches every polygon, so coalesce edits.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._update_preview)

        self._on_extractor_changed()
        self._schedule_preview()

    # -- construction -----------------------------------------------------

    def _build_extractor_box(self) -> QWidget:
        box = QGroupBox("Feature extractor")
        form = QFormLayout(box)

        self.extractor_combo = QComboBox()
        for descriptor in self.registry:
            self.extractor_combo.addItem(
                f"{descriptor.identity}   {descriptor.feature_dim}-d   "
                f"{descriptor.input_width}x{descriptor.input_height}",
                userData=str(descriptor.identity))
        self.extractor_combo.currentIndexChanged.connect(self._on_extractor_changed)
        form.addRow("Model", self.extractor_combo)

        self.extractor_note = QLabel()
        self.extractor_note.setWordWrap(True)
        self.extractor_note.setStyleSheet("color: #999;")
        form.addRow("", self.extractor_note)

        if self.registry.is_empty:
            self.extractor_combo.addItem("No extractors installed", userData=None)
            self.extractor_combo.setEnabled(False)
        return box

    def _build_geometry_box(self) -> QWidget:
        box = QGroupBox("Sampling")
        form = QFormLayout(box)

        self.level_combo = QComboBox()
        for level in range(self.slide.level_count):
            downsample = self.slide.level_downsamples[level]
            dims = self.slide.level_dimensions[level]
            mpp = (self.slide.mpp_x or 0) * downsample
            label = (f"Level {level}  —  {downsample:g}x  "
                     f"({dims.width}x{dims.height}"
                     + (f", {mpp:.2f} um/px)" if mpp else ")"))
            self.level_combo.addItem(label, userData=level)
        self.level_combo.currentIndexChanged.connect(self._schedule_preview)
        form.addRow("Pyramid level", self.level_combo)

        self.patch_size = QSpinBox()
        self.patch_size.setRange(16, 4096)
        self.patch_size.setSingleStep(16)
        self.patch_size.valueChanged.connect(self._schedule_preview)
        form.addRow("Patch size (px at level)", self.patch_size)

        self.stride = QSpinBox()
        self.stride.setRange(1, 4096)
        self.stride.setSingleStep(16)
        self.stride.valueChanged.connect(self._schedule_preview)
        form.addRow("Stride (px at level)", self.stride)

        self.overlap_note = QLabel()
        self.overlap_note.setStyleSheet("color: #999;")
        form.addRow("", self.overlap_note)
        return box

    def _build_filter_box(self) -> QWidget:
        box = QGroupBox("Filters")
        form = QFormLayout(box)

        self.max_white = QDoubleSpinBox()
        self.max_white.setRange(0.0, 1.0)
        self.max_white.setSingleStep(0.05)
        self.max_white.setDecimals(2)
        self.max_white.setValue(0.75)
        self.max_white.setToolTip(
            "Reject a patch when more than this fraction of its pixels have "
            "R, G and B all >= 220 — i.e. slide background rather than tissue.")
        form.addRow("Max white fraction", self.max_white)

        self.min_nuclei = QSpinBox()
        self.min_nuclei.setRange(0, 10_000)
        self.min_nuclei.setToolTip(
            "0 disables nucleus counting. Above 0, each surviving patch is "
            "stain-deconvolved and segmented, which is noticeably slower.")
        self.min_nuclei.valueChanged.connect(self._update_summary)
        form.addRow("Min nuclei per patch", self.min_nuclei)
        return box

    def _build_classes_box(self) -> QWidget:
        box = QGroupBox("Annotations to extract")
        layout = QVBoxLayout(box)

        self.class_list = QListWidget()
        self.class_list.itemChanged.connect(self._schedule_preview)
        # Only checked ("Use") annotations are offered — unchecked ones are
        # excluded from extraction, so listing them would be misleading.
        counts: dict[str, int] = {}
        for annotation in self.store.in_use():
            counts[annotation.classification] = counts.get(annotation.classification, 0) + 1
        unchecked = sum(1 for a in self.store.annotations
                        if not a.is_selected and not a.is_subtractive)
        for name, count in sorted(counts.items()):
            item = QListWidgetItem(f"{name}   ({count} region{'s' if count != 1 else ''})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.class_list.addItem(item)
        layout.addWidget(self.class_list)

        row = QHBoxLayout()
        for text, state in (("All", Qt.CheckState.Checked),
                            ("None", Qt.CheckState.Unchecked)):
            button = QPushButton(text)
            button.clicked.connect(lambda _, s=state: self._set_all(s))
            row.addWidget(button)
        row.addStretch(1)

        if unchecked:
            note = QLabel(f"{unchecked} unchecked annotation(s) excluded — "
                          f"tick them in the Annotations panel to include them.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #d08a20;")
            layout.addWidget(note)

        subtractive_count = len(self.store.subtractive_in_use)
        plural = "s" if subtractive_count != 1 else ""
        self.use_subtractive = QCheckBox(
            f"Carve out {subtractive_count} subtractive region{plural}")
        self.use_subtractive.setChecked(True)
        self.use_subtractive.toggled.connect(self._schedule_preview)
        # Hidden rather than disabled when the slide has none: a greyed-out
        # "0 regions" is just noise on the majority of slides.
        self.use_subtractive.setVisible(subtractive_count > 0)
        row.addWidget(self.use_subtractive)
        layout.addLayout(row)
        return box

    def _set_all(self, state: Qt.CheckState) -> None:
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(state)

    # -- state ------------------------------------------------------------

    @property
    def selected_classes(self) -> set[str]:
        return {self.class_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.class_list.count())
                if self.class_list.item(i).checkState() == Qt.CheckState.Checked}

    @property
    def current_descriptor(self):
        identity = self.extractor_combo.currentData()
        return self.registry.by_identity(identity) if identity else None

    def _annotations(self) -> list:
        chosen = self.selected_classes
        return [a for a in self.store.in_use() if a.classification in chosen]

    def _subtractive(self) -> list:
        if not self.use_subtractive.isChecked():
            return []
        return self.store.subtractive_in_use

    def _settings(self) -> ExtractionSettings:
        return ExtractionSettings(
            patch_size_level=self.patch_size.value(),
            stride_level=self.stride.value(),
            level=self.level_combo.currentData() or 0,
            max_white_fraction=self.max_white.value(),
            min_nuclei=self.min_nuclei.value(),
        )

    def _on_extractor_changed(self) -> None:
        descriptor = self.current_descriptor
        if descriptor is None:
            self.extractor_note.setText(
                "Install a model and its .pathlearn-extractor.json in "
                f"{Path.home() / 'AppData' / 'Local' / 'PathLearn' / 'Extractors'}")
            self.extract_button.setEnabled(False)
            return

        # Default the patch size to the model's input so nothing is resized:
        # a silent resize changes the feature space the bank records.
        self.patch_size.setValue(descriptor.input_width)
        self.stride.setValue(descriptor.input_width)
        self.extractor_note.setText(
            f"{descriptor.notes or ''}  Patches are sampled at "
            f"{descriptor.input_width}x{descriptor.input_height} to match the model; "
            f"a different size is resized before inference.")
        self._schedule_preview()

    # -- preview ----------------------------------------------------------

    def _schedule_preview(self) -> None:
        self._preview_timer.start()
        self._update_summary()

    def _update_preview(self) -> None:
        annotations = self._annotations()
        if not annotations or self.current_descriptor is None:
            self._estimated = 0
        else:
            try:
                self._estimated = estimate_patch_count(self.slide, annotations,
                                                       self._settings())
            except ValueError:
                self._estimated = 0
        self._update_summary()

    def _update_summary(self) -> None:
        estimated = getattr(self, "_estimated", 0)
        descriptor = self.current_descriptor
        self.extract_button.setEnabled(bool(estimated) and descriptor is not None)

        if descriptor is None:
            self.summary.setText("No extractor selected.")
            return
        if not self._annotations():
            self.summary.setText("No annotations selected.")
            return

        ms = _MS_PER_PATCH.get(descriptor.feature_dim, _DEFAULT_MS_PER_PATCH)
        if self.min_nuclei.value() > 0:
            ms += 25.0     # deconvolution + connected components per patch
        seconds = estimated * ms / 1000.0
        duration = (f"{seconds:.0f}s" if seconds < 90
                    else f"{seconds / 60:.1f} min")

        overlap = ""
        if self.stride.value() < self.patch_size.value():
            factor = (self.patch_size.value() / self.stride.value()) ** 2
            overlap = f"  Overlapping — about {factor:.1f}x more patches than a full stride."
        self.overlap_note.setText(overlap.strip())

        already = ""
        if self.bank.is_open:
            existing = self.bank.count(extractor_identity=str(descriptor.identity),
                                       slide_path=str(self.slide.path))
            if existing:
                already = (f"  This slide already has {existing} patches for this "
                           f"extractor; re-extracting adds more.")

        self.summary.setText(
            f"<b>{estimated} patches</b> before filtering, roughly {duration}.{overlap}{already}")

    # -- running ----------------------------------------------------------

    def _start(self) -> None:
        descriptor = self.current_descriptor
        annotations = self._annotations()
        if descriptor is None or not annotations:
            return

        stats = self.bank.stats()
        other = set(stats.by_extractor) - {str(descriptor.identity)}
        if other:
            answer = QMessageBox.question(
                self, "Mixed extractors",
                f"The bank already holds patches from {', '.join(sorted(other))}.\n\n"
                "Feature spaces cannot be mixed when training — a model is trained "
                "against exactly one extractor. Adding these is fine, but training "
                "will make you pick one.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return

        try:
            extractor = self.registry.open(descriptor.identity,
                                           batch_size=self._settings().batch_size)
        except Exception as exc:
            QMessageBox.critical(self, "Extractor failed to load", str(exc))
            return

        if extractor.provider == "CPUExecutionProvider":
            answer = QMessageBox.question(
                self, "Running on CPU",
                "The GPU execution provider is not active, so extraction will run "
                "on the CPU — roughly 20 times slower.\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._set_running(True)
        self.task = BackgroundTask(
            extract_annotations,
            slide=self.slide,
            annotations=annotations,
            extractor=extractor,
            bank=self.bank,
            settings=self._settings(),
            subtractive=self._subtractive(),
        )
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(self._on_finished)
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.progress.setValue(0)
        for widget in (self.extractor_combo, self.level_combo, self.patch_size,
                       self.stride, self.max_white, self.min_nuclei,
                       self.class_list, self.use_subtractive):
            widget.setEnabled(not running)
        self.extract_button.setText("Cancel" if running else "Extract")
        self.close_button.setEnabled(not running)
        try:
            self.extract_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.extract_button.clicked.connect(self._cancel if running else self._start)
        self.extract_button.setEnabled(True)

    def _cancel(self) -> None:
        if self.task is not None:
            self.task.cancel()
            self.status.setText("Cancelling after the current batch…")
            self.extract_button.setEnabled(False)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status.setText(message)

    def _on_finished(self, report) -> None:
        self._set_running(False)
        self.task = None
        if report is None:
            return
        self.report = report
        self.status.setText(report.summary())
        self._update_summary()

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.task = None
        QMessageBox.critical(self, "Extraction failed", message)

    def done(self, result: int) -> None:
        """Shut down cleanly however the dialog is dismissed.

        The debounced preview timer must be stopped: if it fires after the
        window is gone it queries a bank the owner may already have closed.
        Any in-flight extraction is cancelled and waited for, so the worker
        thread never outlives the dialog it reports to.
        """
        self._preview_timer.stop()
        if self.task is not None and self.task.is_running:
            self._cancel()
            self.task.wait(10_000)
        super().done(result)
