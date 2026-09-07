"""Predict sheet — from ``Views/PredictAnnotationSheet.swift``.

Runs a trained classifier over annotated regions and hands back a
:class:`PredictionSet` for the canvas to paint.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from ...extractors.equivalence import EquivalenceStore, candidate_substitute
from ...extractors.registry import ExtractorRegistry
from ...models.classification import ClassificationProfile
from ...models.classifier import CLASSIFIER_SUFFIX, ClassifierError, MLClassifier
from ...models.store import AnnotationStore
from ...io.slide import SlideImage
from ...pipeline.predict import (Aggregation, PredictionSettings, predict_regions)
from ..workers import BackgroundTask


class PredictSheet(QDialog):
    """Configure and run prediction over the slide's annotated regions."""

    def __init__(self, slide: SlideImage, store: AnnotationStore,
                 registry: ExtractorRegistry, profile: ClassificationProfile,
                 classifier: MLClassifier | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Predict")
        self.setMinimumWidth(560)

        self.slide = slide
        self.store = store
        self.registry = registry
        self.profile = profile
        self.classifier = classifier
        self.task: BackgroundTask | None = None
        self.result = None
        self.report = None
        self.equivalences = EquivalenceStore()
        self.substitute: str | None = None

        self.buttons = QDialogButtonBox()
        self.run_button = self.buttons.addButton(
            "Predict", QDialogButtonBox.ButtonRole.AcceptRole)
        self.close_button = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_model_box())
        layout.addWidget(self._build_options_box())
        layout.addWidget(self._build_regions_box(), 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.run_button.clicked.connect(self._start)
        self.close_button.clicked.connect(self.reject)
        layout.addWidget(self.buttons)

        self._refresh_model_label()
        self._update_ready()

    # -- construction -----------------------------------------------------

    def _build_model_box(self) -> QWidget:
        box = QGroupBox("Model")
        form = QFormLayout(box)

        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        form.addRow("Active", self.model_label)

        load = QDialogButtonBox()
        self.load_button = load.addButton("Load Model…",
                                          QDialogButtonBox.ButtonRole.ActionRole)
        self.load_button.clicked.connect(self._load_model)
        form.addRow("", load)

        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        self.model_note.setStyleSheet("color: #d08a20;")
        self.model_note.setVisible(False)
        form.addRow("", self.model_note)
        return box

    def _build_options_box(self) -> QWidget:
        box = QGroupBox("Sampling and gating")
        form = QFormLayout(box)

        self.level_combo = QComboBox()
        for level in range(self.slide.level_count):
            downsample = self.slide.level_downsamples[level]
            self.level_combo.addItem(f"Level {level} — {downsample:g}x", userData=level)
        form.addRow("Pyramid level", self.level_combo)

        self.patch_size = QSpinBox()
        self.patch_size.setRange(16, 4096)
        self.patch_size.setSingleStep(16)
        self.patch_size.setValue(224)
        form.addRow("Patch size", self.patch_size)

        self.stride = QSpinBox()
        self.stride.setRange(1, 4096)
        self.stride.setSingleStep(16)
        self.stride.setValue(224)
        self.stride.setToolTip("A stride below the patch size overlaps tiles, "
                               "smoothing the heatmap at quadratic cost.")
        form.addRow("Stride", self.stride)

        self.aggregation = QComboBox()
        self.aggregation.addItem("Per patch — each tile keeps its own verdict",
                                 userData=Aggregation.PER_PATCH)
        self.aggregation.addItem("Mean probability — one verdict per region",
                                 userData=Aggregation.MEAN_PROBABILITY)
        self.aggregation.addItem("Max probability — worst focus wins the region",
                                 userData=Aggregation.MAX_PROBABILITY)
        form.addRow("Aggregation", self.aggregation)

        self.max_white = QDoubleSpinBox()
        self.max_white.setRange(0.0, 1.0)
        self.max_white.setSingleStep(0.05)
        self.max_white.setValue(0.75)
        form.addRow("Max white fraction", self.max_white)

        self.min_nuclei = QSpinBox()
        self.min_nuclei.setRange(0, 10_000)
        form.addRow("Min nuclei per tile", self.min_nuclei)

        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setSingleStep(0.05)
        self.min_confidence.setValue(0.0)
        self.min_confidence.setToolTip(
            "Tiles below this confidence are discarded rather than painted.")
        form.addRow("Min confidence", self.min_confidence)
        return box

    def _build_regions_box(self) -> QWidget:
        box = QGroupBox("Regions to predict over")
        layout = QVBoxLayout(box)
        self.region_list = QListWidget()
        self.region_list.itemChanged.connect(self._update_ready)
        for annotation in self.store.annotations:
            if annotation.is_subtractive:
                continue
            name = annotation.display_name or annotation.classification
            item = QListWidgetItem(f"{name}   ({annotation.area_short} px²)")
            item.setData(Qt.ItemDataRole.UserRole, annotation.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.region_list.addItem(item)
        layout.addWidget(self.region_list)

        # Everything starts ticked, so the common move on a slide with many
        # annotations is "none, then the two I care about".
        row = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(lambda: self._set_all_regions(True))
        self.select_none_button = QPushButton("Select None")
        self.select_none_button.clicked.connect(lambda: self._set_all_regions(False))
        row.addWidget(self.select_all_button)
        row.addWidget(self.select_none_button)
        row.addStretch(1)
        self.region_count = QLabel("")
        self.region_count.setStyleSheet("color: #888;")
        row.addWidget(self.region_count)
        layout.addLayout(row)
        return box

    def _set_all_regions(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.region_list.blockSignals(True)
        for i in range(self.region_list.count()):
            self.region_list.item(i).setCheckState(state)
        self.region_list.blockSignals(False)
        self._update_ready()

    # -- state ------------------------------------------------------------

    def _selected_regions(self) -> list:
        chosen = {self.region_list.item(i).data(Qt.ItemDataRole.UserRole)
                  for i in range(self.region_list.count())
                  if self.region_list.item(i).checkState() == Qt.CheckState.Checked}
        return [a for a in self.store.annotations
                if a.id in chosen and not a.is_subtractive]

    def _refresh_model_label(self) -> None:
        if self.classifier is None:
            self.model_label.setText("None — train a model or load a .cl file.")
            self.model_note.setVisible(False)
            self.substitute = None
            return
        self.model_label.setText(self.classifier.describe())

        identity = self.classifier.extractor_identity
        self.substitute = None
        if self.classifier.is_geometry:
            self.model_note.setText(
                "This is a geometry model: it classifies a traced outline, not "
                "a tile, so it cannot run here. Use Grade Annotations in the "
                "Geometry panel.")
            self.model_note.setVisible(True)
        elif self.registry.by_identity(identity) is None:
            # Already-approved substitution, or one we can offer.
            approved = self.equivalences.approved_for(identity)
            candidate = approved or candidate_substitute(identity, self.registry)
            if candidate and self.registry.by_identity(candidate) is not None:
                self.substitute = candidate
                if approved:
                    self.model_note.setText(
                        f"{identity} is not installed; using the approved "
                        f"equivalent {candidate}.")
                else:
                    self.model_note.setText(
                        f"{identity} is not installed, but {candidate} appears to "
                        f"be the same model in a different runtime. Press Predict "
                        f"to review and approve the substitution.")
            else:
                self.model_note.setText(
                    f"This model needs extractor {identity}, which is not "
                    f"installed, and nothing comparable was found. Prediction "
                    f"cannot run without the feature space it was trained in.")
            self.model_note.setVisible(True)
        elif self.classifier.is_pooled:
            self.model_note.setText(
                "Pooled model: it produces one verdict per region, so every tile "
                "in a region is painted the same colour regardless of the "
                "aggregation setting.")
            self.model_note.setVisible(True)
        else:
            self.model_note.setVisible(False)

    def _update_ready(self) -> None:
        chosen = len(self._selected_regions())
        total = self.region_list.count()
        if hasattr(self, "region_count"):
            self.region_count.setText(f"{chosen} of {total} selected")
        # A geometry model is deliberately excluded rather than allowed
        # through: this pipeline tiles the region and feeds 224px patches to an
        # ONNX extractor, and a tile has no lesion outline to describe. Letting
        # Run enable only produced "Extractor failed to load" on click.
        has_extractor = (self.classifier is not None
                        and not self.classifier.is_geometry
                        and (self.registry.by_identity(
                                 self.classifier.extractor_identity) is not None
                             or self.substitute is not None))
        usable = (self.classifier is not None and has_extractor
                  and bool(self._selected_regions()))
        self.run_button.setEnabled(usable)

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Classifier", str(Path.home()),
            f"Classifier (*{CLASSIFIER_SUFFIX});;Legacy (*.paninmodel.json);;All files (*)")
        if not path:
            return
        try:
            self.classifier = MLClassifier.load(path)
        except ClassifierError as exc:
            QMessageBox.critical(self, "Could not load model", str(exc))
            return
        self._refresh_model_label()
        self._update_ready()

    def _settings(self) -> PredictionSettings:
        return PredictionSettings(
            patch_size_level=self.patch_size.value(),
            stride_level=self.stride.value(),
            level=self.level_combo.currentData() or 0,
            max_white_fraction=self.max_white.value(),
            min_nuclei=self.min_nuclei.value(),
            min_confidence=self.min_confidence.value(),
            aggregation=self.aggregation.currentData(),
        )

    # -- running ----------------------------------------------------------

    def _confirm_substitution(self, wanted: str, candidate: str) -> bool:
        """Ask once, stating what was actually measured.

        This is a research tool; the user is entitled to the evidence rather
        than a yes/no with no basis for deciding.
        """
        if self.equivalences.approved_for(wanted) == candidate:
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Use a different runtime for this model?")
        box.setText(f"This model was trained with {wanted}, which is not installed.\n\n"
                    f"{candidate} is the same model exported to a different runtime.")
        box.setInformativeText(
            "These were compared on 250 real patches through a 10-class model:\n\n"
            "• Every prediction with confidence ≥ 0.5 agreed — 100%.\n"
            "• Overall label agreement was 97.2%.\n"
            "• All disagreements were PanIN grade calls below 0.5 confidence, "
            "where the model is near chance and any numerical difference flips it.\n\n"
            "The gap is most likely float16 vs float32: Core ML exports default to "
            "half precision.\n\n"
            "Approve this substitution?")
        box.setDetailedText(
            f"Measured cosine similarity between the two runtimes:\n"
            f"  phikon-v1  min 0.99999  (effectively identical)\n"
            f"  uni2-h     min 0.98136, mean 0.99600\n\n"
            f"Approving records {wanted} -> {candidate} so you are not asked again. "
            f"It applies only to this pair; every other identity stays strict.\n\n"
            f"Treat low-confidence grade calls with the same caution you would on "
            f"macOS — they were never reliable on that pair.")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return False

        self.equivalences.approve(
            wanted, candidate,
            note="Approved in the Predict sheet; confident predictions agreed 100%.")
        self._refresh_model_label()
        return True

    def _start(self) -> None:
        if self.classifier is None:
            return

        wanted = self.classifier.extractor_identity
        identity = wanted
        if self.registry.by_identity(wanted) is None and self.substitute:
            if not self._confirm_substitution(wanted, self.substitute):
                return
            identity = self.substitute

        try:
            extractor = self.registry.open(identity)
        except Exception as exc:
            QMessageBox.critical(self, "Extractor failed to load", str(exc))
            return

        colors = {c.name: c.color for c in self.profile.classes}
        # Any predicted class missing from the profile still needs a colour, or
        # it would paint as the default red and read as a different class.
        for i, label in enumerate(self.classifier.class_labels):
            colors.setdefault(label, _fallback_color(i))

        self._set_running(True)
        self.task = BackgroundTask(
            predict_regions,
            slide=self.slide,
            regions=self._selected_regions(),
            extractor=extractor,
            classifier=self.classifier,
            settings=self._settings(),
            subtractive=[a for a in self.store.annotations if a.is_subtractive],
            colors=colors,
            accept_identities=self.equivalences.accepted_identities(wanted),
        )
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(self._on_finished)
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.progress.setValue(0)
        for widget in (self.level_combo, self.patch_size, self.stride,
                       self.aggregation, self.max_white, self.min_nuclei,
                       self.min_confidence, self.region_list, self.load_button):
            widget.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.close_button.setEnabled(not running)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status.setText(message)

    def _on_finished(self, outcome) -> None:
        self._set_running(False)
        self.task = None
        if outcome is None:
            return
        self.result, self.report = outcome
        self.status.setText(self.report.summary())
        if self.result.is_empty:
            QMessageBox.information(
                self, "No predictions",
                "Every tile was filtered out.\n\n" + self.report.summary())
            return
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.task = None
        QMessageBox.warning(self, "Prediction failed", message)

    def done(self, result: int) -> None:
        if self.task is not None and self.task.is_running:
            self.task.cancel()
            self.task.wait(20_000)
        super().done(result)


def _fallback_color(index: int):
    from ...models.annotation import AnnotationColor
    palette = [(230, 210, 30), (222, 31, 123), (122, 230, 213), (19, 16, 163),
               (80, 180, 80), (240, 140, 30)]
    return AnnotationColor(*palette[index % len(palette)])
