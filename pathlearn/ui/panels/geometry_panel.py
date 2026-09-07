"""Geometry panel — describe annotations, then train a grade model on them.

From ``Views/GeometryBankPanel.swift``, extended with the outline-shape source.

Two descriptor blocks are offered because they fail in different ways:

* **Outline shape** needs only the polygon, so it works on every annotation and
  is unaffected by staining or resolution.
* **Interior texture** needs enough resolvable nuclei, so small annotations —
  single duct cross-sections, which is most of this dataset — cannot supply it.

The panel shows how many records carry each block, so choosing a source that the
bank cannot support is visible before training rather than an error after.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QProgressBar, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core.geometry import GeometryConfig
from ...data.geometry_bank import FeatureSource, GeometryBank
from ...models.classifier import CLASSIFIER_SUFFIX, MLClassifier
from ...pipeline.geometry_predict import grade_annotations, source_of
from ...pipeline.geometry_train import (GeometryTrainingSettings, Validation,
                                        describe_annotations,
                                        train_geometry_classifier)
from ..workers import BackgroundTask


class GeometryPanel(QWidget):
    """Compute descriptors for the open slide, then train and score a model."""

    #: Emitted with a trained classifier so the window can make it active.
    model_trained = Signal(object)
    #: Emitted with a GradeReport once annotations have been graded.
    grades_ready = Signal(object)
    status_message = Signal(str)

    def __init__(self, bank: GeometryBank, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bank = bank
        self.slide = None
        self.store = None
        self.task: BackgroundTask | None = None
        self.model: MLClassifier | None = None
        self.last_grades = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(self._build_bank_box())
        layout.addWidget(self._build_describe_box())
        layout.addWidget(self._build_train_box())
        layout.addWidget(self._build_grade_box())

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #bbb;")
        layout.addWidget(self.status)

        self.results = QTableWidget(0, 5)
        self.results.setHorizontalHeaderLabels(["Class", "Precision", "Recall", "F1", "n"])
        self.results.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.results, 1)

        self.refresh()

    # -- construction -----------------------------------------------------

    def _build_bank_box(self) -> QWidget:
        box = QGroupBox("Descriptor bank")
        layout = QVBoxLayout(box)
        self.bank_label = QLabel()
        self.bank_label.setWordWrap(True)
        layout.addWidget(self.bank_label)

        row = QHBoxLayout()
        for text, slot in (("Clear…", self._clear), ("Reload", self._reload)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _build_describe_box(self) -> QWidget:
        box = QGroupBox("Describe this slide")
        form = QFormLayout(box)

        self.target_mpp = QDoubleSpinBox()
        self.target_mpp.setRange(0.1, 8.0)
        self.target_mpp.setSingleStep(0.25)
        self.target_mpp.setValue(GeometryConfig().target_mpp)
        self.target_mpp.setToolTip(
            "Analysis resolution for the interior block. Every annotation is "
            "measured at the pyramid level nearest this, so descriptors stay "
            "comparable across lesions of different sizes.")
        form.addRow("Target µm/px", self.target_mpp)

        self.min_nuclei = QSpinBox()
        self.min_nuclei.setRange(0, 1000)
        self.min_nuclei.setValue(GeometryConfig().min_nuclei)
        self.min_nuclei.setToolTip(
            "Interior descriptors need this many nuclei. A single duct of "
            "~2,000 µm² holds only 10-20, so a high value silently rejects most "
            "small annotations. Outline shape is unaffected by this.")
        form.addRow("Min nuclei", self.min_nuclei)

        self.window_px = QSpinBox()
        self.window_px.setRange(128, 4096)
        self.window_px.setSingleStep(128)
        self.window_px.setValue(512)
        form.addRow("Window px", self.window_px)

        self.describe_button = QPushButton("Describe Annotations")
        self.describe_button.clicked.connect(self._describe)
        form.addRow("", self.describe_button)
        return box

    def _build_train_box(self) -> QWidget:
        box = QGroupBox("Train a grade model")
        form = QFormLayout(box)

        self.source_combo = QComboBox()
        for source in FeatureSource:
            self.source_combo.addItem(f"{source.label}  ({source.dimension}-D)",
                                      userData=source)
        self.source_combo.setToolTip(
            "Shape uses only the traced outline and works on every annotation. "
            "Texture measures interior nuclei and lumina but needs enough of "
            "them. Combined requires both blocks to be present.")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Features", self.source_combo)

        # Class selection matters more here than it looks. The bank accumulates
        # across slides *and* across annotation schemes — PanIN-* and PanINN*
        # are two different approaches, not a naming slip — so training on
        # everything would pool two schemes into one contradictory problem.
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(150)
        self.class_list.setToolTip(
            "Only ticked classes are trained. Keep separate annotation schemes "
            "apart: tick one scheme's classes, train, then the other's.")
        self.class_list.itemChanged.connect(self._update_ready)
        form.addRow("Classes", self.class_list)

        scheme_row = QHBoxLayout()
        for text, slot in (("All", lambda: self._set_all_classes(True)),
                           ("None", lambda: self._set_all_classes(False))):
            button = QPushButton(text)
            button.setMaximumWidth(60)
            button.clicked.connect(slot)
            scheme_row.addWidget(button)
        scheme_row.addStretch(1)
        form.addRow("", scheme_row)

        self.validation = QComboBox()
        for scheme in Validation:
            self.validation.addItem(scheme.label, userData=scheme)
        self.validation.setCurrentIndex(
            [self.validation.itemData(i) for i in range(self.validation.count())]
            .index(Validation.BY_SLIDE))
        self.validation.setToolTip(
            "Stratified splits annotations at random, so every slide appears in "
            "both train and test — the model can score well by recognising a "
            "slide's staining and your tracing style. Leave-one-slide-out holds "
            "out whole slides and is the number that says whether this "
            "generalises. Measured here: shape scores 61.2% stratified but "
            "46.3% by slide, against a 43.3% majority baseline.")
        form.addRow("Validation", self.validation)

        self.folds = QSpinBox()
        self.folds.setRange(2, 20)
        self.folds.setValue(5)
        self.folds.setToolTip(
            "Cross-validation folds. Capped by the smallest class, since a fold "
            "with no example of a class cannot validate it.")
        form.addRow("CV folds", self.folds)

        self.l2 = QDoubleSpinBox()
        self.l2.setRange(0.0, 1.0)
        self.l2.setDecimals(3)
        self.l2.setSingleStep(0.01)
        self.l2.setValue(0.01)
        form.addRow("L2", self.l2)

        row = QHBoxLayout()
        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self._train)
        self.save_button = QPushButton("Save Model…")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        self.edit_classes_button = QPushButton("Edit Classes…")
        self.edit_classes_button.setEnabled(False)
        self.edit_classes_button.setToolTip(
            "Rename the trained classes, or pool several into one — for "
            "example every PanIN grade into a single PanIN class — and save "
            "that as a new model.")
        self.edit_classes_button.clicked.connect(self._edit_classes)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "Abandon the run in progress. Cancellation is cooperative: it "
            "takes effect at the next checkpoint, so a fold already underway "
            "finishes first. Nothing partial is kept.")
        self.stop_button.clicked.connect(self._stop)
        row.addWidget(self.train_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.save_button)
        row.addWidget(self.edit_classes_button)
        form.addRow("", row)
        return box

    def _build_grade_box(self) -> QWidget:
        """Apply a geometry model to the traced annotations on this slide.

        This is the geometry model's inference path. It cannot go through the
        Predict sheet: that tiles a region and feeds 224px patches to an ONNX
        extractor, and a tile has no lesion outline to describe.
        """
        box = QGroupBox("Grade this slide's annotations")
        layout = QVBoxLayout(box)

        self.grade_model_label = QLabel()
        self.grade_model_label.setWordWrap(True)
        self.grade_model_label.setStyleSheet("color: #bbb;")
        layout.addWidget(self.grade_model_label)

        row = QHBoxLayout()
        self.grade_button = QPushButton("Grade Annotations")
        self.grade_button.setToolTip(
            "Describe each checked annotation and ask the model what it is. "
            "Nothing is written back until you choose to apply the verdicts.")
        self.grade_button.clicked.connect(self._grade)
        load = QPushButton("Load Model…")
        load.setToolTip("Grade with a saved geometry model instead of the one "
                        "just trained.")
        load.clicked.connect(self._load_model)
        row.addWidget(self.grade_button)
        row.addWidget(load)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    # -- state ------------------------------------------------------------

    def set_slide(self, slide, store) -> None:
        self.slide = slide
        self.store = store
        self._update_ready()

    def annotations_changed(self) -> None:
        """The store's annotation set — or its tick marks — changed.

        The describe button's label and enabled state are derived from the
        store, so binding them only at slide-open leaves them stale for every
        later edit. Importing a GeoJSON into a slide that had no sidecar was
        the visible case: the annotations appear in the sidebar and on the
        canvas, but the button stays disabled reading “Describe 0”.
        """
        self._update_ready()

    @property
    def source(self) -> FeatureSource:
        return self.source_combo.currentData()

    def _config(self) -> GeometryConfig:
        return GeometryConfig(target_mpp=self.target_mpp.value(),
                              min_nuclei=self.min_nuclei.value(),
                              window_px=self.window_px.value())

    def _on_source_changed(self) -> None:
        # Sources expose different record sets, so the class list has to follow.
        self._refresh_classes()
        self._update_ready()

    def refresh(self) -> None:
        stats = self.bank.stats()
        self.bank_label.setText(stats.summary())
        self._refresh_classes()
        self._update_ready()

    def _refresh_classes(self) -> None:
        """Rebuild the class list, keeping whatever was already ticked."""
        previously = self.selected_classes
        first_time = self.class_list.count() == 0
        counts: dict[str, int] = {}
        for record in self.bank.usable_for(self.source):
            counts[record.classification] = counts.get(record.classification, 0) + 1

        self.class_list.blockSignals(True)
        self.class_list.clear()
        for name, count in sorted(counts.items()):
            item = QListWidgetItem(f"{name}   ({count})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Default to everything on first sight; afterwards respect the
            # user's choice so switching feature source does not silently
            # re-tick a scheme they deselected.
            ticked = first_time or name in previously
            item.setCheckState(Qt.CheckState.Checked if ticked
                               else Qt.CheckState.Unchecked)
            self.class_list.addItem(item)
        self.class_list.blockSignals(False)

    @property
    def selected_classes(self) -> set[str]:
        return {self.class_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.class_list.count())
                if self.class_list.item(i).checkState() == Qt.CheckState.Checked}

    def _set_all_classes(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(state)

    def _update_ready(self) -> None:
        running = self.task is not None
        annotations = self.store.in_use() if self.store else []
        self.describe_button.setEnabled(bool(annotations) and self.slide is not None
                                        and not running)
        if self.store is not None:
            skipped = len(self.store.annotations) - len(annotations)
            self.describe_button.setText(
                f"Describe {len(annotations)} Checked Annotation(s)"
                + (f"  ({skipped} unchecked)" if skipped else ""))

        if self.source is None:
            self.train_button.setEnabled(False)
            return

        records = self.bank.usable_for(self.source)
        chosen = self.selected_classes
        # Count what will actually be trained on, so the readiness rule and the
        # message describe the same run.
        #
        # An empty selection means *nothing*, not everything. The trainer's API
        # treats empty class_labels as "all", which is a reasonable default for
        # a caller that never mentions classes — but in a UI where the user has
        # just unticked every box, silently training on all of them would be the
        # opposite of what they asked for.
        selected = [r for r in records if r.classification in chosen]
        classes = {r.classification for r in selected}
        self.train_button.setEnabled(
            len(selected) >= 4 and len(classes) >= 2 and not running)

        self._update_grade_controls()

        self.status.setStyleSheet("color: #bbb;")
        text = (f"{len(selected)} of {len(records)} record(s) selected "
                f"({self.source.label.lower()}, {len(classes)} class(es))")

        # The confusing case: the bank is full, the class list is empty, and
        # nothing says why. A record only carries the blocks that existed when
        # it was described, so an older bank has none of a newer descriptor.
        missing = self.bank.stats().total - len(records)
        if missing > 0:
            text += (f". {missing} record(s) do not carry the "
                     f"{self.source.label.lower()} block — described before it "
                     f"existed, or the annotation could not support it. "
                     f"Describe those slides again to add it")
        if len(classes) < 2:
            text += " — tick at least 2 classes to train"
        self.status.setText(text + ".")

    # -- actions ----------------------------------------------------------

    def _describe(self) -> None:
        if self.slide is None or self.store is None:
            return
        annotations = self.store.in_use()
        if not annotations:
            return
        self._set_running(True)
        self.task = BackgroundTask(describe_annotations, slide=self.slide,
                                   annotations=annotations, bank=self.bank,
                                   config=self._config())
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(self._on_described)
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _update_grade_controls(self) -> None:
        annotations = self.store.in_use() if self.store else []
        running = self.task is not None
        ready = self.model is not None and self.slide is not None and annotations
        self.grade_button.setEnabled(bool(ready) and not running)
        self.grade_button.setText(
            f"Grade {len(annotations)} Checked Annotation(s)" if annotations
            else "Grade Annotations")

        if self.model is None:
            self.grade_model_label.setText(
                "No model — train one above, or load a saved .cl.")
            return
        try:
            source = source_of(self.model)
        except Exception as exc:                     # noqa: BLE001 - shown, not raised
            self.grade_model_label.setText(str(exc))
            return
        honest = ""
        if self.model.metrics is not None:
            honest = f" · scored {self.model.metrics.val_accuracy:.1%} in validation"
        self.grade_model_label.setText(
            f"{source.label} · {', '.join(self.model.class_labels)}{honest}")

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Geometry Model", str(Path.home()),
            f"Classifier (*{CLASSIFIER_SUFFIX});;All files (*)")
        if path:
            self._load_model_from(Path(path))

    def _load_model_from(self, path: Path) -> None:
        """Adopt a saved model, refusing one that cannot grade an outline.

        Split from the dialog so the check is reachable without a file picker.
        """
        try:
            model = MLClassifier.load(path)
            source_of(model)        # rejects a patch model before it is adopted
        except Exception as exc:                     # noqa: BLE001 - reported inline
            self.status.setText(f"Could not use {Path(path).name}: {exc}")
            self.status.setStyleSheet("color: #d04a20;")
            return
        self.model = model
        self.save_button.setEnabled(True)
        self.edit_classes_button.setEnabled(True)
        self._update_grade_controls()
        self.status.setStyleSheet("color: #bbb;")
        self.status_message.emit(f"Loaded {Path(path).name}.")

    def _grade(self) -> None:
        if self.slide is None or self.store is None or self.model is None:
            return
        annotations = self.store.in_use()
        if not annotations:
            return
        self._set_running(True)
        self.task = BackgroundTask(grade_annotations, slide=self.slide,
                                   annotations=annotations, classifier=self.model,
                                   config=self._config())
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(self._on_graded)
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _train(self) -> None:
        self._set_running(True)
        settings = GeometryTrainingSettings(source=self.source,
                                            class_labels=sorted(self.selected_classes),
                                            validation=self.validation.currentData(),
                                            folds=self.folds.value(),
                                            l2=self.l2.value())
        self.task = BackgroundTask(train_geometry_classifier, bank=self.bank,
                                   settings=settings)
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(self._on_trained)
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.progress.setValue(0)
        for widget in (self.describe_button, self.train_button, self.grade_button,
                       self.source_combo, self.target_mpp, self.min_nuclei,
                       self.window_px, self.validation, self.folds, self.l2):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if not running:
            self._update_grade_controls()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status.setText(message)

    # A failed job emits `failed` and then `finished(None)`, so these two run
    # *after* _on_failed. They must leave its message alone — refresh() and
    # _update_ready() both rewrite the status line.

    def _on_described(self, report) -> None:
        self.task = None
        self._set_running(False)
        if report is None:
            return
        self.refresh()
        self.status.setText(report.summary())
        self.status_message.emit(report.summary())

    def _on_graded(self, report) -> None:
        self.task = None
        self._set_running(False)
        if report is None:
            return
        self.last_grades = report
        self.status.setText(report.summary())
        self.status_message.emit(report.summary())
        # The verdict table is a window's worth of content, and applying it
        # needs the classification profile, which the panel does not own.
        self.grades_ready.emit(report)

    def _on_trained(self, model) -> None:
        self.task = None
        self._set_running(False)
        if model is None:
            return
        self._update_ready()
        self.model = model
        self.save_button.setEnabled(True)
        self.edit_classes_button.setEnabled(True)
        self._update_grade_controls()
        self._show_results(model)
        self.model_trained.emit(model)

    def _on_failed(self, message: str) -> None:
        self.task = None
        self._set_running(False)
        # _update_ready rewrites the status line, so re-enable the controls
        # first and report the failure last — otherwise the explanation is
        # replaced by a record count the moment it appears.
        self._update_ready()
        # Reported inline rather than in a modal box: the panel already has a
        # status line, the messages are explanatory rather than urgent, and a
        # modal here blocks anything driving the panel programmatically.
        self.status.setText(message)
        self.status.setStyleSheet("color: #d04a20;")
        self.status_message.emit(message)

    def _show_results(self, model: MLClassifier) -> None:
        metrics = model.metrics
        if metrics is None:
            return
        scheme = self.validation.currentData()
        protocol = ("leave-one-slide-out" if scheme is Validation.BY_SLIDE
                    else f"{self.folds.value()}-fold")
        headline = (f"{self.source.label}: {metrics.val_accuracy:.1%} "
                    f"{protocol} over {metrics.val_count} "
                    f"annotations · macro-F1 {metrics.macro_f1:.3f}")
        # A majority-class baseline is the only honest reference at this scale.
        supports = [c.support for c in metrics.per_class]
        if supports and sum(supports):
            baseline = max(supports) / sum(supports)
            headline += f"  (majority baseline {baseline:.1%})"
            if metrics.val_accuracy <= baseline + 0.02:
                headline += "  — at or below baseline; this is not a usable signal."
        self.status.setText(headline)
        self.status_message.emit(headline)

        self.results.setRowCount(len(metrics.per_class))
        for row, c in enumerate(metrics.per_class):
            for column, text in enumerate((c.label, f"{c.precision:.3f}",
                                           f"{c.recall:.3f}", f"{c.f1:.3f}",
                                           str(c.support))):
                self.results.setItem(row, column, QTableWidgetItem(text))

    def _stop(self) -> None:
        """Abandon the run in progress.

        Cooperative: the pipeline checks a flag between folds and inside the
        descent loop, so this is prompt but not instant. A cancelled run
        produces no model — a half-fitted one would be worse than none.
        """
        if self.task is not None and self.task.is_running:
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping at the next checkpoint…")
            self.task.cancel()

    def _edit_classes(self) -> None:
        """Rename or pool the trained classes, saving the result separately.

        The regrouped model also becomes the one Grade uses, so the change is
        visible immediately rather than only after a reload.
        """
        if self.model is None:
            return
        from ..sheets.edit_classes import EditClassesSheet

        sheet = EditClassesSheet(self.model, self)
        if sheet.exec() and sheet.result_model is not None:
            self.model = sheet.result_model
            self._update_grade_controls()
            self._show_results(self.model)
            self.status_message.emit(
                f"Saved {sheet.saved_path.name} — now reporting "
                f"{', '.join(self.model.class_labels)}.")

    def _save(self) -> None:
        if self.model is None:
            return
        default = Path.home() / f"geometry-{self.source.value}{CLASSIFIER_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Geometry Model", str(default),
            f"Classifier (*{CLASSIFIER_SUFFIX})")
        if not path:
            return
        try:
            self.model.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.status_message.emit(f"Saved {Path(path).name}.")

    def _reload(self) -> None:
        self.bank.load()
        self.refresh()

    def _clear(self) -> None:
        stats = self.bank.stats()
        if stats.is_empty:
            return
        if QMessageBox.warning(
            self, "Clear Geometry Bank",
            f"Delete all {stats.total} descriptor record(s)?\n\n"
            "They can be recomputed from the annotations.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.bank.clear()
        self.refresh()

    def shutdown(self) -> None:
        if self.task is not None and self.task.is_running:
            self.task.cancel()
            self.task.wait(20_000)
