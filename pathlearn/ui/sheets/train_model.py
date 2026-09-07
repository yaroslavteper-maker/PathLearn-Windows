"""Train Model sheet — from ``Views/TrainModelSheet.swift``.

Picks the feature space, the classes, and the null classes, then trains on a
background thread and shows the metrics — including the confusion matrix, which
is the only view that reveals a collapsed model.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from ...data.bank import PatchBank
from ...models.classifier import CLASSIFIER_SUFFIX, MLClassifier
from ...models.classification import ClassificationProfile
from ...pipeline.train import TrainingError, TrainingSettings, train_classifier
from ..workers import BackgroundTask


class TrainModelSheet(QDialog):
    """Configure and run classifier training over the patch bank."""

    def __init__(self, bank: PatchBank, profile: ClassificationProfile,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Train Model")
        self.setMinimumSize(680, 620)

        self.bank = bank
        self.profile = profile
        self.task: BackgroundTask | None = None
        self.model: MLClassifier | None = None

        self.buttons = QDialogButtonBox()
        self.train_button = self.buttons.addButton(
            "Train", QDialogButtonBox.ButtonRole.AcceptRole)
        self.save_button = self.buttons.addButton(
            "Save Model…", QDialogButtonBox.ButtonRole.ActionRole)
        self.edit_classes_button = self.buttons.addButton(
            "Edit Classes…", QDialogButtonBox.ButtonRole.ActionRole)
        self.edit_classes_button.setToolTip(
            "Rename the trained classes, or pool several into one, and save "
            "the result as a new model.")
        self.edit_classes_button.clicked.connect(self._edit_classes)
        self.stop_button = self.buttons.addButton(
            "Stop", QDialogButtonBox.ButtonRole.DestructiveRole)
        self.stop_button.setToolTip(
            "Abandon the run in progress. Cancellation is cooperative, so it "
            "takes effect at the next checkpoint. Nothing partial is kept.")
        self.stop_button.clicked.connect(self._stop)
        self.close_button = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.save_button.setEnabled(False)
        self.edit_classes_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_source_box())

        tabs = QTabWidget()
        tabs.addTab(self._build_classes_tab(), "Classes")
        tabs.addTab(self._build_options_tab(), "Options")
        tabs.addTab(self._build_results_tab(), "Results")
        self.tabs = tabs
        layout.addWidget(tabs, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.train_button.clicked.connect(self._start)
        self.save_button.clicked.connect(self._save)
        self.close_button.clicked.connect(self.reject)
        layout.addWidget(self.buttons)

        self._on_extractor_changed()

    # -- construction -----------------------------------------------------

    def _build_source_box(self) -> QWidget:
        box = QGroupBox("Feature space")
        form = QFormLayout(box)

        self.extractor_combo = QComboBox()
        stats = self.bank.stats()
        for identity in sorted(stats.by_extractor):
            count = stats.by_extractor[identity]
            dim = stats.feature_dims.get(identity, "?")
            self.extractor_combo.addItem(f"{identity}   {count} patches   {dim}-d",
                                         userData=identity)
        if not stats.by_extractor:
            self.extractor_combo.addItem("Bank is empty — extract patches first",
                                         userData=None)
            self.extractor_combo.setEnabled(False)
        self.extractor_combo.currentIndexChanged.connect(self._on_extractor_changed)
        form.addRow("Trained on", self.extractor_combo)

        self.source_note = QLabel()
        self.source_note.setWordWrap(True)
        self.source_note.setStyleSheet("color: #999;")
        form.addRow("", self.source_note)
        return box

    def _build_classes_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Train these classes:"))
        self.class_list = QListWidget()
        self.class_list.itemChanged.connect(self._update_counts)
        layout.addWidget(self.class_list, 1)

        layout.addWidget(QLabel(
            "Exclude / null classes — their patches never become a label, and "
            "candidates resembling them are dropped:"))
        self.null_list = QListWidget()
        self.null_list.itemChanged.connect(self._update_counts)
        layout.addWidget(self.null_list, 1)

        self.counts_label = QLabel()
        self.counts_label.setWordWrap(True)
        self.counts_label.setStyleSheet("color: #bbb;")
        layout.addWidget(self.counts_label)
        return page

    def _build_options_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.pooled = QComboBox()
        self.pooled.addItem("Per patch — one example per patch", userData=False)
        self.pooled.addItem("Pooled — one mean/max/std vector per annotation",
                            userData=True)
        self.pooled.setToolTip(
            "Pooled training gives one example per annotation, so it needs far "
            "more annotations than patches. Doc 04 records it collapsing to the "
            "majority class at ~136 annotations.")
        self.pooled.currentIndexChanged.connect(self._update_counts)
        form.addRow("Aggregation", self.pooled)

        self.null_threshold = QDoubleSpinBox()
        self.null_threshold.setRange(0.0, 1.0)
        self.null_threshold.setSingleStep(0.01)
        self.null_threshold.setDecimals(2)
        self.null_threshold.setValue(0.85)
        self.null_threshold.setToolTip(
            "Cosine similarity to the nearest null patch. Cosine compares "
            "direction and ignores magnitude, so a patch can match a null "
            "vector despite a very different brightness.")
        form.addRow("Null threshold", self.null_threshold)

        self.val_fraction = QDoubleSpinBox()
        self.val_fraction.setRange(0.05, 0.5)
        self.val_fraction.setSingleStep(0.05)
        self.val_fraction.setDecimals(2)
        self.val_fraction.setValue(0.2)
        form.addRow("Validation fraction", self.val_fraction)

        self.iterations = QSpinBox()
        self.iterations.setRange(10, 5000)
        self.iterations.setSingleStep(50)
        self.iterations.setValue(200)
        form.addRow("Iterations", self.iterations)

        self.l2 = QDoubleSpinBox()
        self.l2.setRange(0.0, 1.0)
        self.l2.setDecimals(4)
        self.l2.setSingleStep(0.001)
        self.l2.setValue(0.001)
        form.addRow("L2 regularisation", self.l2)

        self.max_white = QDoubleSpinBox()
        self.max_white.setRange(0.0, 1.0)
        self.max_white.setSingleStep(0.05)
        self.max_white.setDecimals(2)
        self.max_white.setValue(1.0)
        self.max_white.setToolTip("1.00 keeps every patch already in the bank.")
        self.max_white.valueChanged.connect(self._update_counts)
        form.addRow("Max white fraction", self.max_white)

        note = QLabel(
            "Validation splits <b>patches</b>, not annotations, so patches from "
            "one region can land on both sides. Scores are optimistic when a "
            "class comes from few regions.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #d08a20;")
        form.addRow("", note)
        return page

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.result_headline = QLabel("Not trained yet.")
        self.result_headline.setWordWrap(True)
        self.result_headline.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.result_headline)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #d04a20; font-weight: 600;")
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)

        layout.addWidget(QLabel("Per class:"))
        self.per_class_table = QTableWidget(0, 5)
        self.per_class_table.setHorizontalHeaderLabels(
            ["Class", "Precision", "Recall", "F1", "n"])
        self.per_class_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.per_class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.per_class_table)

        layout.addWidget(QLabel("Confusion (rows = actual, columns = predicted):"))
        self.confusion_table = QTableWidget(0, 0)
        self.confusion_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.confusion_table)
        return page

    # -- state ------------------------------------------------------------

    @property
    def extractor_identity(self) -> str | None:
        return self.extractor_combo.currentData()

    def _checked(self, widget: QListWidget) -> list[str]:
        return [widget.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(widget.count())
                if widget.item(i).checkState() == Qt.CheckState.Checked]

    def _on_extractor_changed(self) -> None:
        identity = self.extractor_identity
        self.class_list.blockSignals(True)
        self.null_list.blockSignals(True)
        self.class_list.clear()
        self.null_list.clear()

        if identity is None:
            self.source_note.setText(
                "Use Machine Learning ▸ Extract Patches to populate the bank.")
            self.train_button.setEnabled(False)
            self.class_list.blockSignals(False)
            self.null_list.blockSignals(False)
            return

        null_names = self.profile.null_class_names
        counts: dict[str, int] = {}
        for patch in self.bank.fetch(extractor_identity=identity):
            counts[patch.classification] = counts.get(patch.classification, 0) + 1

        for name, count in sorted(counts.items()):
            for widget, default_on in ((self.class_list, name not in null_names),
                                       (self.null_list, name in null_names)):
                item = QListWidgetItem(f"{name}   ({count} patches)")
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if default_on
                                   else Qt.CheckState.Unchecked)
                widget.addItem(item)

        self.class_list.blockSignals(False)
        self.null_list.blockSignals(False)
        self.source_note.setText(
            f"{sum(counts.values())} patches across {len(counts)} classes. "
            f"A model is trained against exactly one feature space.")
        self._update_counts()

    def _update_counts(self) -> None:
        classes = [c for c in self._checked(self.class_list)
                   if c not in self._checked(self.null_list)]
        nulls = self._checked(self.null_list)
        ready = self.extractor_identity is not None and len(classes) >= 2
        self.train_button.setEnabled(ready)

        parts = [f"{len(classes)} class(es) selected"]
        if nulls:
            parts.append(f"{len(nulls)} null class(es): {', '.join(nulls)}")
        if len(classes) < 2:
            parts.append("— at least 2 are needed to train")
        self.counts_label.setText("  ".join(parts))

    def _settings(self) -> TrainingSettings:
        nulls = self._checked(self.null_list)
        classes = [c for c in self._checked(self.class_list) if c not in nulls]
        return TrainingSettings(
            extractor_identity=self.extractor_identity,
            class_labels=classes,
            null_labels=nulls,
            null_threshold=self.null_threshold.value() if nulls else None,
            pooled=bool(self.pooled.currentData()),
            max_white_fraction=(self.max_white.value()
                                if self.max_white.value() < 1.0 else None),
            iterations=self.iterations.value(),
            l2=self.l2.value(),
            val_fraction=self.val_fraction.value(),
        )

    # -- running ----------------------------------------------------------

    def _start(self) -> None:
        self._set_running(True)
        self.task = BackgroundTask(train_classifier, bank=self.bank,
                                   settings=self._settings())
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(self._on_finished)
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.progress.setValue(0)
        for widget in (self.extractor_combo, self.class_list, self.null_list,
                       self.pooled, self.null_threshold, self.val_fraction,
                       self.iterations, self.l2, self.max_white):
            widget.setEnabled(not running)
        self.train_button.setEnabled(not running)
        self.close_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status.setText(message)

    def _on_finished(self, model) -> None:
        self._set_running(False)
        self.task = None
        if model is None:
            return
        self.model = model
        self.save_button.setEnabled(True)
        self.edit_classes_button.setEnabled(True)
        self._show_results(model)
        self.tabs.setCurrentIndex(2)

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.task = None
        self.status.setText("")
        QMessageBox.warning(self, "Training failed", message)

    def _show_results(self, model: MLClassifier) -> None:
        metrics = model.metrics
        if metrics is None:
            return
        self.result_headline.setText(f"{model.describe()}\n{metrics.summary().splitlines()[0]}")
        self.status.setText("")

        if metrics.is_collapsed:
            self.warning_label.setText(
                "This model predicts a single class — it has collapsed. Accuracy "
                "is meaningless here; look at the confusion matrix.")
            self.warning_label.setVisible(True)
        elif metrics.predicted_labels_used < len(metrics.class_labels):
            self.warning_label.setText(
                f"Only {metrics.predicted_labels_used} of "
                f"{len(metrics.class_labels)} classes are ever predicted.")
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)

        self.per_class_table.setRowCount(len(metrics.per_class))
        for row, c in enumerate(metrics.per_class):
            for column, text in enumerate((c.label, f"{c.precision:.3f}",
                                           f"{c.recall:.3f}", f"{c.f1:.3f}",
                                           str(c.support))):
                self.per_class_table.setItem(row, column, QTableWidgetItem(text))

        labels = metrics.class_labels
        self.confusion_table.setRowCount(len(labels))
        self.confusion_table.setColumnCount(len(labels))
        self.confusion_table.setHorizontalHeaderLabels(labels)
        self.confusion_table.setVerticalHeaderLabels(labels)
        for r, row_values in enumerate(metrics.confusion):
            for c, value in enumerate(row_values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if r == c and value:
                    item.setForeground(Qt.GlobalColor.darkGreen)
                elif value:
                    item.setForeground(Qt.GlobalColor.darkRed)
                self.confusion_table.setItem(r, c, item)

    def _stop(self) -> None:
        """Abandon the run in progress; cancellation is cooperative."""
        if self.task is not None and self.task.is_running:
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping at the next checkpoint…")
            self.task.cancel()

    def _edit_classes(self) -> None:
        """Rename or pool the trained classes, saving the result separately."""
        if self.model is None:
            return
        from .edit_classes import EditClassesSheet

        sheet = EditClassesSheet(self.model, self)
        if sheet.exec() and sheet.saved_path is not None:
            self.status.setText(f"Saved {sheet.saved_path.name} — "
                                f"{', '.join(sheet.result_model.class_labels)}.")

    def _save(self) -> None:
        if self.model is None:
            return
        default = Path.home() / f"model{CLASSIFIER_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Classifier", str(default),
            f"Classifier (*{CLASSIFIER_SUFFIX});;All files (*)")
        if not path:
            return
        try:
            self.model.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.status.setText(f"Saved {Path(path).name}.")

    def done(self, result: int) -> None:
        if self.task is not None and self.task.is_running:
            self.task.cancel()
            self.task.wait(20_000)
        super().done(result)
