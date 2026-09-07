"""Convert a HuggingFace model to an installed extractor, from inside the app.

The work happens in a separate interpreter (see
:mod:`pathlearn.extractors.conversion`) because torch is ~1.1 GB and the app
does not otherwise need it. This sheet is the front end: pick a model, set it
up if necessary, watch the log, and rescan when it finishes.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QRadioButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ...extractors.conversion import (DEFAULT_VENV, ConversionError,
                                      convert_command, find_environment,
                                      interpreter_for, requirements_path,
                                      run_streaming, setup_command)
from ..workers import BackgroundTask

#: Built-ins carry hand-written architecture recipes verified against the
#: banks the macOS build produced, so they are offered by name.
BUILT_INS = [
    ("phikon-v1", "Phikon v1 — Owkin, 768-d (ungated)"),
    ("uni-v1", "UNI v1 — MahmoodLab, 1024-d (gated)"),
    ("uni2-h", "UNI2-h — MahmoodLab, 1536-d (gated)"),
]


class ConvertModelSheet(QDialog):
    """Pick a model, convert it, and install the result."""

    #: Emitted once an extractor has been written, so the registry can rescan.
    converted = Signal()

    def __init__(self, out_dir: Path, parent: QWidget | None = None,
                 interpreter: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert a Model to ONNX")
        self.resize(820, 640)
        self.out_dir = Path(out_dir)
        self.task: BackgroundTask | None = None
        self.wrote_something = False

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_env_box())
        layout.addWidget(self._build_source_box())

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)          # indeterminate; pip has no total
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText(
            "Conversion output appears here. Expect several minutes and a "
            "large download the first time.")
        layout.addWidget(self.log, 1)

        row = QHBoxLayout()
        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self._convert)
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        row.addWidget(self.convert_button)
        row.addWidget(self.cancel_button)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)

        self.status = find_environment(interpreter)
        self._on_source_changed()
        self._refresh_environment()

    # -- construction -----------------------------------------------------

    def _build_env_box(self) -> QWidget:
        box = QGroupBox("Conversion environment")
        layout = QVBoxLayout(box)
        self.env_label = QLabel()
        self.env_label.setWordWrap(True)
        layout.addWidget(self.env_label)

        note = QLabel(
            "Conversion needs torch, timm and transformers — about 1.1 GB. "
            "They are installed into a separate environment so the app itself "
            "stays small; nothing is added to PathLearn's own dependencies.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        layout.addWidget(note)

        row = QHBoxLayout()
        self.setup_button = QPushButton("Set Up…")
        self.setup_button.setToolTip(f"Create {DEFAULT_VENV} and install the "
                                     f"conversion dependencies into it.")
        self.setup_button.clicked.connect(self._setup)
        self.choose_button = QPushButton("Use Existing Python…")
        self.choose_button.setToolTip(
            "Point at a python.exe that already has torch, timm and "
            "transformers installed.")
        self.choose_button.clicked.connect(self._choose_interpreter)
        row.addWidget(self.setup_button)
        row.addWidget(self.choose_button)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _build_source_box(self) -> QWidget:
        box = QGroupBox("Model")
        form = QFormLayout(box)

        # An explicit group: the two radios end up in different parent
        # widgets (each row is wrapped), and Qt auto-exclusivity is
        # per parent — without this, both can be checked at once.
        self.source_group = QButtonGroup(self)
        self.builtin_radio = QRadioButton("Built-in")
        self.builtin_radio.setChecked(True)
        self.source_group.addButton(self.builtin_radio)
        self.source_group.buttonToggled.connect(self._on_source_changed)
        self.builtin_combo = QComboBox()
        for key, label in BUILT_INS:
            self.builtin_combo.addItem(label, userData=key)
        self.builtin_combo.setToolTip(
            "These carry verified architecture recipes, so their output "
            "matches the feature space existing banks were built in.")
        builtin_row = QHBoxLayout()
        builtin_row.addWidget(self.builtin_radio)
        builtin_row.addWidget(self.builtin_combo, 1)
        form.addRow("", _wrap(builtin_row))

        self.repo_radio = QRadioButton("Other")
        self.source_group.addButton(self.repo_radio)
        self.repo_edit = QLineEdit()
        self.repo_edit.setPlaceholderText("owner/model, or a local folder")
        self.repo_edit.textChanged.connect(self._update_ready)
        self.repo_edit.setToolTip(
            "Any HuggingFace repository id, or a directory containing the "
            "weights AND their config.json. A bare .safetensors file cannot "
            "be converted: a state dict does not record its architecture.")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_folder)
        repo_row = QHBoxLayout()
        repo_row.addWidget(self.repo_radio)
        repo_row.addWidget(self.repo_edit, 1)
        repo_row.addWidget(browse)
        form.addRow("", _wrap(repo_row))

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("derived from the repository")
        self.name_edit.setToolTip("Descriptor name, e.g. conch-v1.")
        form.addRow("Name", self.name_edit)

        self.revision = QSpinBox()
        self.revision.setRange(1, 999)
        self.revision.setToolTip(
            "Part of the extractor identity. Bump it when the weights change "
            "so patches extracted with the old ones stay distinguishable.")
        form.addRow("Revision", self.revision)

        self.input_size = QSpinBox()
        self.input_size.setRange(64, 1024)
        self.input_size.setSingleStep(32)
        self.input_size.setValue(224)
        form.addRow("Input size", self.input_size)

        self.destination = QLabel(str(self.out_dir))
        self.destination.setStyleSheet("color: #888;")
        self.destination.setWordWrap(True)
        form.addRow("Install to", self.destination)
        # Deliberately not calling _on_source_changed() here: it reaches the
        # buttons below, which this box is built before.
        return box

    # -- state ------------------------------------------------------------

    def _on_source_changed(self, *_args) -> None:
        custom = self.repo_radio.isChecked()
        self.builtin_combo.setEnabled(not custom)
        for widget in (self.repo_edit, self.name_edit, self.revision,
                       self.input_size):
            widget.setEnabled(custom)
        self._update_ready()

    def _refresh_environment(self) -> None:
        self.env_label.setText(self.status.summary())
        self.env_label.setStyleSheet("color: #bbb;" if self.status.ready
                                     else "color: #d0762a;")
        self.setup_button.setText("Set Up…" if not self.status.ready
                                  else "Reinstall…")
        self._update_ready()

    def _update_ready(self) -> None:
        running = self.task is not None
        has_source = (not self.repo_radio.isChecked()
                      or bool(self.repo_edit.text().strip()))
        self.convert_button.setEnabled(self.status.ready and has_source
                                       and not running)
        self.cancel_button.setEnabled(running)
        for widget in (self.setup_button, self.choose_button,
                       self.builtin_radio, self.repo_radio):
            widget.setEnabled(not running)

    def _append(self, line: str) -> None:
        self.log.appendPlainText(line)

    # -- environment ------------------------------------------------------

    def _choose_interpreter(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a Python with torch installed", str(Path.home()),
            "Python (python.exe python);;All files (*)")
        if not path:
            return
        self.set_interpreter(Path(path))

    def set_interpreter(self, path: Path) -> None:
        """Adopt *path* if it can convert; say what is missing if it cannot."""
        from ...extractors.conversion import inspect_environment

        self.status = inspect_environment(Path(path))
        self._append(f"Checked {path}: {self.status.summary()}")
        self._refresh_environment()

    def _setup(self) -> None:
        if self.task is not None:
            return
        if QMessageBox.question(
            self, "Set Up Conversion Environment",
            f"Create {DEFAULT_VENV} and install torch, timm and transformers "
            f"into it?\n\nThis downloads roughly 1.1 GB and takes several "
            f"minutes. Nothing is added to PathLearn's own environment.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            commands = setup_command(DEFAULT_VENV, requirements_path())
        except ConversionError as exc:
            self._append(str(exc))
            return
        self._start(commands, self._on_setup_done)

    def _on_setup_done(self, code: int) -> None:
        if code == 0:
            self.set_interpreter(interpreter_for(DEFAULT_VENV))
        else:
            self._append(f"Setup failed with exit code {code}.")
            self._refresh_environment()

    # -- conversion -------------------------------------------------------

    def _convert(self) -> None:
        if self.task is not None or not self.status.ready:
            return
        try:
            command = self._command()
        except ConversionError as exc:
            self._append(str(exc))
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._append(f"$ {' '.join(command)}")
        self._start([command], self._on_convert_done)

    def _command(self) -> list[str]:
        if self.repo_radio.isChecked():
            return convert_command(
                self.status.interpreter, self.out_dir,
                repo=self.repo_edit.text().strip(),
                name=self.name_edit.text().strip() or None,
                revision=self.revision.value(),
                input_size=self.input_size.value())
        return convert_command(self.status.interpreter, self.out_dir,
                               known=self.builtin_combo.currentData())

    def _on_convert_done(self, code: int) -> None:
        if code == 0:
            self.wrote_something = True
            self._append("Done. The extractor is installed; rescanning.")
            self.converted.emit()
        else:
            self._append(f"Conversion failed with exit code {code}. "
                         f"The log above says why — a gated model needs "
                         f"access approval and a HuggingFace login.")

    # -- subprocess plumbing ----------------------------------------------

    def _start(self, commands: list[list[str]], done) -> None:
        """Run commands in order, streaming output, stopping at the first failure."""
        def work(progress=None, should_cancel=None, **_):
            # Worker.progress carries (done, total, message); there is no total
            # for pip or for a conversion, so the counters stay at zero and the
            # line goes in the message slot.
            emit = (lambda line: progress(0, 0, line)) if progress else None
            code = 0
            for command in commands:
                code = run_streaming(command, progress=emit,
                                     should_cancel=should_cancel)
                if code != 0:
                    break
            return code

        self._set_running(True)
        self.task = BackgroundTask(work)
        self.task.worker.progress.connect(
            lambda _done, _total, line: self._append(line))
        self.task.worker.finished.connect(lambda code: self._finish(code, done))
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _finish(self, code, done) -> None:
        self.task = None
        self._set_running(False)
        if code is not None:
            done(code)
        self._update_ready()

    def _on_failed(self, message: str) -> None:
        self.task = None
        self._set_running(False)
        self._append(message)
        self._update_ready()

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self._update_ready()

    def _cancel(self) -> None:
        if self.task is not None and self.task.is_running:
            self._append("Stopping…")
            self.task.cancel()

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Folder containing the model and its config.json",
            str(Path.home()))
        if path:
            self.repo_radio.setChecked(True)
            self.repo_edit.setText(path)

    def done(self, result: int) -> None:
        if self.task is not None and self.task.is_running:
            self.task.cancel()
            self.task.wait(20_000)
        super().done(result)


def _wrap(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget
