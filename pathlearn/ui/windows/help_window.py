"""The manual, with a topic list and a search box.

Non-modal on purpose: the whole point is to read an instruction and then do
the thing, which a modal dialog would prevent.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from ..help_content import TOPICS, Topic, as_html, search


class HelpWindow(QDialog):
    """Topic list on the left, the topic on the right, search above both."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PathLearn Help")
        self.resize(1000, 720)
        # Non-modal, and it survives being left open while you work.
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Window, True)

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Search:"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText(
            "lasso, stride, batch effect, gated, upside-down…")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._on_search)
        top.addWidget(self.search_box, 1)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.topic_list = QListWidget()
        # The longest titles ("Geometry: describing annotations") overflow a
        # narrow list; elide rather than grow a horizontal scrollbar.
        self.topic_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.topic_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.topic_list.currentRowChanged.connect(self._on_topic)
        splitter.addWidget(self.topic_list)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 700])
        layout.addWidget(splitter, 1)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #888;")
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self._shown: list[Topic] = []
        self._populate(list(TOPICS))

    # -- content ----------------------------------------------------------

    def _populate(self, topics: list[Topic]) -> None:
        self._shown = topics
        self.topic_list.blockSignals(True)
        self.topic_list.clear()
        for topic in topics:
            self.topic_list.addItem(QListWidgetItem(topic.title))
        self.topic_list.blockSignals(False)
        if topics:
            self.topic_list.setCurrentRow(0)
        else:
            # An empty list must not leave the last topic showing as if it
            # were a result.
            self.view.setHtml("<p>No topic matches that.</p>")
        self._update_status()

    def _on_search(self, text: str) -> None:
        self._populate(search(text))

    def _on_topic(self, row: int) -> None:
        if 0 <= row < len(self._shown):
            self.view.setHtml(as_html(self._shown[row]))
            self.view.verticalScrollBar().setValue(0)
        self._update_status()

    def _update_status(self) -> None:
        total = len(TOPICS)
        shown = len(self._shown)
        self.status.setText(f"{total} topics"
                            if shown == total else f"{shown} of {total} topics")

    def show_topic(self, title: str) -> bool:
        """Jump to a topic by title. Used by context-sensitive Help entries."""
        self.search_box.clear()
        for row, topic in enumerate(self._shown):
            if topic.title == title:
                self.topic_list.setCurrentRow(row)
                return True
        return False
