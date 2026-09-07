"""Annotation list + class palette — from ``Views/AnnotationSidebar.swift``
and ``Views/ClassesListView.swift``."""

from __future__ import annotations

import uuid

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...models.annotation import AnnotationColor
from ...models.classification import Classification, ClassificationProfile
from ...models.store import AnnotationStore


def color_icon(color: AnnotationColor, size: int = 12) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color.r, color.g, color.b))
    return QIcon(pixmap)


class AnnotationSidebar(QWidget):
    """Class picker on top, annotation table below."""

    #: Emitted when the user asks to zoom to an annotation.
    zoom_requested = Signal(object)   # uuid.UUID
    #: Emitted when the active drawing class changes.
    class_changed = Signal()
    #: Emitted when annotations change in a way the canvas must repaint for.
    annotations_changed = Signal()
    #: Asks the window for the class-breakdown sheet — the pixel size lives
    #: on the slide, which the sidebar does not hold.
    breakdown_requested = Signal()
    #: Asks the window to rename a class. The sidebar owns the palette but
    #: not the banks or the other slides the name is also stamped on.
    rename_class_requested = Signal(str)

    def __init__(self, store: AnnotationStore, profile: ClassificationProfile,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.profile = profile
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(_section_label("Draw as"))
        class_row = QHBoxLayout()
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_class_selected)
        class_row.addWidget(self.class_combo, 1)

        # The swatch is the control, not a legend. The colour used to be
        # reachable only through a separate "Colour…" button, which is the last
        # place you look when the thing you want to click is the colour itself.
        self.color_button = QPushButton()
        self.color_button.setFixedSize(36, 24)
        self.color_button.clicked.connect(self._recolor_class)
        class_row.addWidget(self.color_button)
        layout.addLayout(class_row)

        class_buttons = QHBoxLayout()
        add_class = QPushButton("Add Class…")
        add_class.clicked.connect(self._add_class)
        class_buttons.addWidget(add_class)
        self.rename_class_button = QPushButton("Rename Class…")
        self.rename_class_button.setToolTip(
            "Rename this class everywhere it is recorded — the annotations "
            "on this slide, the patch bank, the geometry bank, and "
            "optionally the other slides you have opened.")
        self.rename_class_button.clicked.connect(self._rename_class)
        class_buttons.addWidget(self.rename_class_button)
        class_buttons.addStretch(1)
        layout.addLayout(class_buttons)

        self.null_checkbox = QCheckBox("Exclude / null class")
        self.null_checkbox.setToolTip(
            "Patches of a null class never train as a real class, and candidates "
            "that resemble them are dropped from training and prediction."
        )
        self.null_checkbox.toggled.connect(self._on_null_toggled)
        layout.addWidget(self.null_checkbox)

        layout.addSpacing(6)
        layout.addWidget(_section_label("Annotations"))
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Use", "Class", "Name", "Area"])
        self.tree.setToolTip(
            "Use — include this annotation in patch extraction, geometry "
            "and batch delete. Unticked annotations are still drawn, as a "
            "faint dotted outline, so exclusion stays visible.")
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._on_row_selected)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_row_double_clicked)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree, 1)

        # Bulk Use, because the likely repair on a slide imported from macOS is
        # "put everything back in play" rather than sixteen individual ticks.
        use_row = QHBoxLayout()
        self.use_all_button = QPushButton("Use All")
        self.use_all_button.clicked.connect(lambda: self._select_all(True))
        self.use_none_button = QPushButton("Use None")
        self.use_none_button.clicked.connect(lambda: self._select_all(False))
        use_row.addWidget(self.use_all_button)
        use_row.addWidget(self.use_none_button)
        # Batch delete is driven by the same Use ticks, so what gets deleted is
        # what the table already shows as checked. The count is in the label
        # because Use defaults to *everything* — an unlabelled "Delete Checked"
        # next to a fully ticked list is a wipe-the-slide button in disguise.
        self.delete_checked_button = QPushButton("Delete Checked")
        self.delete_checked_button.setToolTip(
            "Delete every annotation ticked under Use, in one step.")
        self.delete_checked_button.clicked.connect(self._delete_checked)
        use_row.addWidget(self.delete_checked_button)
        use_row.addStretch(1)
        layout.addLayout(use_row)

        self.breakdown_button = QPushButton("Class Breakdown…")
        self.breakdown_button.setToolTip(
            "What percentage of your annotations each class accounts for — "
            "by number of regions and by the area they cover.")
        self.breakdown_button.clicked.connect(self.breakdown_requested.emit)
        use_row.addWidget(self.breakdown_button)

        self.count_label = QLabel("—")
        self.count_label.setStyleSheet("color: #888;")
        layout.addWidget(self.count_label)

        buttons = QHBoxLayout()
        for text, slot in (("Rename…", self._rename_selected),
                           ("Reclassify…", self._reclassify_selected),
                           ("Delete", self._delete_selected)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.subtractive_checkbox = QCheckBox("Selected region is subtractive")
        self.subtractive_checkbox.setToolTip(
            "Subtractive polygons carve regions out of enclosing annotations: "
            "patches whose centre falls inside are cancelled before extraction."
        )
        self.subtractive_checkbox.toggled.connect(self._on_subtractive_toggled)
        layout.addWidget(self.subtractive_checkbox)

        self.reload_profile(profile)

    # -- profile ----------------------------------------------------------

    def reload_profile(self, profile: ClassificationProfile) -> None:
        self.profile = profile
        self._syncing = True
        self.class_combo.clear()
        for cls in profile.classes:
            self.class_combo.addItem(color_icon(cls.color), cls.name, userData=cls.name)
        self._syncing = False
        if profile.classes:
            index = max(0, self.class_combo.findData(self.store.current_label))
            self.class_combo.setCurrentIndex(index)
            self._on_class_selected(index)
        else:
            self._sync_color_button()

    @property
    def current_class(self) -> Classification | None:
        return self.profile.by_name(self.class_combo.currentData())

    def _sync_color_button(self) -> None:
        """Paint the swatch in the current class's colour."""
        cls = self.current_class
        if cls is None:
            self.color_button.setEnabled(False)
            self.color_button.setStyleSheet("")
            self.color_button.setToolTip("No class selected.")
            return
        self.color_button.setEnabled(True)
        # A border, because a swatch the same colour as the panel would look
        # like an empty gap rather than a button.
        self.color_button.setStyleSheet(
            f"background-color: rgb({cls.color.r}, {cls.color.g}, {cls.color.b});"
            " border: 1px solid #666; border-radius: 3px;")
        self.color_button.setToolTip(
            f"Colour for {cls.name} — click to change it. Every annotation "
            f"already drawn in this class is recoloured too.")

    def _on_class_selected(self, index: int) -> None:
        if self._syncing or index < 0:
            return
        cls = self.profile.by_name(self.class_combo.itemData(index))
        if cls is None:
            return
        self.store.current_label = cls.name
        self.store.current_color = cls.color
        self._syncing = True
        self.null_checkbox.setChecked(cls.is_null)
        self._syncing = False
        self._sync_color_button()
        self.class_changed.emit()

    def _add_class(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if self.profile.by_name(name) is not None:
            QMessageBox.warning(self, "Add Class", f"“{name}” already exists.")
            return
        chosen = QColorDialog.getColor(QColor(200, 60, 60), self, "Colour for " + name)
        if not chosen.isValid():
            return
        self.profile.classes.append(
            Classification(name, AnnotationColor(chosen.red(), chosen.green(), chosen.blue()))
        )
        self.reload_profile(self.profile)
        self.class_combo.setCurrentIndex(self.class_combo.findData(name))

    def _rename_class(self) -> None:
        current = self.current_class
        if current is not None:
            self.rename_class_requested.emit(current.name)

    def apply_class_rename(self, old: str, new: str) -> None:
        """Point the palette at the new name and rebuild both lists."""
        for classification in self.profile.classes:
            if classification.name == old:
                classification.name = new
        # A merge leaves two palette entries with one name; keep the first.
        seen: set[str] = set()
        kept = []
        for classification in self.profile.classes:
            if classification.name in seen:
                continue
            seen.add(classification.name)
            kept.append(classification)
        self.profile.classes[:] = kept
        self.reload_profile(self.profile)
        self.refresh()

    def _recolor_class(self) -> None:
        cls = self.current_class
        if cls is None:
            return
        current = QColor(cls.color.r, cls.color.g, cls.color.b)
        chosen = QColorDialog.getColor(current, self, "Colour for " + cls.name)
        if not chosen.isValid():
            return
        self.set_class_color(
            cls.name, AnnotationColor(chosen.red(), chosen.green(), chosen.blue()))

    def set_class_color(self, name: str, color: AnnotationColor) -> int:
        """Recolour a class and everything already drawn in it.

        Split from the picker so the effect is reachable without a modal
        dialog. Returns how many annotations changed — the colour lives on
        each annotation as well as on the class, so leaving them behind would
        make the class colour and the canvas disagree.
        """
        cls = self.profile.by_name(name)
        if cls is None:
            return 0
        cls.color = color
        changed = 0
        for ann in self.store.annotations:
            if ann.classification == cls.name:
                ann.color = color
                changed += 1
        if changed:
            self.store.save()
        self.reload_profile(self.profile)
        self.refresh()
        self.annotations_changed.emit()
        return changed

    def _on_null_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        cls = self.current_class
        if cls is not None:
            cls.is_null = checked

    # -- annotation table -------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the table from the store."""
        self._syncing = True
        self.tree.clear()
        for ann in self.store.annotations:
            item = QTreeWidgetItem([
                "",
                ann.classification + (" (sub)" if ann.is_subtractive else ""),
                ann.display_name,
                ann.area_short,
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, ann.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if ann.is_selected
                               else Qt.CheckState.Unchecked)
            item.setIcon(1, color_icon(ann.color))
            self.tree.addTopLevelItem(item)
            if ann.id == self.store.selected_id:
                item.setSelected(True)
        self._syncing = False

        self._update_counts()
        self._sync_subtractive_checkbox()

    def select_annotation(self, annotation_id) -> bool:
        """Highlight one annotation's row and scroll it into view.

        Scrolling is the point. ``refresh`` already marked the row selected,
        but on a slide with a hundred regions the highlighted one is usually
        somewhere off the bottom of the list, so clicking the slide looked
        like it had done nothing at all.
        """
        self.refresh()
        if annotation_id is None:
            self.tree.setCurrentItem(None)
            return False
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole) == annotation_id:
                self._syncing = True
                self.tree.setCurrentItem(item)
                self._syncing = False
                self.tree.scrollToItem(
                    item, QAbstractItemView.ScrollHint.PositionAtCenter)
                return True
        return False

    def _sync_subtractive_checkbox(self) -> None:
        selected = self.store.selected
        self._syncing = True
        self.subtractive_checkbox.setEnabled(selected is not None)
        self.subtractive_checkbox.setChecked(bool(selected and selected.is_subtractive))
        self._syncing = False

    def _selected_id(self) -> uuid.UUID | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.ItemDataRole.UserRole)

    def _on_row_selected(self) -> None:
        if self._syncing:
            return
        self.store.selected_id = self._selected_id()
        self._sync_subtractive_checkbox()
        self.annotations_changed.emit()

    def _on_row_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        annotation_id = item.data(0, Qt.ItemDataRole.UserRole)
        if annotation_id is not None:
            self.zoom_requested.emit(annotation_id)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != 0:
            return
        annotation_id = item.data(0, Qt.ItemDataRole.UserRole)
        if annotation_id is None:
            return
        checked = item.checkState(column) == Qt.CheckState.Checked
        self.store.set_selected(annotation_id, checked)
        self._update_counts()
        self.annotations_changed.emit()

    def _update_counts(self) -> None:
        total = len(self.store.annotations)
        checked = len(self.store.in_use(include_subtractive=True))
        self.delete_checked_button.setEnabled(checked > 0)
        self.delete_checked_button.setText(
            f"Delete Checked ({checked})" if checked else "Delete Checked")
        self.count_label.setText(
            f"{total} annotation(s) — {self.store.selected_count} in use"
            if total else "No annotations yet — pick Lasso or Polygon and draw.")

    def _on_subtractive_toggled(self, checked: bool) -> None:
        if self._syncing or self.store.selected_id is None:
            return
        self.store.set_subtractive(self.store.selected_id, checked)
        self.refresh()
        self.annotations_changed.emit()

    # -- row actions ------------------------------------------------------

    def _select_all(self, selected: bool) -> None:
        if self.store.select_all(selected):
            self.refresh()
            self.annotations_changed.emit()

    def _delete_checked(self) -> None:
        """Delete every annotation ticked under Use.

        Subtractive polygons are included when ticked: the row shows a checkbox
        like any other, so excluding them would contradict what the user sees.
        """
        doomed = self.store.in_use(include_subtractive=True)
        if not doomed:
            return
        counts: dict[str, int] = {}
        for annotation in doomed:
            counts[annotation.classification] = counts.get(annotation.classification, 0) + 1

        # Spell out the batch: the count alone does not say whether this is the
        # three you ticked or the whole slide.
        lines = [f"Delete {len(doomed)} checked annotation(s)?"]
        if len(doomed) == len(self.store.annotations):
            lines.append("That is every annotation on this slide.")
        lines.append("")
        lines += [f"  {name} — {n}" for name, n in sorted(counts.items())]
        lines += ["", "This cannot be undone."]

        if QMessageBox.warning(
            self, "Delete Checked Annotations", "\n".join(lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.delete_checked_now(doomed)

    def delete_checked_now(self, doomed) -> int:
        """Perform the batch delete. Split out so tests skip the dialog."""
        removed = self.store.remove_many([a.id for a in doomed])
        self.refresh()
        self.annotations_changed.emit()
        return removed

    def _rename_selected(self) -> None:
        annotation_id = self._selected_id()
        if annotation_id is None:
            return
        current = self.store.by_id(annotation_id)
        name, ok = QInputDialog.getText(self, "Rename Annotation", "Name:",
                                        text=current.display_name if current else "")
        if ok:
            self.store.rename(annotation_id, name)
            self.refresh()
            self.annotations_changed.emit()

    def _reclassify_selected(self) -> None:
        annotation_id = self._selected_id()
        if annotation_id is None or not self.profile.classes:
            return
        names = [c.name for c in self.profile.classes]
        current = self.store.by_id(annotation_id)
        start = names.index(current.classification) if current and current.classification in names else 0
        name, ok = QInputDialog.getItem(self, "Reclassify", "Class:", names, start, False)
        if not ok:
            return
        cls = self.profile.by_name(name)
        if cls is not None:
            self.store.set_classification(annotation_id, cls.name, cls.color)
            self.refresh()
            self.annotations_changed.emit()

    def _delete_selected(self) -> None:
        annotation_id = self._selected_id()
        if annotation_id is None:
            return
        annotation = self.store.by_id(annotation_id)
        label = annotation.classification if annotation else "this annotation"
        if QMessageBox.question(self, "Delete Annotation",
                                f"Delete {label}? This cannot be undone.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.store.remove(annotation_id)
        self.refresh()
        self.annotations_changed.emit()


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight: 600; color: #ccc;")
    return label
