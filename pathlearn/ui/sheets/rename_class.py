"""Renaming a class everywhere it is recorded, not just in the palette.

A class name is not only a palette entry.  The same string is stamped on every
annotation, on every extracted patch in the bank, and on every geometry
record.  Renaming the palette alone leaves those pointing at a name that no
longer exists — and, worse, the next patches you extract carry the new name,
so the bank ends up training two classes that were always meant to be one.
That failure is silent and only shows up as a model that will not learn.

So this sheet does the whole job, and shows the count for each place before
touching any of it.  Every part is opt-in and every part reports what it did.

OTHER SLIDES
============
Annotations on slides that are not open live in their own ``.geojson``
sidecars.  They can be updated too, and the sheet counts them first so the
choice is informed rather than blind.  The sidecar rewrite preserves each
file's mirroring marker — see ``rename_class_in_sidecar``, where getting that
wrong would silently freeze a macOS slide in its mirrored state forever.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
                               QLabel, QLineEdit, QVBoxLayout, QWidget)

from ...models.store import count_class_in_sidecar


class RenameClassSheet(QDialog):
    """Ask for the new name, and for how far the rename should reach."""

    def __init__(self, old_name: str, *, existing_names: list[str],
                 annotations_here: int = 0, patches: int = 0,
                 geometry_records: int = 0,
                 other_slides: list[Path] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rename Class")
        self.resize(520, 340)
        self.old_name = old_name
        self.existing_names = [n for n in existing_names if n != old_name]
        self.annotations_here = annotations_here
        self.patches = patches
        self.geometry_records = geometry_records
        self.other_slides = list(other_slides or [])
        self.other_counts: dict[Path, int] = {}

        layout = QVBoxLayout(self)

        heading = QLabel(f"Rename <b>{old_name}</b> everywhere it is recorded.")
        heading.setWordWrap(True)
        layout.addWidget(heading)

        form = QFormLayout()
        self.name_edit = QLineEdit(old_name)
        self.name_edit.selectAll()
        self.name_edit.textChanged.connect(self._validate)
        form.addRow("New name:", self.name_edit)
        layout.addLayout(form)

        self.also_patches = QCheckBox("")
        self.also_geometry = QCheckBox("")
        self.also_slides = QCheckBox("")
        for box in (self.also_patches, self.also_geometry, self.also_slides):
            box.setChecked(True)
            layout.addWidget(box)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #d0762a;")
        layout.addWidget(self.warning)

        layout.addStretch(1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._count_other_slides()
        self._describe()
        self._validate()

    # -- contents ---------------------------------------------------------

    def _count_other_slides(self) -> None:
        """Count before offering, so the choice is informed rather than blind."""
        for path in self.other_slides:
            sidecar = Path(path).with_suffix(".geojson")
            if not sidecar.exists():
                continue
            found = count_class_in_sidecar(sidecar, self.old_name)
            if found:
                self.other_counts[sidecar] = found

    @property
    def other_slide_total(self) -> int:
        return sum(self.other_counts.values())

    def _describe(self) -> None:
        self.also_patches.setText(
            f"Also relabel {self.patches:,} patch(es) in the patch bank")
        self.also_patches.setEnabled(self.patches > 0)
        self.also_patches.setChecked(self.patches > 0)

        self.also_geometry.setText(
            f"Also relabel {self.geometry_records:,} geometry record(s)")
        self.also_geometry.setEnabled(self.geometry_records > 0)
        self.also_geometry.setChecked(self.geometry_records > 0)

        slides = len(self.other_counts)
        self.also_slides.setText(
            f"Also update {self.other_slide_total:,} annotation(s) on "
            f"{slides} other slide(s)")
        self.also_slides.setEnabled(slides > 0)
        self.also_slides.setChecked(slides > 0)
        if slides:
            self.also_slides.setToolTip(
                "\n".join(f"{p.name}: {n}"
                          for p, n in list(self.other_counts.items())[:20]))

    def _validate(self) -> None:
        name = self.new_name
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if not name:
            self.warning.setText("Give the class a name.")
            ok.setEnabled(False)
            return
        if name == self.old_name:
            self.warning.setText("That is the current name.")
            ok.setEnabled(False)
            return
        if name in self.existing_names:
            # Merging two classes is a real operation with real consequences
            # for a trained model, so it is stated rather than discovered.
            self.warning.setText(
                f"“{name}” already exists — renaming into it MERGES the two "
                "classes wherever they are recorded. This cannot be undone by "
                "renaming back.")
            ok.setEnabled(True)
            return
        self.warning.setText(
            "Annotations on slides you have never opened in PathLearn are not "
            "listed above and keep the old name."
            if not self.other_counts else "")
        ok.setEnabled(True)

    # -- results ----------------------------------------------------------

    @property
    def new_name(self) -> str:
        return self.name_edit.text().strip()

    @property
    def is_merge(self) -> bool:
        return self.new_name in self.existing_names

    @property
    def update_patches(self) -> bool:
        return self.also_patches.isChecked() and self.also_patches.isEnabled()

    @property
    def update_geometry(self) -> bool:
        return self.also_geometry.isChecked() and self.also_geometry.isEnabled()

    @property
    def update_other_slides(self) -> bool:
        return self.also_slides.isChecked() and self.also_slides.isEnabled()

    def sidecars_to_update(self) -> list[Path]:
        return list(self.other_counts) if self.update_other_slides else []
