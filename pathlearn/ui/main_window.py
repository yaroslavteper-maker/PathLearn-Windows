"""Main window — from ``ContentView.swift`` and ``PaNIN_detectorApp.swift``.

Menu actions are wired directly to methods rather than through a notification
bus; the macOS build used ``NotificationCenter`` only to decouple SwiftUI menu
commands from the view, which Qt does not need.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import (QAction, QActionGroup, QDesktopServices,
                           QKeySequence)
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel,
                               QMainWindow, QMessageBox, QToolBar, QWidget)

from ..coords import CoordinateSpace, Provenance, infer_space
from ..io import geojson
from ..io.slide import SUPPORTED_EXTENSIONS, SlideError, SlideImage
from ..models.annotation import mirror_all_y
from ..models.classification import ClassificationProfile
from ..data.bank import PatchBank
from ..extractors.registry import ExtractorRegistry
from ..models.prediction import (PredictionSet, auto_sidecar_path,
                                 find_sidecar)
from ..models.store import (AnnotationStore, SidecarError,
                            rename_class_in_sidecar)
from .canvas import SlideCanvas, Tool
from .panels.annotation_sidebar import AnnotationSidebar
from .panels.bank_panel import BankPanel
from .sheets.extract_patches import ExtractPatchesSheet
from .sheets.train_model import TrainModelSheet
from .sheets.predict import PredictSheet
from .sheets.rename_class import RenameClassSheet
from .sheets.grade_results import GradeResultsSheet
from .sheets.convert_model import ConvertModelSheet
from .sheets.export_images import ExportImagesSheet
from ..pipeline.composition import composition
from ..pipeline.annotation_stats import breakdown as annotation_breakdown
from ..project import (ANNOTATED, PROJECT_SUFFIX, Project, ProjectEntry,
                       ProjectError, default_project_dir)
from .sheets.annotation_stats import AnnotationStatsSheet
from .sheets.composition import CompositionSheet
from .sheets.stitch_regions import StitchRegionsSheet
from ..models.classifier import MLClassifier
from ..state import (STATE_SUFFIX, StateError, load_state, read_manifest,
                     save_state)
from .panels.heatmap_panel import HeatmapPanel
from .panels.geometry_panel import GeometryPanel
from .windows.help_window import HelpWindow
from .windows.project_window import ProjectWindow
from .windows.tsne_window import TSNEWindow
from ..data.geometry_bank import GeometryBank

APP_NAME = "PathLearn"
ORG_NAME = "PathLearn"


def app_data_dir() -> Path:
    """``%LOCALAPPDATA%\\PathLearn`` — never a shared or default location.

    ``04-DESIGN-DECISIONS.md`` §6: the macOS build once let its store land in a
    shared path and another app clobbered the patch bank.
    """
    base = Path.home() / "AppData" / "Local"
    directory = base / APP_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 950)

        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.slide: SlideImage | None = None
        self.profile = self._load_current_profile()
        self.project: Project | None = None
        self.project_window: ProjectWindow | None = None

        # Auto-saving the heatmap. Serialising 50,000 tiles costs about half
        # a second, so view-state changes are debounced rather than written
        # on every tick of the confidence slider.
        self._heatmap_save_timer = QTimer(self)
        self._heatmap_save_timer.setSingleShot(True)
        self._heatmap_save_timer.setInterval(2000)
        self._heatmap_save_timer.timeout.connect(self._save_heatmap_now)

        self.store = AnnotationStore(on_change=self._on_store_changed)
        self.store.current_label = (self.profile.classes[0].name
                                    if self.profile.classes else "Unlabeled")
        if self.profile.classes:
            self.store.current_color = self.profile.classes[0].color

        self.canvas = SlideCanvas(self.store)
        self.canvas.status_message.connect(self._show_status)
        self.canvas.view_changed.connect(self._update_status)
        self.canvas.selection_changed.connect(self._on_canvas_selection)
        self.setCentralWidget(self.canvas)

        self.sidebar = AnnotationSidebar(self.store, self.profile)
        self.sidebar.zoom_requested.connect(self._zoom_to_annotation)
        self.sidebar.annotations_changed.connect(self.canvas.update)
        self.sidebar.breakdown_requested.connect(
            self.show_annotation_breakdown)
        self.sidebar.rename_class_requested.connect(self.rename_class)
        dock = QDockWidget("Annotations", self)
        dock.setWidget(self.sidebar)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.annotations_dock = dock

        # ML side: extractor registry and patch bank. Both are constructed here
        # so the bank panel can show contents at startup, but no model file is
        # opened until an extraction actually runs — UNI2-h is 2.7 GB and must
        # not be paid for on every launch.
        self.registry = ExtractorRegistry()
        self.bank = PatchBank()
        self.bank_panel = BankPanel(self.bank)
        self.classifier = None
        bank_dock = QDockWidget("Patch Bank", self)
        bank_dock.setWidget(self.bank_panel)
        bank_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                                  | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, bank_dock)
        self.tabifyDockWidget(dock, bank_dock)
        self.bank_dock = bank_dock

        self.heatmap_panel = HeatmapPanel()
        self.heatmap_panel.changed.connect(self._on_heatmap_changed)
        self.heatmap_panel.stitch_requested.connect(self.stitch_predictions)
        self.heatmap_panel.composition_requested.connect(
            self.show_composition)
        self.heatmap_panel.autosave_toggled.connect(self._on_autosave_toggled)
        heatmap_dock = QDockWidget("Heatmap", self)
        heatmap_dock.setWidget(self.heatmap_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, heatmap_dock)
        self.tabifyDockWidget(bank_dock, heatmap_dock)
        self.heatmap_dock = heatmap_dock

        self.geometry_bank = GeometryBank()
        self.geometry_panel = GeometryPanel(self.geometry_bank)
        self.geometry_panel.model_trained.connect(self._on_geometry_model)
        self.geometry_panel.grades_ready.connect(self._show_grades)
        self.geometry_panel.status_message.connect(self._show_status)
        geometry_dock = QDockWidget("Geometry", self)
        geometry_dock.setWidget(self.geometry_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, geometry_dock)
        self.tabifyDockWidget(heatmap_dock, geometry_dock)
        self.geometry_dock = geometry_dock
        dock.raise_()

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._update_actions_enabled()
        self.heatmap_panel.set_autosave(
            self.settings.value("saveHeatmapBesideSlide", True) not in
            (False, "false", "False", 0, "0"))
        self._reopen_last_project()
        self._update_project_actions()
        self._update_status()

    # -- construction -----------------------------------------------------

    def _build_actions(self) -> None:
        self.action_open = QAction("&Open Slide…", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self.open_slide_dialog)

        self.action_close = QAction("&Close Slide", self)
        self.action_close.setShortcut(QKeySequence("Ctrl+W"))
        self.action_close.triggered.connect(self.close_slide)

        self.action_import = QAction("&Import Annotations (GeoJSON)…", self)
        self.action_import.triggered.connect(self.import_geojson_dialog)

        self.action_export = QAction("&Export Annotations (GeoJSON)…", self)
        self.action_export.triggered.connect(self.export_geojson_dialog)

        self.action_export_images = QAction("Export Annotations as &JPEG…", self)
        self.action_export_images.triggered.connect(self.export_images_dialog)

        self.action_reveal = QAction("Show Slide in E&xplorer", self)
        self.action_reveal.triggered.connect(self.reveal_slide)

        self.action_quit = QAction("&Quit", self)
        self.action_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_quit.triggered.connect(self.close)

        # Tools
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.action_pan = self._make_tool_action("&Pan", "V", Tool.PAN, checked=True)
        self.action_lasso = self._make_tool_action("&Lasso", "L", Tool.LASSO)
        self.action_polygon = self._make_tool_action("Pol&ygon", "P", Tool.POLYGON)

        self.action_zoom_in = QAction("Zoom &In", self)
        self.action_zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.action_zoom_in.triggered.connect(lambda: self.canvas.zoom_by(1.25))

        self.action_zoom_out = QAction("Zoom &Out", self)
        self.action_zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.action_zoom_out.triggered.connect(lambda: self.canvas.zoom_by(1 / 1.25))

        self.action_zoom_fit = QAction("Zoom to &Fit", self)
        self.action_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        self.action_zoom_fit.triggered.connect(self.canvas.zoom_to_fit)

        self.action_toggle_annotations = QAction("Show &Annotations", self)
        self.action_toggle_annotations.setCheckable(True)
        self.action_toggle_annotations.setChecked(True)
        self.action_toggle_annotations.setShortcut(QKeySequence("Ctrl+H"))
        self.action_toggle_annotations.toggled.connect(self._toggle_annotations)

        self.action_delete_all = QAction("Delete &All Annotations", self)
        self.action_delete_all.triggered.connect(self.delete_all_annotations)

        self.action_extract = QAction("&Extract Patches…", self)
        self.action_extract.setShortcut(QKeySequence("Ctrl+E"))
        self.action_extract.triggered.connect(self.extract_patches)

        self.action_extractors = QAction("Installed E&xtractors…", self)
        self.action_extractors.triggered.connect(self.show_extractors)

        self.action_convert = QAction("&Convert a Model to ONNX…", self)
        self.action_convert.setToolTip(
            "Turn a HuggingFace model into an installed extractor. Runs in "
            "a separate environment, so torch is never added to this one.")
        self.action_convert.triggered.connect(self.convert_model)

        self.action_train = QAction("&Train Model…", self)
        self.action_train.setShortcut(QKeySequence("Ctrl+T"))
        self.action_train.triggered.connect(self.train_model)

        self.action_predict = QAction("&Predict…", self)
        self.action_predict.setShortcut(QKeySequence("Ctrl+R"))
        self.action_predict.triggered.connect(self.predict)

        self.action_tsne = QAction("t-&SNE Plot…", self)
        self.action_tsne.triggered.connect(self.show_tsne)

        # Projects — a cohort of per-slide composition results.
        self.action_new_project = QAction("&New Project…", self)
        self.action_new_project.triggered.connect(self.new_project)

        self.action_open_project = QAction("&Open Project…", self)
        self.action_open_project.triggered.connect(self.open_project)

        self.action_close_project = QAction("&Close Project", self)
        self.action_close_project.triggered.connect(self.close_project)

        self.action_add_to_project = QAction(
            "&Add Predicted Composition", self)
        self.action_add_to_project.setToolTip(
            "File the current prediction's class breakdown in the open "
            "project, so it can be compared with the other slides in it.")
        self.action_add_to_project.triggered.connect(self.add_to_project)

        self.action_add_annotations_to_project = QAction(
            "Add Annotation &Breakdown", self)
        self.action_add_annotations_to_project.setToolTip(
            "File what you drew on this slide in the open project, as its "
            "own row beside any predicted one.")
        self.action_add_annotations_to_project.triggered.connect(
            self.add_annotations_to_project)

        self.action_project_table = QAction("Project &Table…", self)
        self.action_project_table.triggered.connect(self.show_project_window)

        # Profiles
        self.action_save_state = QAction("Save &State…", self)
        self.action_save_state.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save_state.setToolTip(
            "Save the class palette, both banks, the trained models, the "
            "annotations for every slide you have opened, and your current "
            "settings, as one file.")
        self.action_save_state.triggered.connect(self.save_state_dialog)

        self.action_load_state = QAction("&Open State…", self)
        self.action_load_state.triggered.connect(self.load_state_dialog)

        self.action_slide_info = QAction("Slide &Properties…", self)
        self.action_slide_info.triggered.connect(self.show_slide_properties)

        self.action_help = QAction("PathLearn &Help", self)
        self.action_help.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.action_help.triggered.connect(self.show_help)

        self.action_about = QAction("&About PathLearn", self)
        self.action_about.triggered.connect(self.show_about)

    def _make_tool_action(self, text: str, shortcut: str, tool: Tool,
                          checked: bool = False) -> QAction:
        action = QAction(text, self)
        action.setCheckable(True)
        action.setChecked(checked)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda: self._set_tool(tool))
        self.tool_group.addAction(action)
        return action

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_close)
        file_menu.addSeparator()
        file_menu.addAction(self.action_import)
        file_menu.addAction(self.action_export)
        file_menu.addAction(self.action_export_images)
        file_menu.addSeparator()
        file_menu.addAction(self.action_save_state)
        file_menu.addAction(self.action_load_state)
        file_menu.addSeparator()
        file_menu.addAction(self.action_reveal)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)

        project_menu = self.menuBar().addMenu("&Project")
        project_menu.addAction(self.action_new_project)
        project_menu.addAction(self.action_open_project)
        project_menu.addAction(self.action_close_project)
        project_menu.addSeparator()
        project_menu.addAction(self.action_add_to_project)
        project_menu.addAction(self.action_add_annotations_to_project)
        project_menu.addAction(self.action_project_table)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.action_zoom_in)
        view_menu.addAction(self.action_zoom_out)
        view_menu.addAction(self.action_zoom_fit)
        view_menu.addSeparator()
        view_menu.addAction(self.action_toggle_annotations)
        view_menu.addAction(self.annotations_dock.toggleViewAction())

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction(self.action_pan)
        tools_menu.addAction(self.action_lasso)
        tools_menu.addAction(self.action_polygon)
        tools_menu.addSeparator()
        tools_menu.addAction(self.action_delete_all)

        ml_menu = self.menuBar().addMenu("&Machine Learning")
        ml_menu.addAction(self.action_extract)
        ml_menu.addAction(self.action_train)
        ml_menu.addAction(self.action_predict)
        ml_menu.addAction(self.action_tsne)
        ml_menu.addSeparator()
        ml_menu.addAction(self.action_extractors)
        ml_menu.addAction(self.action_convert)
        ml_menu.addAction(self.bank_dock.toggleViewAction())
        ml_menu.addAction(self.heatmap_dock.toggleViewAction())
        ml_menu.addAction(self.geometry_dock.toggleViewAction())


        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.action_help)
        # Straight to the topic, since these are the two people go looking for
        # when something is already going wrong.
        for title in ("Keyboard shortcuts", "Troubleshooting"):
            action = QAction(title, self)
            action.triggered.connect(lambda _=False, t=title: self.show_help(t))
            help_menu.addAction(action)
        help_menu.addSeparator()
        help_menu.addAction(self.action_slide_info)
        help_menu.addAction(self.action_about)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        toolbar.addAction(self.action_open)
        toolbar.addSeparator()
        toolbar.addAction(self.action_pan)
        toolbar.addAction(self.action_lasso)
        toolbar.addAction(self.action_polygon)
        toolbar.addSeparator()
        toolbar.addAction(self.action_zoom_out)
        toolbar.addAction(self.action_zoom_fit)
        toolbar.addAction(self.action_zoom_in)
        self.addToolBar(toolbar)

    def _build_status_bar(self) -> None:
        self.status_slide = QLabel("No slide")
        self.status_position = QLabel("")
        self.status_zoom = QLabel("")
        for widget in (self.status_slide, self.status_position, self.status_zoom):
            self.statusBar().addPermanentWidget(widget)

    # -- slide lifecycle --------------------------------------------------

    def open_slide_dialog(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
        start_dir = self.settings.value("lastSlideDir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Slide", start_dir,
            f"Whole-slide images ({patterns});;All files (*)",
        )
        if path:
            self.open_slide(Path(path))

    def open_slide(self, path: Path) -> None:
        self.close_slide()
        try:
            slide = SlideImage(path)
        except SlideError as exc:
            QMessageBox.critical(self, "Could not open slide", str(exc))
            return

        self.slide = slide
        self.settings.setValue("lastSlideDir", str(path.parent))
        self._note_slide(path)
        self.canvas.set_slide(slide)

        try:
            report = self.store.bind(path, slide.dimensions.height)
        except SidecarError as exc:
            QMessageBox.warning(self, "Annotations", str(exc))
            report = None

        self._update_window_title()
        self.sidebar.refresh()
        self.geometry_panel.set_slide(slide, self.store)
        self._load_heatmap_for(path)
        self._update_actions_enabled()
        self._update_status()

        if report is not None and report.migrated:
            self._explain_migration(report)
        elif report is not None:
            self._show_status(report.message)

    def _explain_migration(self, report) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Annotations migrated")
        box.setText(f"Migrated {report.count} annotation(s) from the macOS layout.")
        box.setInformativeText(
            "The macOS build drew slides vertically mirrored and saved its sidecar "
            "in that mirrored space, so the coordinates have been flipped once "
            "(y → slideHeight − y) to match this build and QuPath.\n\n"
            f"The original file was backed up as {report.backup_path.name}."
        )
        box.setDetailedText(
            "Verify against QuPath: the sidecar is now written with a top-left "
            "origin and carries a \"pathlearn\" marker, so it will not be migrated "
            "again. If the annotations look upside-down relative to the tissue, "
            "restore the .bak file and report it — that would mean this slide's "
            "sidecar was already upright (e.g. hand-copied from a QuPath export)."
        )
        box.exec()

    def close_slide(self) -> None:
        if self.slide is None:
            return
        # Flush any pending view-state change before the slide goes: after
        # this the window no longer knows which slide the heatmap belonged to.
        if self._heatmap_save_timer.isActive():
            self._heatmap_save_timer.stop()
            self._save_heatmap_now()
        self.heatmap_panel.set_predictions(None)
        self.canvas.set_predictions(None)
        self.store.unbind()
        self.canvas.set_slide(None)
        self.slide.close()
        self.slide = None
        self._update_window_title()
        self.geometry_panel.set_slide(None, self.store)
        self.sidebar.refresh()
        self._update_actions_enabled()
        self._update_status()

    def reveal_slide(self) -> None:
        if self.slide is None:
            return
        import subprocess
        subprocess.Popen(["explorer", "/select,", str(self.slide.path)])

    # -- import / export --------------------------------------------------

    def import_geojson_dialog(self) -> None:
        if self.slide is None:
            QMessageBox.information(self, "Import", "Open a slide first — "
                                    "annotations live in slide coordinates.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Annotations", str(self.slide.path.parent),
            "GeoJSON (*.geojson *.json);;All files (*)")
        if not path:
            return
        try:
            document = geojson.decode_document(Path(path).read_text(encoding="utf-8"))
            incoming = geojson.decode_features(document)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Import failed", f"Could not read GeoJSON:\n{exc}")
            return
        if not incoming:
            QMessageBox.information(self, "Import",
                                    f"No annotations found in {Path(path).name}.")
            return

        # An explicitly picked file is assumed to be QuPath-style top-left, but
        # the user may know better (e.g. a hand-copied macOS sidecar).
        space = infer_space(document, Provenance.IMPORTED)
        if space is CoordinateSpace.TOP_LEFT:
            flip = QMessageBox.question(
                self, "Coordinate origin",
                f"Import {len(incoming)} annotation(s) from {Path(path).name}.\n\n"
                "Treat coordinates as QuPath-style (top-left origin)?\n\n"
                "Choose No if this file came from the macOS PathLearn build, "
                "whose coordinates are vertically mirrored.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if flip == QMessageBox.StandardButton.No:
                incoming = mirror_all_y(incoming, self.slide.dimensions.height)

        replace = True
        if self.store.annotations:
            answer = QMessageBox.question(
                self, "Import Annotations",
                f"This slide already has {len(self.store.annotations)} annotation(s).\n\n"
                "Replace them, or append the imported ones?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            replace = answer == QMessageBox.StandardButton.Yes

        self.store.import_merge(incoming, replace=replace)
        self.sidebar.refresh()
        self.canvas.update()
        self._show_status(f"Imported {len(incoming)} annotation(s).")

    def export_images_dialog(self) -> None:
        """Write one JPEG per ticked annotation, cut from the slide."""
        if self.slide is None:
            QMessageBox.information(self, "Export Images", "Open a slide first.")
            return
        if not self.store.in_use():
            QMessageBox.information(
                self, "Export Images",
                "No annotations are checked under Use, so there is nothing to "
                "export. Use All in the Annotations panel selects everything.")
            return
        sheet = ExportImagesSheet(self.slide, self.store, self)
        sheet.exec()
        if sheet.report is not None and sheet.report.written:
            self._show_status(sheet.report.summary())

    def export_geojson_dialog(self) -> None:
        if self.slide is None or not self.store.annotations:
            QMessageBox.information(self, "Export", "Nothing to export.")
            return
        default = self.slide.path.with_name(self.slide.path.stem + "-qupath.geojson")
        path, _ = QFileDialog.getSaveFileName(self, "Export Annotations", str(default),
                                              "GeoJSON (*.geojson)")
        if not path:
            return
        try:
            # No transform: our internal space *is* QuPath's space.
            Path(path).write_text(geojson.encode(self.store.annotations), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._show_status(f"Exported {len(self.store.annotations)} annotation(s) "
                          f"to {Path(path).name}.")

    # -- classes and state -------------------------------------------------

    def _load_current_profile(self) -> ClassificationProfile:
        """The class palette, carried in settings rather than a profile file.

        The Profile menu is gone: a palette on its own was never a useful unit
        of work, and it is now one part of a saved state. Keeping the JSON in
        settings preserves the thing the old menu actually provided — classes
        and colours surviving a restart.
        """
        stored = self.settings.value("profileJson", "")
        if stored:
            try:
                return ClassificationProfile.from_json(stored)
            except (ValueError, KeyError, json.JSONDecodeError):
                pass  # fall through to the default
        return ClassificationProfile.default()

    def _remember_profile(self) -> None:
        try:
            self.settings.setValue("profileJson", self.profile.to_json())
        except (TypeError, ValueError):
            pass

    def _note_slide(self, path) -> None:
        """Record a slide, so a state knows whose annotations to carry.

        A sidecar is only findable from its slide, and nothing else records
        which slides were worked on.
        """
        known = list(self.settings.value("knownSlides", []) or [])
        if str(path) not in known:
            known.append(str(path))
            self.settings.setValue("knownSlides", known)

    def _slides_with_annotations(self) -> list:
        known = list(self.settings.value("knownSlides", []) or [])
        if self.slide is not None and str(self.slide.path) not in known:
            known.append(str(self.slide.path))
        return [p for p in known if Path(p).is_file()]

    def _current_models(self) -> dict:
        models = {}
        if self.classifier is not None:
            models["patch"] = self.classifier.to_json()
        geometry = getattr(self.geometry_panel, "model", None)
        if geometry is not None:
            models["geometry"] = geometry.to_json()
        return models

    def _state_settings(self) -> dict:
        """The working settings worth carrying, not every widget value."""
        panel = self.geometry_panel
        return {
            "targetMpp": panel.target_mpp.value(),
            "minNuclei": panel.min_nuclei.value(),
            "windowPx": panel.window_px.value(),
            "featureSource": panel.source.value if panel.source else "",
            "folds": panel.folds.value(),
            "l2": panel.l2.value(),
        }

    def _apply_state_settings(self, settings: dict) -> None:
        panel = self.geometry_panel
        for key, widget in (("targetMpp", panel.target_mpp),
                            ("minNuclei", panel.min_nuclei),
                            ("windowPx", panel.window_px),
                            ("folds", panel.folds), ("l2", panel.l2)):
            if key in settings:
                try:
                    widget.setValue(type(widget.value())(settings[key]))
                except (TypeError, ValueError):
                    pass
        source = settings.get("featureSource")
        if source:
            for i in range(panel.source_combo.count()):
                data = panel.source_combo.itemData(i)
                if data is not None and data.value == source:
                    panel.source_combo.setCurrentIndex(i)
                    break

    def save_state_dialog(self) -> None:
        """Write everything the app is holding into one file."""
        self.store.save()
        self._remember_profile()
        default = Path.home() / f"pathlearn-state{STATE_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save State", str(default), f"PathLearn state (*{STATE_SUFFIX})")
        if not path:
            return
        try:
            manifest = self.write_state(Path(path))
        except StateError as exc:
            QMessageBox.critical(self, "Save State", str(exc))
            return
        size = Path(path).stat().st_size / 1e6
        self._show_status(f"Saved {Path(path).name} ({size:.1f} MB) — "
                          f"{manifest.summary()}")

    def write_state(self, path):
        """Do the save. Split from the dialog so it is testable."""
        return save_state(
            path,
            profile_json=self.profile.to_json(),
            profile_name=self.profile.name,
            patch_bank_path=self.bank.path,
            geometry_bank_path=self.geometry_bank.path,
            equivalences_path=app_data_dir() / "extractor-equivalences.json",
            slide_paths=self._slides_with_annotations(),
            models=self._current_models(),
            settings=self._state_settings(),
            current_slide=str(self.slide.path) if self.slide else "",
            patch_count=self.bank.stats().total,
            geometry_count=len(self.geometry_bank),
        )

    def load_state_dialog(self) -> None:
        """Restore a saved state over the live data, after confirming."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open State", str(Path.home()),
            f"PathLearn state (*{STATE_SUFFIX});;All files (*)")
        if not path:
            return
        try:
            manifest = read_manifest(path)
        except StateError as exc:
            QMessageBox.critical(self, "Open State", str(exc))
            return

        lines = [
            manifest.summary(),
            "",
            "This replaces:",
            f"  the patch bank ({self.bank.stats().total:,} patches)",
            f"  the geometry bank ({len(self.geometry_bank)} records)",
            "  the class palette",
            "",
            "and rewrites the .geojson beside every slide it carries. Each "
            "sidecar is backed up first.",
            "",
            "Continue?",
        ]
        if QMessageBox.warning(
            self, "Open State", chr(10).join(lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.apply_state(Path(path))

    def apply_state(self, path):
        """Do the restore. Split from the dialog so it is testable."""
        self.close_slide()
        # The bank holds an open SQLite connection and its file is about to be
        # replaced underneath it, so it has to be closed and reopened.
        self.bank.close()
        try:
            report = load_state(
                path,
                patch_bank_path=self.bank.path,
                geometry_bank_path=self.geometry_bank.path,
                equivalences_path=app_data_dir() / "extractor-equivalences.json")
        except StateError as exc:
            self.bank.reopen()
            QMessageBox.critical(self, "Open State", str(exc))
            return None
        finally:
            if not self.bank.is_open:
                self.bank.reopen()

        self.bank_panel.refresh()
        self.geometry_bank.load()
        self.geometry_panel.refresh()

        if report.profile_json:
            try:
                self.profile = ClassificationProfile.from_json(report.profile_json)
                self.sidebar.reload_profile(self.profile)
                self._remember_profile()
            except (ValueError, KeyError, json.JSONDecodeError):
                pass

        for role, document in report.models.items():
            try:
                model = MLClassifier.from_json(document)
            except Exception:                    # noqa: BLE001 - skipped, reported
                continue
            if role == "geometry":
                self.geometry_panel.adopt_model(model)
            else:
                self.classifier = model

        self._apply_state_settings(report.manifest.settings)
        for slide in report.manifest.slides:
            self._note_slide(slide.slide_path)

        if report.missing_slides:
            QMessageBox.information(
                self, "Open State",
                "These slides are not where the state remembers them, so their "
                "annotations were not restored:" + chr(10) + chr(10)
                + chr(10).join(report.missing_slides[:10]))

        self._show_status(report.summary())
        current = report.manifest.current_slide
        if current and Path(current).is_file():
            self.open_slide(Path(current))
        return report

    # -- actions ----------------------------------------------------------

    def _set_tool(self, tool: Tool) -> None:
        self.canvas.set_tool(tool)
        hints = {
            Tool.PAN: "Pan: drag to move, wheel to zoom, double-click to select.",
            Tool.LASSO: "Lasso: drag to trace, or click once and trace hands-free; "
                        "double-click to close the loop, Esc to cancel.",
            Tool.POLYGON: "Polygon: click vertices, double-click (or Enter, or the "
                          "first point) to close, Backspace to undo, Esc to cancel.",
        }
        self._show_status(hints[tool])

    def _toggle_annotations(self, visible: bool) -> None:
        self.canvas.show_annotations = visible
        self.canvas.update()

    def delete_all_annotations(self) -> None:
        if not self.store.annotations:
            return
        count = len(self.store.annotations)
        if QMessageBox.question(
            self, "Delete All Annotations",
            f"Delete all {count} annotation(s) on this slide? This cannot be undone.",
        ) != QMessageBox.StandardButton.Yes:
            return
        self.store.remove_all()
        self.sidebar.refresh()
        self.canvas.update()
        self._show_status(f"Deleted {count} annotation(s).")

    def _zoom_to_annotation(self, annotation_id: uuid.UUID) -> None:
        annotation = self.store.by_id(annotation_id)
        if annotation is not None:
            self.store.selected_id = annotation_id
            self.canvas.zoom_to_annotation(annotation)
            self.canvas.update()

    def _on_canvas_selection(self, annotation_id: object) -> None:
        self.sidebar.select_annotation(annotation_id)
        # Raise the dock only when it is behind another tab: the highlight is
        # useless if you cannot see it, but stealing the tab from someone
        # reading the heatmap would be worse.
        if annotation_id is not None and not self.annotations_dock.isVisible():
            self.annotations_dock.raise_()
        annotation = self.store.selected
        if annotation is not None:
            self._show_status(
                f"{annotation.classification} — {annotation.display_name} "
                f"({annotation.area_short} px²)")

    def rename_class(self, old_name: str) -> str:
        """Rename a class everywhere the name is stamped, not just the palette.

        Returns the status message, so a test can read the outcome without
        going through the status bar.
        """
        patches = self.bank.count(classifications=[old_name])
        geometry = sum(1 for r in self.geometry_bank.records
                       if r.classification == old_name)
        here = sum(1 for a in self.store.annotations
                   if a.classification == old_name)
        others = [Path(p) for p in self._known_slides()
                  if self.slide is None or Path(p) != Path(self.slide.path)]

        sheet = RenameClassSheet(
            old_name, existing_names=[c.name for c in self.profile.classes],
            annotations_here=here, patches=patches, geometry_records=geometry,
            other_slides=others, parent=self)
        if not sheet.exec():
            return "Rename cancelled."

        new_name = sheet.new_name
        done = []

        changed_here = self.store.rename_class(old_name, new_name)
        if changed_here:
            done.append(f"{changed_here} annotation(s) on this slide")

        if sheet.update_patches:
            count = self.bank.rename_class(old_name, new_name)
            if count:
                done.append(f"{count} patch(es)")
                self.bank_panel.refresh()

        if sheet.update_geometry:
            count = self.geometry_bank.rename_class(old_name, new_name)
            if count:
                done.append(f"{count} geometry record(s)")
                self.geometry_panel.annotations_changed()

        failed = []
        slides_done = 0
        annotations_done = 0
        for sidecar in sheet.sidecars_to_update():
            try:
                count = rename_class_in_sidecar(sidecar, old_name, new_name)
            except (OSError, ValueError) as exc:
                # One unreadable sidecar must not abandon the rest half-done.
                log.warning("Could not rename in %s: %s", sidecar, exc)
                failed.append(sidecar.name)
                continue
            if count:
                slides_done += 1
                annotations_done += count
        if slides_done:
            done.append(f"{annotations_done} annotation(s) on "
                        f"{slides_done} other slide(s)")

        self.sidebar.apply_class_rename(old_name, new_name)
        self._remember_profile()
        self.canvas.update()

        verb = "Merged" if sheet.is_merge else "Renamed"
        message = (f"{verb} “{old_name}” to “{new_name}”"
                   + (" — " + ", ".join(done) if done else
                      " — nothing was using it yet") + ".")
        if failed:
            message += (f"  {len(failed)} sidecar(s) could not be updated: "
                        + ", ".join(failed[:3]) + ".")
        self._show_status(message)
        return message

    def _known_slides(self) -> list[str]:
        return [str(p) for p in (self.settings.value("knownSlides", []) or [])]

    # -- machine learning -------------------------------------------------

    def extract_patches(self) -> None:
        if self.slide is None:
            QMessageBox.information(self, "Extract Patches", "Open a slide first.")
            return
        if not [a for a in self.store.annotations if not a.is_subtractive]:
            QMessageBox.information(
                self, "Extract Patches",
                "Draw at least one annotation first — patches are sampled from "
                "inside annotated regions.")
            return
        if self.registry.is_empty:
            self.show_extractors()
            return

        sheet = ExtractPatchesSheet(self.slide, self.store, self.registry,
                                    self.bank, self)
        sheet.exec()
        self.bank_panel.refresh()
        if sheet.report is not None and sheet.report.saved:
            self.bank_dock.raise_()
            self._show_status(sheet.report.summary())

    def train_model(self) -> None:
        if self.bank.stats().is_empty:
            QMessageBox.information(
                self, "Train Model",
                "The patch bank is empty. Extract patches first "
                "(Machine Learning ▸ Extract Patches).")
            return
        sheet = TrainModelSheet(self.bank, self.profile, self)
        sheet.exec()
        if sheet.model is not None:
            self.classifier = sheet.model
            self._show_status(f"Trained {sheet.model.describe()}")

    def predict(self) -> None:
        if self.slide is None:
            QMessageBox.information(self, "Predict", "Open a slide first.")
            return
        if not [a for a in self.store.annotations if not a.is_subtractive]:
            QMessageBox.information(
                self, "Predict",
                "Draw at least one region to predict over. Prediction samples "
                "tiles inside annotations, not across the whole slide.")
            return

        sheet = PredictSheet(self.slide, self.store, self.registry, self.profile,
                             self.classifier, self)
        if sheet.exec() and sheet.result is not None:
            self.classifier = sheet.classifier
            self.canvas.set_predictions(sheet.result)
            self.heatmap_panel.set_predictions(sheet.result)
            self.heatmap_dock.raise_()
            # Straight away rather than debounced: a finished run is the
            # thing worth not losing, and the user may quit right after it.
            self._save_heatmap_now()
            self._show_status(sheet.report.summary())

    def show_tsne(self) -> None:
        if self.bank.stats().is_empty:
            QMessageBox.information(
                self, "t-SNE",
                "The patch bank is empty. Extract patches first, or import a "
                ".bank file from the Patch Bank panel.")
            return
        TSNEWindow(self.bank, self).exec()

    def _on_geometry_model(self, model) -> None:
        """A geometry model stays in the geometry panel.

        It is deliberately NOT adopted as ``self.classifier``: that is the
        model the Predict sheet runs, and Predict tiles a region for an ONNX
        extractor. A geometry model reads a traced outline, so it is applied
        from the panel's Grade button instead.
        """
        self.geometry_dock.raise_()
        self._show_status(f"Geometry model ready — {model.describe()}. "
                          f"Use Grade Annotations to apply it.")

    def _show_grades(self, report) -> None:
        """Open the verdict table for a finished grading run."""
        sheet = GradeResultsSheet(report, self.store, self.profile, self)
        sheet.exec()
        if sheet.applied:
            self.sidebar.refresh()
            self.canvas.update()
            self._show_status(f"Reclassified {sheet.applied} annotation(s).")

    def stitch_predictions(self) -> None:
        """Turn predicted tiles of one class into annotations on this slide."""
        predictions = self.heatmap_panel.predictions
        if predictions is None or predictions.is_empty:
            QMessageBox.information(self, "Stitch Predictions",
                                    "Run a prediction first.")
            return
        if self.slide is None:
            QMessageBox.information(self, "Stitch Predictions", "Open a slide first.")
            return
        sheet = StitchRegionsSheet(predictions, self.store, self)
        if sheet.exec() and sheet.added:
            self.sidebar.refresh()
            self.canvas.update()
            self._show_status(
                f"Added {sheet.added} stitched annotation(s) — {sheet.label}. "
                f"Describe them in the Geometry panel to analyse their shape.")

    def show_composition(self) -> None:
        """Break the predicted tissue down by class, as percentages and areas."""
        predictions = self.heatmap_panel.predictions
        if predictions is None or predictions.is_empty:
            QMessageBox.information(self, "Tissue Composition",
                                    "Run a prediction first.")
            return
        # Deliberately works without an open slide: a prediction set loaded
        # from a sidecar is still worth breaking down. Without the slide there
        # is no pixel size, so the sheet falls back to pixel areas and says so.
        sheet = CompositionSheet(
            predictions, mpp=self._slide_mpp(),
            project_name=self.project.name if self.project else "",
            parent=self)
        sheet.add_to_project_requested.connect(
            lambda: sheet.note(self.add_to_project(quiet=True)))
        sheet.exec()

    def _slide_mpp(self) -> float | None:
        """One micron-per-pixel figure for area, or None if the slide omits it.

        Anisotropic pixels are combined as the geometric mean, which is exact
        for area — the scale factor is mpp_x * mpp_y, and sqrt(xy) squared is
        precisely that. Averaging them would not be.
        """
        if self.slide is None:
            return None
        x, y = self.slide.mpp_x, self.slide.mpp_y
        if x and y:
            return math.sqrt(x * y)
        return x or y or None

    # -- the heatmap sidecar ----------------------------------------------

    def _load_heatmap_for(self, path: Path) -> None:
        """Bring back the heatmap saved beside a slide, if there is one."""
        found = find_sidecar(path)
        if found is None:
            return
        try:
            predictions = PredictionSet.load(found)
        except (OSError, ValueError, EOFError) as exc:
            # A corrupt sidecar must not stop the slide opening. Say so once
            # in the status bar and carry on without a heatmap.
            log.warning("Could not read %s: %s", found, exc)
            self._show_status(f"Could not read {found.name} — "
                              "the slide opened without its heatmap.")
            return
        self.heatmap_panel.set_predictions(predictions)
        self.canvas.set_predictions(predictions)
        self._show_status(f"{predictions.summary()}  (from {found.name})")

    def _schedule_heatmap_save(self) -> None:
        if self.slide is not None and self.heatmap_panel.autosave:
            self._heatmap_save_timer.start()

    def _save_heatmap_now(self) -> bool:
        """Write the heatmap beside the slide. True if a file was written."""
        predictions = self.heatmap_panel.predictions
        if (self.slide is None or not self.heatmap_panel.autosave
                or predictions is None or predictions.is_empty):
            return False
        # Stamp the slide it belongs to, so a sidecar that gets copied
        # somewhere else still says what it was computed on.
        predictions.slide_path = str(self.slide.path)
        target = auto_sidecar_path(self.slide.path)
        try:
            predictions.save(target)
        except OSError as exc:
            log.warning("Could not save %s: %s", target, exc)
            self._show_status(f"Could not save the heatmap: {exc}")
            return False
        return True

    def _on_autosave_toggled(self, enabled: bool) -> None:
        self.settings.setValue("saveHeatmapBesideSlide", enabled)
        if enabled and self._save_heatmap_now():
            self._show_status(
                f"Heatmap saved as {auto_sidecar_path(self.slide.path).name}.")
        elif not enabled:
            self._heatmap_save_timer.stop()
            self._show_status(
                "The heatmap will no longer be saved beside the slide. "
                "Any file already written is left alone.")

    def show_annotation_breakdown(self) -> None:
        """What fraction of the tracing each class accounts for."""
        annotations = list(self.store.annotations)
        if not annotations:
            QMessageBox.information(
                self, "Class Breakdown",
                "There are no annotations on this slide yet.")
            return
        sheet = AnnotationStatsSheet(
            annotations,
            # Subtractive polygons are holes, so they come along even though
            # they are not themselves "in use" — the breakdown needs them to
            # deduct correctly from whatever encloses them.
            in_use=self.store.in_use(include_subtractive=True),
            mpp=self._slide_mpp(),
            project_name=self.project.name if self.project else "",
            parent=self)
        sheet.add_to_project_requested.connect(
            lambda: sheet.note(self.add_annotations_to_project(
                quiet=True, use_only=sheet.use_only.isChecked())))
        sheet.exec()

    # -- projects ---------------------------------------------------------

    def new_project(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New Project", "Project name:", text="PanIN cohort")
        if not ok or not name.strip():
            return
        safe = "".join(c for c in name.strip() if c not in '\\/:*?"<>|')
        path, _ = QFileDialog.getSaveFileName(
            self, "New Project",
            str(default_project_dir() / f"{safe.strip() or 'project'}{PROJECT_SUFFIX}"),
            f"PathLearn project (*{PROJECT_SUFFIX})")
        if not path:
            return
        try:
            project = Project.create(path, name.strip())
        except ProjectError as exc:
            QMessageBox.critical(self, "New Project", str(exc))
            return
        self._adopt_project(project)
        self.show_project_window()

    def open_project(self) -> None:
        start = self.settings.value("lastProjectDir", str(default_project_dir()))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", str(start),
            f"PathLearn project (*{PROJECT_SUFFIX});;JSON (*.json)")
        if not path:
            return
        try:
            project = Project.load(path)
        except ProjectError as exc:
            QMessageBox.critical(self, "Open Project", str(exc))
            return
        self._adopt_project(project)
        self.show_project_window()

    def close_project(self) -> None:
        if self.project is None:
            return
        # Every change is written through as it is made, so there is nothing
        # unsaved to warn about — closing is just letting go of it.
        name = self.project.name
        self.project = None
        if self.project_window is not None:
            self.project_window.close()
            self.project_window = None
        self.settings.setValue("lastProjectPath", "")
        self._update_project_actions()
        self._update_window_title()
        self._show_status(f"Closed the project {name}.")

    def _adopt_project(self, project: Project) -> None:
        self.project = project
        if project.path is not None:
            self.settings.setValue("lastProjectPath", str(project.path))
            self.settings.setValue("lastProjectDir", str(project.path.parent))
        if self.project_window is not None:
            self.project_window.set_project(project)
        self._update_project_actions()
        self._update_window_title()
        self._show_status(project.summary())

    def _reopen_last_project(self) -> None:
        """Re-open last session's project, quietly.

        A project that has been moved or deleted must not raise an error box
        at launch: the user did not ask for it this time, so it simply does
        not open.
        """
        stored = str(self.settings.value("lastProjectPath", "") or "")
        if not stored or not Path(stored).exists():
            return
        try:
            self.project = Project.load(stored)
        except ProjectError as exc:
            log.warning("Could not reopen project %s: %s", stored, exc)
            return
        self._update_project_actions()
        self._update_window_title()

    def _update_project_actions(self) -> None:
        is_open = self.project is not None
        self.action_close_project.setEnabled(is_open)
        self.action_project_table.setEnabled(is_open)
        self.action_add_to_project.setEnabled(is_open)
        self.action_add_annotations_to_project.setEnabled(is_open)

    def _update_window_title(self) -> None:
        title = APP_NAME
        if self.slide is not None:
            title += f" \u2014 {self.slide.name}"
        if self.project is not None:
            title += f"  [{self.project.name}]"
        self.setWindowTitle(title)

    def show_project_window(self) -> None:
        if self.project is None:
            QMessageBox.information(
                self, "Project",
                "No project is open. Project > New Project... to start one.")
            return
        if self.project_window is None:
            self.project_window = ProjectWindow(self.project, self)
            self.project_window.changed.connect(self._save_project)
        else:
            self.project_window.set_project(self.project)
        self.project_window.show()
        self.project_window.raise_()
        self.project_window.activateWindow()

    def add_to_project(self, *, quiet: bool = False) -> str:
        """File the current prediction's breakdown in the open project.

        Returns the message shown to the user so the composition sheet can
        report the outcome in place, rather than stacking a dialog on a dialog.
        """
        def refuse(message: str, warn: bool = False) -> str:
            if not quiet:
                box = QMessageBox.warning if warn else QMessageBox.information
                box(self, "Add to Project", message)
            return message

        if self.project is None:
            return refuse(
                "No project is open. Project > New Project... to start one.")

        predictions = self.heatmap_panel.predictions
        if predictions is None or predictions.is_empty:
            return refuse("Run a prediction first - there is nothing to add.")

        report = composition(predictions, mpp=self._slide_mpp())
        if report.is_empty:
            return refuse("No tiles passed the confidence threshold, so there "
                          "is no breakdown to file.")

        slide_path = (str(self.slide.path) if self.slide is not None
                      else predictions.slide_path)
        if not slide_path:
            return refuse("These predictions do not name a slide, so they "
                          "cannot be filed against one.", warn=True)

        existing = self.project.entry_for(slide_path)
        replace = True
        if existing is not None:
            # Two runs of one slide is a legitimate thing to want (comparing
            # models), so this asks rather than assuming either way.
            choice = QMessageBox.question(
                self, "Slide already in the project",
                f"{existing.slide_name} is already in this project, added "
                f"{existing.added}.\n\nReplace that entry with this run?\n\n"
                "Yes replaces it. No keeps both rows.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if choice == QMessageBox.StandardButton.Cancel:
                return "Not added."
            replace = choice == QMessageBox.StandardButton.Yes

        entry = ProjectEntry.from_report(
            report, predictions, slide_path=slide_path,
            model_name=self._model_name())
        replaced = self.project.add(entry, replace=replace)
        if not self._save_project():
            return "Could not save the project - the entry was not kept."

        message = (f"{'Replaced' if replaced else 'Added'} {entry.slide_name} "
                   f"- {len(self.project)} slide(s) in the project. "
                   f"{report.summary()}")
        self._show_status(message)
        if self.project_window is not None:
            self.project_window.reload()
        return message

    def add_annotations_to_project(self, *, quiet: bool = False,
                                   use_only: bool = False) -> str:
        """File what you drew on this slide as its own project row.

        Separate from the predicted row rather than merged with it: the two
        measure different things over different denominators, and a project
        that conflated them would invite averaging one into the other.
        """
        def refuse(message: str, warn: bool = False) -> str:
            if not quiet:
                box = QMessageBox.warning if warn else QMessageBox.information
                box(self, "Add to Project", message)
            return message

        if self.project is None:
            return refuse(
                "No project is open. Project > New Project... to start one.")
        if self.slide is None:
            return refuse("Open a slide first.")

        annotations = (self.store.in_use(include_subtractive=True)
                       if use_only else list(self.store.annotations))
        if not annotations:
            return refuse("There are no annotations on this slide to file.")

        scope = ("annotations checked under Use" if use_only
                 else "all annotations")
        report = annotation_breakdown(annotations, mpp=self._slide_mpp(),
                                      scope=scope)
        if report.is_empty:
            return refuse(
                "Nothing to file — every annotation here is a subtractive "
                "polygon, which carves area out rather than being a region.")

        slide_path = str(self.slide.path)
        existing = self.project.entry_for(slide_path, ANNOTATED)
        replace = True
        if existing is not None:
            choice = QMessageBox.question(
                self, "Slide already in the project",
                f"An annotation breakdown for {existing.slide_name} is "
                f"already in this project, added {existing.added}.\n\n"
                "Replace that row with this one?\n\n"
                "Yes replaces it. No keeps both rows.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if choice == QMessageBox.StandardButton.Cancel:
                return "Not added."
            replace = choice == QMessageBox.StandardButton.Yes

        entry = ProjectEntry.from_breakdown(
            report, slide_path=slide_path,
            # The palette in force is what makes a class you drew none of a
            # real zero rather than a blank.
            palette_classes=[c.name for c in self.profile.classes])
        replaced = self.project.add(entry, replace=replace)
        if not self._save_project():
            return "Could not save the project - the row was not kept."

        message = (f"{'Replaced' if replaced else 'Added'} the annotation "
                   f"breakdown for {entry.slide_name} - "
                   f"{len(self.project)} row(s) in the project. "
                   f"{report.summary()}")
        self._show_status(message)
        if self.project_window is not None:
            self.project_window.reload()
        return message

    def _model_name(self) -> str:
        """A human label for whatever produced the current predictions."""
        if self.classifier is None:
            return ""
        labels = getattr(self.classifier, "class_labels", []) or []
        return f"{len(labels)}-class model"

    def _save_project(self) -> bool:
        if self.project is None:
            return False
        try:
            self.project.save()
        except ProjectError as exc:
            QMessageBox.critical(self, "Project", str(exc))
            return False
        return True

    def _on_heatmap_changed(self) -> None:
        self.canvas.show_heatmap = self.heatmap_panel.show_heatmap
        self.canvas.set_predictions(self.heatmap_panel.predictions)
        # The confidence threshold and the hidden classes are part of the
        # stored result, so a change to either makes the sidecar stale.
        self._schedule_heatmap_save()

    def show_extractors(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Installed Extractors")
        if self.registry.is_empty:
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText("No feature extractors are installed.")
            box.setInformativeText(
                "Drop a model file and its matching .pathlearn-extractor.json into "
                "the folder below, then Rescan.\n\n"
                f"{self.registry.directories[0]}\n\n"
                "tools/convert_extractor.py builds both from a HuggingFace "
                "model. Geometry models need no extractor at all.")
        else:
            box.setText(f"{len(self.registry)} extractor(s) installed.")
            box.setInformativeText(self.registry.describe() + "\n\n"
                                   + self._provider_summary())
        box.setDetailedText("Searched:\n" +
                            "\n".join(str(d) for d in self.registry.directories))
        open_folder = box.addButton("Open Folder", QMessageBox.ButtonRole.ActionRole)
        rescan = box.addButton("Rescan", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is open_folder:
            self.open_extractors_folder()
            self.show_extractors()
        elif box.clickedButton() is rescan:
            self.registry.rescan()
            self.show_extractors()

    def open_extractors_folder(self) -> Path:
        """Reveal the extractors folder, creating it if it does not exist yet.

        Creating it matters: on a fresh install the folder is absent, so
        "drop your files in %LOCALAPPDATA%\\PathLearn\\Extractors" sends the
        user somewhere that is not there.
        """
        folder = self.registry.directories[0]
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        return folder

    def _provider_summary(self) -> str:
        """Which ONNX Runtime execution provider extraction will actually use.

        Worth stating in the UI because the failure is silent: with the CPU
        build of onnxruntime, or a GPU build missing its CUDA runtime, sessions
        are created successfully and simply run ~20x slower.
        """
        try:
            from ..extractors.onnx_extractor import available_providers, best_provider
        except ImportError:
            return "ONNX Runtime is not installed, so extraction cannot run."
        try:
            offered = available_providers()
            active = best_provider()
        except Exception as exc:                     # noqa: BLE001 - informational
            return f"Could not query ONNX Runtime providers: {exc}"
        line = f"Execution provider: {active}"
        if active == "CPUExecutionProvider":
            line += ("  — CPU only. Install onnxruntime-gpu (and a matching "
                     "CUDA runtime) for GPU inference.")
        return line + f"\nAvailable: {', '.join(offered)}"

    def convert_model(self) -> None:
        """Build an extractor from a HuggingFace model, in a side environment."""
        sheet = ConvertModelSheet(self.registry.directories[0], self)
        sheet.converted.connect(self._on_model_converted)
        sheet.exec()

    def _on_model_converted(self) -> None:
        self.registry.rescan()
        self._show_status(f"Extractors rescanned — {len(self.registry)} installed.")

    def show_help(self, topic: str | None = None) -> "HelpWindow":
        """Open the manual, reusing the window if it is already up.

        Kept on the instance rather than created fresh so that scroll position
        and search text survive flipping back to the app — the manual is meant
        to be read alongside the thing it describes.
        """
        existing = getattr(self, "_help_window", None)
        if existing is None:
            existing = HelpWindow(self)
            self._help_window = existing
        if topic:
            existing.show_topic(topic)
        existing.show()
        existing.raise_()
        existing.activateWindow()
        return existing

    def show_slide_properties(self) -> None:
        if self.slide is None:
            QMessageBox.information(self, "Slide Properties", "No slide open.")
            return
        slide = self.slide
        lines = [
            f"File: {slide.path}",
            f"Dimensions (level 0): {slide.dimensions.width} × {slide.dimensions.height} px",
            f"Levels: {slide.level_count}",
            f"Downsamples: {', '.join(f'{d:.1f}' for d in slide.level_downsamples)}",
            f"MPP: {slide.mpp_x or '—'} × {slide.mpp_y or '—'} µm/px",
        ]
        box = QMessageBox(self)
        box.setWindowTitle("Slide Properties")
        box.setText("\n".join(lines))
        box.setDetailedText("\n".join(f"{k} = {v}" for k, v in sorted(slide.properties.items())))
        box.exec()

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About PathLearn",
            "<b>PathLearn</b> (Windows)<br><br>"
            "Whole-slide-image annotation and machine-learning workbench "
            "for digital pathology.<br><br>"
            "Coordinates: level-0 pixels, top-left origin, Y down — the same "
            "space as OpenSlide and QuPath, with no mirroring.<br><br>"
            "<i>Research tooling. Not a medical device; not for diagnostic use.</i>"
        )

    # -- status -----------------------------------------------------------

    def _on_store_changed(self) -> None:
        self.canvas.update()
        # Every store mutation funnels through here — add, delete, import,
        # tick, untick — so this is the one place that keeps panels derived
        # from the annotation set honest.
        self.geometry_panel.annotations_changed()

    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 6000)

    def _update_status(self) -> None:
        if self.slide is None:
            self.status_slide.setText("No slide")
            self.status_position.setText("")
            self.status_zoom.setText("")
            return
        dims = self.slide.dimensions
        self.status_slide.setText(f"{self.slide.name}  ({dims.width}×{dims.height})")
        centre = self.canvas.center
        self.status_position.setText(f"x={centre.x():.0f}  y={centre.y():.0f}")
        self.status_zoom.setText(
            f"{self.canvas.magnification_text}  ·  level {self.canvas.current_level}"
        )

    def _update_actions_enabled(self) -> None:
        has_slide = self.slide is not None
        for action in (self.action_close, self.action_import, self.action_export,
                       self.action_export_images,
                       self.action_reveal, self.action_zoom_in, self.action_zoom_out,
                       self.action_zoom_fit, self.action_slide_info,
                       self.action_delete_all, self.action_lasso, self.action_polygon,
                       self.action_extract, self.action_predict):
            action.setEnabled(has_slide)

    # -- shutdown ---------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._remember_profile()
        self.canvas.shutdown()
        self.geometry_panel.shutdown()
        self.registry.close()
        self.bank.close()
        if self.slide is not None:
            self.store.save()
            self.slide.close()
        event.accept()
