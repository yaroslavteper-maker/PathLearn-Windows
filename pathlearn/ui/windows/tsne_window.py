"""t-SNE plot window — from ``Views/TSNEPlotWindow.swift``.

Embeds the patch bank in 2-D so the feature space can be looked at rather than
only scored.

COLOUR BY SLIDE IS NOT A SIDE FEATURE
=====================================
Measured during development on a real bank, a UNI2 embedding predicted **which
slide** a patch came from *more* reliably than **what tissue** it shows, and
grade accuracy fell by more than half when whole slides were held out instead
of stratifying. Colouring the same embedding by slide rather than class makes
that visible immediately: if the two colourings produce the same cluster
structure, the model is learning slides.

So the colour control is placed beside the plot rather than buried, and the
window says what it found.

WHAT THIS VIEW CANNOT TELL YOU
==============================
t-SNE distances are not meaningful globally. Visual separation neither
guarantees nor precludes separability in the original space — a linear
classifier routinely separates what t-SNE tangles. Use it to spot structure,
mislabels and batch effects, never to decide that a classifier will work.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ...core.tsne import TSNEConfig, TSNEError, embed, stratified_subsample
from ...data.bank import PatchBank
from ..workers import BackgroundTask

#: Distinguishable at small point sizes on a dark ground.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (128, 128, 0), (170, 255, 195),
]


class TSNEPlot(QWidget):
    """Scatter of a 2-D embedding, coloured by a chosen grouping."""

    point_hovered = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 360)
        self.setMouseTracking(True)
        self.points: np.ndarray | None = None
        self.groups: list[str] = []
        self.show_centroids = True
        self._hover = -1

    def set_data(self, points: np.ndarray | None, groups: list[str]) -> None:
        self.points = points
        self.groups = list(groups)
        self._hover = -1
        self.update()

    @property
    def colours(self) -> dict[str, tuple[int, int, int]]:
        order = [name for name, _ in Counter(self.groups).most_common()]
        return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(order)}

    def _transform(self) -> tuple[float, float, float, float, float]:
        """Scale and offset mapping embedding coords into the widget."""
        assert self.points is not None
        low = self.points.min(axis=0)
        high = self.points.max(axis=0)
        span = np.maximum(high - low, 1e-9)
        margin = 24.0
        legend = 160.0
        width = max(self.width() - legend - 2 * margin, 50.0)
        height = max(self.height() - 2 * margin, 50.0)
        scale = min(width / span[0], height / span[1])
        # Centre the cloud in the plotting area rather than stretching it, so
        # the aspect ratio of the embedding is preserved.
        ox = margin + (width - span[0] * scale) / 2 - low[0] * scale
        oy = margin + (height - span[1] * scale) / 2 - low[1] * scale
        return scale, ox, oy, low[0], low[1]

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 27))
        if self.points is None or len(self.points) == 0:
            painter.setPen(QColor(150, 150, 155))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No embedding yet — press Run.")
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale, ox, oy, _, _ = self._transform()
        colours = self.colours

        painter.setPen(Qt.PenStyle.NoPen)
        for index, ((x, y), group) in enumerate(zip(self.points, self.groups)):
            r, g, b = colours.get(group, (200, 200, 200))
            painter.setBrush(QColor(r, g, b, 110 if index == self._hover else 190))
            radius = 5.0 if index == self._hover else 2.8
            painter.drawEllipse(QPointF(x * scale + ox, y * scale + oy), radius, radius)

        if self.show_centroids:
            self._paint_centroids(painter, scale, ox, oy, colours)
        self._paint_legend(painter, colours)

    def _paint_centroids(self, painter, scale, ox, oy, colours) -> None:
        for name in sorted(set(self.groups)):
            mask = [i for i, g in enumerate(self.groups) if g == name]
            if not mask:
                continue
            centre = self.points[mask].mean(axis=0)
            r, g, b = colours.get(name, (200, 200, 200))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2.0))
            point = QPointF(centre[0] * scale + ox, centre[1] * scale + oy)
            painter.drawEllipse(point, 7.0, 7.0)
            painter.setPen(QPen(QColor(r, g, b), 2.0))
            painter.drawEllipse(point, 5.0, 5.0)

    def _paint_legend(self, painter, colours) -> None:
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        counts = Counter(self.groups)
        x = self.width() - 150
        for i, (name, count) in enumerate(counts.most_common(18)):
            y = 26 + i * 15
            r, g, b = colours.get(name, (200, 200, 200))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(r, g, b))
            painter.drawEllipse(QPointF(x, y - 3), 4, 4)
            painter.setPen(QColor(215, 215, 220))
            painter.drawText(int(x + 10), int(y), f"{name[:18]} ({count})")
        if len(counts) > 18:
            painter.setPen(QColor(150, 150, 155))
            painter.drawText(int(x), int(26 + 18 * 15), f"+{len(counts) - 18} more")

    def mouseMoveEvent(self, event) -> None:
        if self.points is None or len(self.points) == 0:
            return
        scale, ox, oy, _, _ = self._transform()
        position = event.position()
        screen = self.points * scale + np.array([ox, oy])
        distances = np.hypot(screen[:, 0] - position.x(), screen[:, 1] - position.y())
        nearest = int(np.argmin(distances))
        hover = nearest if distances[nearest] < 8.0 else -1
        if hover != self._hover:
            self._hover = hover
            self.point_hovered.emit(hover)
            self.update()


class TSNEWindow(QDialog):
    """Configure and run a t-SNE over the patch bank, then look at it."""

    def __init__(self, bank: PatchBank, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("t-SNE — feature space")
        self.resize(1050, 720)
        self.bank = bank
        self.task: BackgroundTask | None = None
        self.patches: list = []
        self.points: np.ndarray | None = None

        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        body.addWidget(self._build_controls(), 0)
        self.plot = TSNEPlot()
        self.plot.point_hovered.connect(self._on_hover)
        body.addWidget(self.plot, 1)
        layout.addLayout(body, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.finding = QLabel("")
        self.finding.setWordWrap(True)
        self.finding.setStyleSheet("color: #d08a20;")
        self.finding.setVisible(False)
        layout.addWidget(self.finding)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_sources()

    # -- construction -----------------------------------------------------

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(260)
        outer = QVBoxLayout(panel)

        box = QGroupBox("Embedding")
        form = QFormLayout(box)

        self.extractor_combo = QComboBox()
        form.addRow("Feature space", self.extractor_combo)

        self.max_points = QSpinBox()
        self.max_points.setRange(50, 20_000)
        self.max_points.setSingleStep(250)
        self.max_points.setValue(1500)
        self.max_points.setToolTip(
            "Patches are stratified-subsampled to this many, keeping class "
            "proportions. t-SNE is O(n²) per iteration, so large banks are slow.")
        form.addRow("Max points", self.max_points)

        self.perplexity = QSpinBox()
        self.perplexity.setRange(2, 200)
        self.perplexity.setValue(30)
        self.perplexity.setToolTip(
            "Roughly the number of neighbours each point balances. Must satisfy "
            "perplexity * 3 < point count.")
        form.addRow("Perplexity", self.perplexity)

        self.iterations = QSpinBox()
        self.iterations.setRange(100, 5000)
        self.iterations.setSingleStep(100)
        self.iterations.setValue(750)
        form.addRow("Iterations", self.iterations)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._start)
        form.addRow("", self.run_button)
        outer.addWidget(box)

        display = QGroupBox("Display")
        display_form = QFormLayout(display)
        self.colour_by = QComboBox()
        self.colour_by.addItem("Class", userData="class")
        self.colour_by.addItem("Slide  (batch effect)", userData="slide")
        self.colour_by.setToolTip(
            "If colouring by slide reproduces the same clusters as colouring by "
            "class, the embedding encodes which slide a patch came from and any "
            "classifier trained on it will partly be recognising slides.")
        self.colour_by.currentIndexChanged.connect(self._recolour)
        display_form.addRow("Colour by", self.colour_by)

        self.centroids = QPushButton("Hide centroids")
        self.centroids.setCheckable(True)
        self.centroids.setChecked(False)
        self.centroids.toggled.connect(self._toggle_centroids)
        display_form.addRow("", self.centroids)
        outer.addWidget(display)

        self.hover_label = QLabel("")
        self.hover_label.setWordWrap(True)
        self.hover_label.setStyleSheet("color: #999;")
        outer.addWidget(self.hover_label)
        outer.addStretch(1)
        return panel

    def _refresh_sources(self) -> None:
        self.extractor_combo.clear()
        stats = self.bank.stats()
        for identity in sorted(stats.by_extractor):
            self.extractor_combo.addItem(
                f"{identity}  ({stats.by_extractor[identity]} patches)",
                userData=identity)
        if not stats.by_extractor:
            self.extractor_combo.addItem("Bank is empty — extract patches first",
                                         userData=None)
            self.extractor_combo.setEnabled(False)
            self.run_button.setEnabled(False)
            self.status.setText("Nothing to embed.")

    # -- running ----------------------------------------------------------

    def _start(self) -> None:
        identity = self.extractor_combo.currentData()
        if identity is None:
            return
        patches = self.bank.fetch(extractor_identity=identity)
        if len(patches) < 10:
            self.status.setText(
                f"Only {len(patches)} patches for {identity}; t-SNE needs at least 10.")
            return

        labels = [p.classification for p in patches]
        keep = stratified_subsample(labels, self.max_points.value(), seed=42)
        self.patches = [patches[i] for i in keep]
        matrix = np.stack([p.features for p in self.patches]).astype(np.float32)

        config = TSNEConfig(perplexity=float(self.perplexity.value()),
                            iterations=self.iterations.value(), seed=42)
        self._set_running(True)
        self.task = BackgroundTask(_embed_job, matrix=matrix, config=config)
        self.task.worker.progress.connect(self._on_progress)
        self.task.worker.finished.connect(self._on_finished)
        self.task.worker.failed.connect(self._on_failed)
        self.task.start()

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.progress.setValue(0)
        for widget in (self.run_button, self.extractor_combo, self.max_points,
                       self.perplexity, self.iterations):
            widget.setEnabled(not running)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status.setText(message)

    def _on_finished(self, points) -> None:
        self.task = None
        self._set_running(False)
        if points is None:
            return
        self.points = points
        self._recolour()
        self._report_batch_effect()

    def _on_failed(self, message: str) -> None:
        self.task = None
        self._set_running(False)
        self.status.setText(message)

    # -- display ----------------------------------------------------------

    def _grouping(self) -> list[str]:
        if self.colour_by.currentData() == "slide":
            return [p.slide_name for p in self.patches]
        return [p.classification for p in self.patches]

    def _recolour(self) -> None:
        if self.points is None:
            return
        groups = self._grouping()
        self.plot.set_data(self.points, groups)
        kind = "slides" if self.colour_by.currentData() == "slide" else "classes"
        self.status.setText(
            f"{len(self.points)} patches · {len(set(groups))} {kind}")

    def _toggle_centroids(self, hidden: bool) -> None:
        self.plot.show_centroids = not hidden
        self.centroids.setText("Show centroids" if hidden else "Hide centroids")
        self.plot.update()

    def _on_hover(self, index: int) -> None:
        if index < 0 or index >= len(self.patches):
            self.hover_label.setText("")
            return
        patch = self.patches[index]
        self.hover_label.setText(
            f"{patch.classification}\n{patch.slide_name}\n"
            f"({patch.patch_x}, {patch.patch_y})  white {patch.white_fraction:.2f}")

    def _report_batch_effect(self) -> None:
        """Report how recoverable class and slide are **from the features**.

        Deliberately *not* measured on the 2-D layout. A spread ratio over
        t-SNE output is untrustworthy for this: it compares groupings with
        different group counts (15 slides against 10 classes), and t-SNE is a
        lossy non-linear projection whose geometry does not represent the space
        a classifier actually works in. An early version did exactly that and
        reported the opposite of the truth.

        Instead a small linear probe is fitted on the sampled feature vectors —
        the same thing a classifier would see. Accuracy is quoted against each
        grouping's majority baseline, since 15 slides and 10 classes have very
        different chance levels and the raw numbers are not comparable.
        """
        if not self.patches:
            return
        matrix = np.stack([p.features for p in self.patches]).astype(np.float32)
        by_class = _probe(matrix, [p.classification for p in self.patches])
        by_slide = _probe(matrix, [p.slide_name for p in self.patches])
        if by_class is None or by_slide is None:
            self.finding.setVisible(False)
            return

        class_acc, class_base = by_class
        slide_acc, slide_base = by_slide
        class_lift = class_acc - class_base
        slide_lift = slide_acc - slide_base

        summary = (f"Recoverable from these features — "
                   f"class {class_acc:.0%} (baseline {class_base:.0%}), "
                   f"slide {slide_acc:.0%} (baseline {slide_base:.0%}).")
        if slide_lift >= class_lift:
            self.finding.setText(
                summary + " Slide identity is at least as recoverable as tissue "
                "class, so a classifier trained here will partly be recognising "
                "slides. Validate by holding out whole slides, not random patches.")
        else:
            self.finding.setText(
                summary + " Tissue class is more recoverable than slide identity, "
                "which is the right way round — though holding out whole slides "
                "is still the honest validation.")
        self.finding.setVisible(True)

    def done(self, result: int) -> None:
        if self.task is not None and self.task.is_running:
            self.task.cancel()
            self.task.wait(20_000)
        super().done(result)


def _embed_job(matrix: np.ndarray, config: TSNEConfig, progress=None,
               should_cancel=None) -> np.ndarray:
    """Runs on the worker thread; t-SNE takes seconds to minutes."""
    return embed(matrix, config,
                 progress=lambda d, t, m: progress(d, t, m) if progress else None)


def _probe(features: np.ndarray, groups: list[str], folds: int = 3,
           seed: int = 0) -> tuple[float, float] | None:
    """Cross-validated linear-probe accuracy for recovering *groups*, and its
    majority baseline.

    Small and fast on purpose — this is a diagnostic beside a plot, not a
    modelling step. Groups with fewer members than folds are dropped, since
    they cannot be validated.
    """
    from ...core.logistic import train as train_logistic, zscore_apply, zscore_fit

    counts = Counter(groups)
    usable = [i for i, g in enumerate(groups) if counts[g] >= folds]
    if len(usable) < folds * 2:
        return None
    names = sorted({groups[i] for i in usable})
    if len(names) < 2:
        return None

    index = {n: i for i, n in enumerate(names)}
    y = np.array([index[groups[i]] for i in usable], dtype=np.intp)
    X = features[usable]
    mean, std = zscore_fit(X)
    Z = zscore_apply(X, mean, std)

    rng = np.random.default_rng(seed)
    assignment = np.zeros(len(y), dtype=int)
    for k in np.unique(y):
        where = np.flatnonzero(y == k)
        rng.shuffle(where)
        for position, i in enumerate(where):
            assignment[i] = position % folds

    correct = total = 0
    for fold in range(folds):
        train_idx = np.flatnonzero(assignment != fold)
        val_idx = np.flatnonzero(assignment == fold)
        if val_idx.size == 0 or len(np.unique(y[train_idx])) < 2:
            continue
        model = train_logistic(Z[train_idx], y[train_idx], len(names),
                               iterations=120, l2=0.05)
        predicted = np.argmax(model.predict_proba(Z[val_idx]), axis=1)
        correct += int((predicted == y[val_idx]).sum())
        total += val_idx.size
    if total == 0:
        return None
    return correct / total, max(Counter(y.tolist()).values()) / len(y)


def _separation(points: np.ndarray, groups: list[str]) -> float:
    """Between-group spread over within-group spread, in the 2-D layout.

    Kept for the plot only. NOT comparable between groupings with different
    group counts — see _report_batch_effect for why that matters.
    """
    names = sorted(set(groups))
    if len(names) < 2:
        return 0.0
    centres, spreads = [], []
    for name in names:
        member = points[[i for i, g in enumerate(groups) if g == name]]
        if len(member) == 0:
            continue
        centres.append(member.mean(axis=0))
        spreads.append(float(np.linalg.norm(member - member.mean(axis=0), axis=1).mean()))
    if len(centres) < 2:
        return 0.0
    centres = np.stack(centres)
    between = float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).mean())
    within = float(np.mean(spreads)) or 1e-9
    return between / within
