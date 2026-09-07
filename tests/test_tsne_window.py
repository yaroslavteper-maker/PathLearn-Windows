"""t-SNE window: embedding, recolouring, and the batch-effect report."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from pathlearn.data.bank import Patch, PatchBank
from pathlearn.ui.windows.tsne_window import TSNEPlot, TSNEWindow, _separation

DIM = 8


def patch(classification, slide, vector):
    return Patch(slide_path=f"E:/{slide}", slide_name=slide,
                 annotation_id=uuid.uuid4(), classification=classification,
                 patch_x=0, patch_y=0, patch_level=0, patch_size_level=224,
                 features=np.asarray(vector, dtype=np.float32),
                 extractor_identity="onnx:test:r1", white_fraction=0.1)


@pytest.fixture
def bank(tmp_path):
    with PatchBank(tmp_path / "b.db") as b:
        yield b


def fill(bank, per_group=30, class_driven=True):
    """Embeddings driven either by class or by slide, so the report is testable."""
    rng = np.random.default_rng(0)
    rows = []
    for ci, cls in enumerate(("Acinar", "PanIN-3")):
        for si, slide in enumerate(("a.svs", "b.svs")):
            centre = np.zeros(DIM)
            centre[0] = (ci if class_driven else si) * 10.0
            for _ in range(per_group):
                rows.append(patch(cls, slide, centre + rng.normal(0, 0.3, DIM)))
    bank.add_many(rows)


@pytest.fixture
def window(qtbot, bank):
    w = TSNEWindow(bank)
    qtbot.addWidget(w)
    yield w
    w.reject()


class TestSeparation:
    def test_well_separated_groups_score_high(self):
        points = np.array([[0.0, 0.0]] * 20 + [[10.0, 0.0]] * 20)
        assert _separation(points, ["a"] * 20 + ["b"] * 20) > 5

    def test_interleaved_groups_score_low(self):
        rng = np.random.default_rng(0)
        points = rng.normal(0, 1, (40, 2))
        assert _separation(points, ["a", "b"] * 20) < 1

    def test_single_group_is_zero(self):
        assert _separation(np.zeros((10, 2)), ["a"] * 10) == 0.0


class TestPlotWidget:
    def test_handles_no_data(self, qtbot):
        plot = TSNEPlot()
        qtbot.addWidget(plot)
        plot.resize(400, 300)
        plot.grab()          # must not raise

    def test_renders_points(self, qtbot):
        plot = TSNEPlot()
        qtbot.addWidget(plot)
        plot.resize(400, 300)
        rng = np.random.default_rng(0)
        plot.set_data(rng.normal(0, 1, (50, 2)), ["a"] * 25 + ["b"] * 25)
        assert not plot.grab().isNull()

    def test_colours_are_stable_per_group(self, qtbot):
        plot = TSNEPlot()
        qtbot.addWidget(plot)
        plot.set_data(np.zeros((4, 2)), ["a", "b", "a", "b"])
        first = plot.colours
        plot.set_data(np.zeros((4, 2)), ["a", "b", "a", "b"])
        assert plot.colours == first

    def test_many_groups_do_not_overflow_the_palette(self, qtbot):
        plot = TSNEPlot()
        qtbot.addWidget(plot)
        plot.resize(500, 400)
        groups = [f"s{i}" for i in range(40)]
        plot.set_data(np.random.default_rng(0).normal(0, 1, (40, 2)), groups)
        assert len(plot.colours) == 40
        assert not plot.grab().isNull()


class TestWindow:
    def test_empty_bank_disables_run(self, qtbot, bank):
        w = TSNEWindow(bank)
        qtbot.addWidget(w)
        assert not w.run_button.isEnabled()
        w.reject()

    def test_lists_feature_spaces(self, window, bank):
        fill(bank)
        window._refresh_sources()
        assert window.extractor_combo.currentData() == "onnx:test:r1"

    def test_too_few_patches_is_explained(self, window, bank):
        bank.add_many([patch("A", "a.svs", np.zeros(DIM)) for _ in range(5)])
        window._refresh_sources()
        window._start()
        assert "at least 10" in window.status.text()

    def test_embeds_and_plots(self, window, bank, qtbot):
        fill(bank)
        window._refresh_sources()
        window.iterations.setValue(150)
        window.perplexity.setValue(10)
        window._start()
        qtbot.waitUntil(lambda: window.task is None, timeout=180_000)
        assert window.points is not None
        assert window.points.shape == (len(window.patches), 2)

    def test_recolour_switches_grouping(self, window, bank, qtbot):
        fill(bank)
        window._refresh_sources()
        window.iterations.setValue(120)
        window.perplexity.setValue(10)
        window._start()
        qtbot.waitUntil(lambda: window.task is None, timeout=180_000)

        assert set(window._grouping()) == {"Acinar", "PanIN-3"}
        window.colour_by.setCurrentIndex(1)
        assert set(window._grouping()) == {"a.svs", "b.svs"}

    def test_reports_class_driven_structure(self, window, bank, qtbot):
        """When classes explain the layout, say so."""
        fill(bank, class_driven=True)
        window._refresh_sources()
        window.iterations.setValue(250)
        window.perplexity.setValue(10)
        window._start()
        qtbot.waitUntil(lambda: window.task is None, timeout=180_000)
        assert window.finding.isVisibleTo(window)
        assert "Tissue class is more recoverable" in window.finding.text()

    def test_reports_slide_driven_structure(self, window, bank, qtbot):
        """When slides explain the layout, warn — this is the batch effect."""
        fill(bank, class_driven=False)
        window._refresh_sources()
        window.iterations.setValue(250)
        window.perplexity.setValue(10)
        window._start()
        qtbot.waitUntil(lambda: window.task is None, timeout=180_000)
        assert window.finding.isVisibleTo(window)
        assert "recognising slides" in window.finding.text()

    def test_perplexity_too_large_is_reported_not_raised(self, window, bank, qtbot):
        fill(bank, per_group=4)
        window._refresh_sources()
        window.perplexity.setValue(100)
        window._start()
        qtbot.waitUntil(lambda: window.task is None, timeout=180_000)
        assert "perplexity" in window.status.text().lower()

    def test_hover_reports_the_patch(self, window, bank, qtbot):
        fill(bank)
        window._refresh_sources()
        window.iterations.setValue(120)
        window.perplexity.setValue(10)
        window._start()
        qtbot.waitUntil(lambda: window.task is None, timeout=180_000)
        window._on_hover(0)
        assert window.patches[0].slide_name in window.hover_label.text()
        window._on_hover(-1)
        assert window.hover_label.text() == ""


class TestProbe:
    """The batch-effect verdict must come from the features, not the layout."""

    def test_recovers_a_perfectly_predictable_grouping(self):
        from pathlearn.ui.windows.tsne_window import _probe
        rng = np.random.default_rng(0)
        X = np.concatenate([rng.normal(0, 0.2, (40, DIM)),
                            rng.normal(8, 0.2, (40, DIM))]).astype(np.float32)
        accuracy, baseline = _probe(X, ["a"] * 40 + ["b"] * 40)
        assert accuracy > 0.95 and baseline == pytest.approx(0.5)

    def test_unpredictable_grouping_scores_near_baseline(self):
        from pathlearn.ui.windows.tsne_window import _probe
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (80, DIM)).astype(np.float32)
        accuracy, baseline = _probe(X, ["a", "b"] * 40)
        assert accuracy < baseline + 0.25

    def test_reports_baseline_so_group_counts_are_comparable(self):
        """15 slides and 10 classes have very different chance levels."""
        from pathlearn.ui.windows.tsne_window import _probe
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (90, DIM)).astype(np.float32)
        _, baseline = _probe(X, [f"g{i % 9}" for i in range(90)])
        assert baseline == pytest.approx(1 / 9, abs=0.02)

    def test_too_little_data_returns_none(self):
        from pathlearn.ui.windows.tsne_window import _probe
        assert _probe(np.zeros((4, DIM), dtype=np.float32), ["a", "a", "b", "b"]) is None

    def test_single_group_returns_none(self):
        from pathlearn.ui.windows.tsne_window import _probe
        assert _probe(np.zeros((30, DIM), dtype=np.float32), ["a"] * 30) is None
