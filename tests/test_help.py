"""The manual: the window, and whether its claims still match the code.

The second part matters more. A manual that describes a control that was
renamed or removed is worse than no manual, and nothing else in the suite
would notice.
"""

from __future__ import annotations

import pytest

from pathlearn.ui.help_content import TOPICS, Topic, as_html, plain_text, search
from pathlearn.ui.windows.help_window import HelpWindow


class TestContent:
    def test_every_topic_has_a_title_and_body(self):
        for topic in TOPICS:
            assert topic.title.strip()
            assert len(plain_text(topic)) > 200, f"{topic.title} is a stub"

    def test_titles_are_unique(self):
        titles = [t.title for t in TOPICS]
        assert len(titles) == len(set(titles))

    def test_every_topic_opens_with_a_heading(self):
        for topic in TOPICS:
            assert topic.html.lstrip().startswith("<h2>"), topic.title

    def test_html_tags_are_balanced(self):
        """A stray '<' would swallow the rest of a topic when rendered."""
        for topic in TOPICS:
            assert topic.html.count("<") == topic.html.count(">"), topic.title

    def test_as_html_carries_the_stylesheet(self):
        assert "<style>" in as_html(TOPICS[0])

    @pytest.mark.parametrize("feature", [
        "Lasso", "Polygon", "subtractive", "Use All", "Delete Checked",
        "null class", "profile", "GeoJSON", "Extract Patches", "patch bank",
        "t-SNE", "Predict", "heatmap", "Geometry", "Grade", "shortcut",
        "extractor", "batch effect",
    ])
    def test_the_manual_covers(self, feature):
        haystack = " ".join(plain_text(t) + t.title for t in TOPICS).lower()
        assert feature.lower() in haystack, f"nothing documents {feature}"


class TestSearch:
    def test_empty_query_returns_everything(self):
        assert len(search("")) == len(TOPICS)

    def test_title_matches_come_first(self):
        results = search("geometry")
        assert results[0].title.startswith("Geometry")

    def test_body_text_is_searched(self):
        """"gated" appears only in the extractors body, never in a title."""
        assert not any("gated" in t.title.lower() for t in TOPICS)
        assert [t.title for t in search("gated")] == ["Feature extractors"]

    def test_keywords_widen_the_search(self):
        assert search("hotkeys"), "keyword-only match failed"

    def test_no_match_is_empty_not_everything(self):
        assert search("zzzznotathing") == []

    def test_search_is_case_insensitive(self):
        assert search("LASSO") == search("lasso")


class TestWindow:
    @pytest.fixture
    def help_window(self, qtbot):
        widget = HelpWindow()
        qtbot.addWidget(widget)
        yield widget
        widget.close()

    def test_lists_every_topic(self, help_window):
        assert help_window.topic_list.count() == len(TOPICS)

    def test_opens_on_the_first_topic(self, help_window):
        assert help_window.topic_list.currentRow() == 0
        assert "Getting started" in help_window.view.toPlainText()

    def test_selecting_a_topic_shows_it(self, help_window):
        row = next(i for i, t in enumerate(TOPICS) if t.title == "Keyboard shortcuts")
        help_window.topic_list.setCurrentRow(row)
        assert "Ctrl+O" in help_window.view.toPlainText()

    def test_search_filters_the_list(self, help_window):
        help_window.search_box.setText("lasso")
        assert 0 < help_window.topic_list.count() < len(TOPICS)

    def test_clearing_search_restores_everything(self, help_window):
        help_window.search_box.setText("lasso")
        help_window.search_box.setText("")
        assert help_window.topic_list.count() == len(TOPICS)

    def test_a_search_with_no_results_says_so(self, help_window):
        help_window.search_box.setText("zzzznotathing")
        assert help_window.topic_list.count() == 0
        assert "No topic matches" in help_window.view.toPlainText()

    def test_the_status_line_counts_results(self, help_window):
        assert f"{len(TOPICS)} topics" in help_window.status.text()
        help_window.search_box.setText("lasso")
        assert " of " in help_window.status.text()

    def test_show_topic_jumps(self, help_window):
        assert help_window.show_topic("Troubleshooting")
        assert help_window.topic_list.currentItem().text() == "Troubleshooting"

    def test_show_topic_rejects_an_unknown_title(self, help_window):
        assert not help_window.show_topic("Nope")

    def test_it_is_not_modal(self, help_window):
        """Modal would defeat the purpose: read an instruction, then do it."""
        assert not help_window.isModal()


class TestTheManualMatchesTheCode:
    """Claims that would silently rot if a control were renamed."""

    @pytest.fixture
    def manual(self):
        return " ".join(plain_text(t) for t in TOPICS)

    def test_menu_paths_exist(self, manual):
        import pathlearn.ui.main_window as mw
        for name in ("show_extractors", "extract_patches", "train_model",
                     "predict", "show_tsne", "show_slide_properties",
                     "open_extractors_folder", "show_help"):
            assert hasattr(mw.MainWindow, name), name

    def test_documented_shortcuts_are_the_real_ones(self, qtbot, manual):
        from PySide6.QtGui import QKeySequence
        import pathlearn.ui.main_window as mw

        expected = {"action_close": "Ctrl+W", "action_zoom_fit": "Ctrl+0",
                    "action_toggle_annotations": "Ctrl+H",
                    "action_extract": "Ctrl+E", "action_train": "Ctrl+T",
                    "action_predict": "Ctrl+R"}
        window = mw.MainWindow()
        qtbot.addWidget(window)
        for attr, keys in expected.items():
            actual = getattr(window, attr).shortcut()
            assert actual == QKeySequence(keys), f"{attr}: {actual.toString()}"
            assert keys in manual, f"{keys} is not in the manual"
        window.close()

    def test_documented_defaults_are_the_real_ones(self, manual):
        from pathlearn.core.geometry import GeometryConfig
        assert GeometryConfig().min_nuclei == 10
        assert "Default 10" in manual

    def test_the_sidebar_columns_are_as_documented(self, qtbot, tmp_path):
        from pathlearn.models.classification import ClassificationProfile
        from pathlearn.models.store import AnnotationStore
        from pathlearn.ui.panels.annotation_sidebar import AnnotationSidebar

        sidebar = AnnotationSidebar(AnnotationStore(), ClassificationProfile.default())
        qtbot.addWidget(sidebar)
        headers = [sidebar.tree.headerItem().text(c)
                   for c in range(sidebar.tree.columnCount())]
        assert headers == ["Use", "Class", "Name", "Area"]
        for column in headers:
            assert column in plain_text(
                next(t for t in TOPICS if t.title == "The Annotations panel"))

    def test_the_feature_sources_are_as_documented(self, manual):
        from pathlearn.data.geometry_bank import FeatureSource
        for source in FeatureSource:
            assert source.label in manual, source.label
        assert f"({FeatureSource.SHAPE.dimension}-D)" in manual
