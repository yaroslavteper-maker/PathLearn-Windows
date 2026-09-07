"""Renaming and pooling a trained model's classes.

Pooling a softmax means summing **probabilities**, not weights: P(a or b) =
P(a) + P(b) is the exact marginal, while adding weight columns is a different
function entirely. Most of what follows pins that.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlearn.models.classifier import ClassifierError, MLClassifier
from pathlearn.models.metrics import ClassMetrics, TrainingMetrics
from pathlearn.ui.sheets.edit_classes import EditClassesSheet, _common_prefix

GRADES = ["PanIN-1a", "PanIN-1b", "PanIN-2", "PanIN-3"]


def model(labels=None, dim=6, seed=0):
    labels = labels or GRADES
    rng = np.random.default_rng(seed)
    return MLClassifier(
        class_labels=list(labels),
        weights=rng.normal(0, 1, (dim, len(labels))).astype(np.float32),
        biases=rng.normal(0, 1, len(labels)).astype(np.float32),
        extractor_identity="geometry:handcrafted:r2",
        feature_source="geometry", aggregation="panin")


def metrics_for(labels, confusion):
    per_class = [ClassMetrics(label=l, precision=0.5, recall=0.5, f1=0.5,
                              support=int(sum(row)))
                 for l, row in zip(labels, confusion)]
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(len(labels)))
    return TrainingMetrics(class_labels=list(labels), train_accuracy=0.9,
                           val_accuracy=correct / total, per_class=per_class,
                           confusion=confusion, final_loss=0.1,
                           train_count=total, val_count=total)


class TestPoolingIsExact:
    def test_pooled_probability_is_the_sum(self):
        m = model()
        x = np.random.default_rng(1).normal(0, 1, 6).astype(np.float32)
        before = m.predict_proba(x)[0]
        pooled = m.with_class_mapping({g: "PanIN" for g in GRADES})
        after = pooled.predict_proba(x)[0]
        assert after.shape == (1,)
        assert after[0] == pytest.approx(before.sum(), abs=1e-6)

    def test_a_partial_pool_sums_only_its_members(self):
        m = model()
        x = np.random.default_rng(2).normal(0, 1, 6).astype(np.float32)
        before = m.predict_proba(x)[0]
        pooled = m.with_class_mapping({"PanIN-1a": "low", "PanIN-1b": "low",
                                       "PanIN-2": "high", "PanIN-3": "high"})
        after = pooled.predict_proba(x)[0]
        assert pooled.class_labels == ["low", "high"]
        assert after[0] == pytest.approx(before[0] + before[1], abs=1e-6)
        assert after[1] == pytest.approx(before[2] + before[3], abs=1e-6)

    def test_probabilities_still_sum_to_one(self):
        m = model()
        x = np.random.default_rng(3).normal(0, 1, (5, 6)).astype(np.float32)
        pooled = m.with_class_mapping({"PanIN-1a": "PanIN", "PanIN-1b": "PanIN",
                                       "PanIN-2": "PanIN"})
        assert np.allclose(pooled.predict_proba(x).sum(axis=1), 1.0, atol=1e-5)

    def test_the_weights_are_untouched(self):
        """Nothing is refitted, so accuracy cannot be invented."""
        m = model()
        pooled = m.with_class_mapping({g: "PanIN" for g in GRADES})
        assert np.array_equal(pooled.weights, m.weights)
        assert np.array_equal(pooled.biases, m.biases)

    def test_the_winner_can_change_for_the_right_reason(self):
        """Pooling can beat a class that was individually ahead of each part."""
        m = MLClassifier(class_labels=["a", "b", "c"],
                         weights=np.zeros((1, 3), dtype=np.float32),
                         biases=np.array([1.0, 1.0, 1.6], dtype=np.float32))
        x = np.zeros(1, dtype=np.float32)
        assert m.predict(x)[0] == "c"
        pooled = m.with_class_mapping({"a": "ab", "b": "ab"})
        assert pooled.predict(x)[0] == "ab"


class TestRenaming:
    def test_a_pure_rename_keeps_the_class_count(self):
        renamed = model().with_class_mapping({"PanIN-1a": "Low grade"})
        assert renamed.class_labels == ["Low grade", "PanIN-1b", "PanIN-2", "PanIN-3"]
        assert not renamed.is_regrouped

    def test_a_pure_rename_does_not_change_predictions(self):
        m = model()
        x = np.random.default_rng(4).normal(0, 1, 6).astype(np.float32)
        renamed = m.with_class_mapping({g: g.upper() for g in GRADES})
        assert np.allclose(m.predict_proba(x), renamed.predict_proba(x))

    def test_omitted_classes_keep_their_names(self):
        renamed = model().with_class_mapping({"PanIN-3": "High"})
        assert renamed.class_labels[:3] == GRADES[:3]

    def test_an_unknown_class_is_refused(self):
        with pytest.raises(ClassifierError, match="not classes of this model"):
            model().with_class_mapping({"Nonsense": "x"})

    def test_the_new_model_gets_its_own_id(self):
        m = model()
        assert m.with_class_mapping({"PanIN-1a": "x"}).id != m.id


class TestMetricsAreRescored:
    def confusion_model(self):
        # 1a and 1b are confused with each other; 2 and 3 are clean.
        from dataclasses import replace

        # 1a and 1b are confused with each other; 2 and 3 are clean.
        confusion = [[6, 4, 0, 0],
                     [5, 5, 0, 0],
                     [0, 0, 8, 2],
                     [0, 0, 1, 9]]
        return replace(model(), metrics=metrics_for(GRADES, confusion))

    def test_pooling_folds_the_confusion_matrix(self):
        pooled = self.confusion_model().with_class_mapping(
            {"PanIN-1a": "PanIN-1", "PanIN-1b": "PanIN-1"})
        assert pooled.metrics.class_labels == ["PanIN-1", "PanIN-2", "PanIN-3"]
        assert pooled.metrics.confusion[0][0] == 20      # 6+4+5+5

    def test_accuracy_rises_because_those_errors_stop_existing(self):
        original = self.confusion_model()
        pooled = original.with_class_mapping({"PanIN-1a": "PanIN-1",
                                              "PanIN-1b": "PanIN-1"})
        assert pooled.metrics.val_accuracy > original.metrics.val_accuracy
        assert pooled.metrics.val_accuracy == pytest.approx((20 + 8 + 9) / 40)

    def test_totals_are_conserved(self):
        pooled = self.confusion_model().with_class_mapping(
            {g: "PanIN" for g in GRADES})
        assert sum(sum(r) for r in pooled.metrics.confusion) == 40
        assert pooled.metrics.val_accuracy == pytest.approx(1.0)

    def test_per_class_support_is_recomputed(self):
        pooled = self.confusion_model().with_class_mapping(
            {"PanIN-1a": "PanIN-1", "PanIN-1b": "PanIN-1"})
        by_label = {c.label: c for c in pooled.metrics.per_class}
        assert by_label["PanIN-1"].support == 20

    def test_a_model_without_metrics_survives(self):
        assert model().with_class_mapping({"PanIN-1a": "x"}).metrics is None


class TestPersistence:
    def test_a_pooled_model_round_trips(self, tmp_path):
        pooled = model().with_class_mapping({"PanIN-1a": "low", "PanIN-1b": "low"})
        path = tmp_path / "pooled.cl"
        pooled.save(path)
        loaded = MLClassifier.load(path)
        assert loaded.class_labels == pooled.class_labels
        assert loaded.class_groups == pooled.class_groups
        assert loaded.source_labels == pooled.training_labels

    def test_predictions_survive_the_round_trip(self, tmp_path):
        pooled = model().with_class_mapping({"PanIN-1a": "low", "PanIN-1b": "low"})
        path = tmp_path / "pooled.cl"
        pooled.save(path)
        x = np.random.default_rng(5).normal(0, 1, 6).astype(np.float32)
        assert np.allclose(pooled.predict_proba(x),
                           MLClassifier.load(path).predict_proba(x), atol=1e-6)

    def test_an_ordinary_model_writes_no_grouping_fields(self, tmp_path):
        import json
        path = tmp_path / "plain.cl"
        model().save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("classGroups") is None

    def test_inconsistent_grouping_is_refused(self):
        m = model()
        with pytest.raises(ClassifierError, match="class_groups"):
            MLClassifier(class_labels=["a"], weights=m.weights, biases=m.biases,
                         source_labels=GRADES, class_groups=[0, 0])


class TestSheet:
    @pytest.fixture
    def sheet(self, qtbot):
        widget = EditClassesSheet(model())
        qtbot.addWidget(widget)
        yield widget
        widget.reject()

    def test_one_row_per_trained_class(self, sheet):
        assert sheet.table.rowCount() == 4
        assert sheet.table.item(0, 0).text() == "PanIN-1a"

    def test_the_trained_name_is_not_editable(self, sheet):
        from PySide6.QtCore import Qt
        assert not (sheet.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable)

    def test_it_starts_as_an_identity_mapping(self, sheet):
        assert sheet.mapping() == {g: g for g in GRADES}
        assert "No changes" in sheet.note.text()

    def test_typing_a_shared_name_pools(self, sheet):
        for row in range(4):
            sheet.table.item(row, 1).setText("PanIN")
        assert sheet.build().class_labels == ["PanIN"]

    def test_pooling_some_reports_how_many_went(self, sheet):
        sheet.table.item(0, 1).setText("PanIN-1")
        sheet.table.item(1, 1).setText("PanIN-1")
        assert sheet.build().class_labels == ["PanIN-1", "PanIN-2", "PanIN-3"]
        assert "1 class(es) pooled away" in sheet.note.text()

    def test_the_preview_lists_the_outputs(self, sheet):
        sheet.table.item(0, 1).setText("low")
        sheet.table.item(1, 1).setText("low")
        assert "3 output class(es)" in sheet.preview.text()
        assert "low" in sheet.preview.text()

    def test_merge_selected_needs_two_rows(self, sheet):
        sheet.table.selectRow(0)
        sheet._merge_selected()
        assert "two or more" in sheet.note.text()

    def test_merge_selected_applies_a_name(self, sheet):
        sheet.table.selectAll()
        sheet.merge_edit.setText("PanIN")
        sheet._merge_selected()
        assert sheet.build().class_labels == ["PanIN"]

    def test_merge_suggests_the_common_prefix(self, sheet):
        sheet.table.selectAll()
        sheet._merge_selected()
        assert sheet.merge_edit.text() == "PanIN"

    def test_reset_restores_the_original_names(self, sheet):
        sheet.table.item(0, 1).setText("something")
        sheet.reload()
        assert sheet.mapping() == {g: g for g in GRADES}

    def test_a_blank_name_falls_back_to_the_trained_one(self, sheet):
        sheet.table.item(0, 1).setText("   ")
        assert sheet.build().class_labels[0] == "PanIN-1a"

    def test_editing_an_already_pooled_model_shows_the_pooling(self, qtbot):
        pooled = model().with_class_mapping({"PanIN-1a": "low", "PanIN-1b": "low"})
        widget = EditClassesSheet(pooled)
        qtbot.addWidget(widget)
        assert widget.table.rowCount() == 4          # still four trained classes
        assert widget.table.item(0, 1).text() == "low"
        assert widget.table.item(1, 1).text() == "low"
        widget.reject()


class TestCommonPrefix:
    @pytest.mark.parametrize("labels,expected", [
        (["PanIN-1a", "PanIN-1b", "PanIN-2"], "PanIN"),
        (["PanIN-1a", "PanIN-1b"], "PanIN-1"),
        (["Acinar", "Islets"], ""),
        ([], ""),
        (["same", "same"], "same"),
    ])
    def test_prefixes(self, labels, expected):
        assert _common_prefix(labels) == expected


class TestOneClassIsCalledOut:
    """Pooling everything scores 100% and says nothing. Say so."""

    @pytest.fixture
    def sheet(self, qtbot):
        widget = EditClassesSheet(model())
        qtbot.addWidget(widget)
        yield widget
        widget.reject()

    def test_the_warning_appears(self, sheet):
        for row in range(sheet.table.rowCount()):
            sheet.table.item(row, 1).setText("PanIN")
        assert "cannot be wrong" in sheet.note.text()

    def test_it_is_styled_as_a_warning(self, sheet):
        for row in range(sheet.table.rowCount()):
            sheet.table.item(row, 1).setText("PanIN")
        assert "#d0762a" in sheet.note.styleSheet()

    def test_two_classes_are_not_warned_about(self, sheet):
        for row, name in enumerate(["low", "low", "high", "high"]):
            sheet.table.item(row, 1).setText(name)
        assert "cannot be wrong" not in sheet.note.text()
        assert "pooled away" in sheet.note.text()

    def test_saving_one_class_is_still_allowed(self, sheet):
        """Warned about, not forbidden — it is a legitimate detector."""
        for row in range(sheet.table.rowCount()):
            sheet.table.item(row, 1).setText("PanIN")
        assert sheet.save_button.isEnabled()
        assert sheet.build().class_labels == ["PanIN"]
