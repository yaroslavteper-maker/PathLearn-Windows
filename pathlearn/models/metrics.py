"""Training metrics — from ``Models/TrainingMetrics.swift`` (``02-DATA-FORMATS.md`` §7).

Serialised inside the `.cl` file so a saved model carries the evidence for how
well it did, not just its weights.

The confusion matrix is the honest part of this: with ~136 annotations across
10 classes, a model that collapses to the majority class can still post a
respectable accuracy, and only the matrix shows it
(``04-DESIGN-DECISIONS.md`` §2 records that exact failure happening twice).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict:
        return {"label": self.label, "precision": self.precision,
                "recall": self.recall, "f1": self.f1, "support": self.support}

    @classmethod
    def from_dict(cls, data: dict) -> "ClassMetrics":
        return cls(
            label=str(data.get("label", "?")),
            precision=float(data.get("precision", 0.0)),
            recall=float(data.get("recall", 0.0)),
            f1=float(data.get("f1", 0.0)),
            support=int(data.get("support", 0)),
        )


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    class_labels: list[str]
    train_accuracy: float
    val_accuracy: float
    per_class: list[ClassMetrics]
    #: ``confusion[true][predicted]``, counted on the validation split.
    confusion: list[list[int]]
    final_loss: float
    train_count: int
    val_count: int
    extractor_revision: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def regrouped(self, labels: list[str], groups: list[int]) -> "TrainingMetrics":
        """Re-score these metrics with classes pooled, from the confusion matrix.

        Exact, not an approximation: pooling two classes merges their rows and
        columns, and a confusion *between* them stops being an error because
        the pooled model never had to tell them apart. Accuracy therefore rises
        — legitimately — and the new per-class figures are recomputed rather
        than carried over.
        """
        matrix = np.asarray(self.confusion, dtype=float)
        if matrix.size == 0 or matrix.shape[0] != len(groups):
            return replace(self, class_labels=list(labels), per_class=[],
                           confusion=[])

        size = len(labels)
        folded = np.zeros((size, size), dtype=float)
        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                folded[gi][gj] += matrix[i][j]

        per_class = []
        for index, label in enumerate(labels):
            true_positive = folded[index][index]
            predicted = folded[:, index].sum()
            actual = folded[index].sum()
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / actual if actual else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) else 0.0)
            per_class.append(ClassMetrics(label=label, precision=precision,
                                          recall=recall, f1=f1,
                                          support=int(round(actual))))

        total = folded.sum()
        return replace(self, class_labels=list(labels), per_class=per_class,
                       confusion=folded.astype(int).tolist(),
                       val_accuracy=float(np.trace(folded) / total) if total else 0.0)

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1 — the number a collapsed model cannot fake."""
        if not self.per_class:
            return 0.0
        return float(np.mean([c.f1 for c in self.per_class]))

    @property
    def predicted_labels_used(self) -> int:
        """How many distinct classes the model ever actually predicts.

        Fewer than the class count means partial or total collapse.
        """
        matrix = np.asarray(self.confusion)
        if matrix.size == 0:
            return 0
        return int((matrix.sum(axis=0) > 0).sum())

    @property
    def is_collapsed(self) -> bool:
        return len(self.class_labels) > 1 and self.predicted_labels_used <= 1

    def summary(self) -> str:
        lines = [
            f"train {self.train_accuracy:.1%} ({self.train_count} patches) · "
            f"val {self.val_accuracy:.1%} ({self.val_count} patches) · "
            f"macro-F1 {self.macro_f1:.3f}",
        ]
        if self.is_collapsed:
            lines.append("WARNING: the model predicts a single class — it has collapsed.")
        elif self.predicted_labels_used < len(self.class_labels):
            lines.append(
                f"NOTE: only {self.predicted_labels_used} of {len(self.class_labels)} "
                f"classes are ever predicted.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "classLabels": list(self.class_labels),
            "trainAccuracy": self.train_accuracy,
            "valAccuracy": self.val_accuracy,
            "perClass": [c.to_dict() for c in self.per_class],
            "confusion": [list(map(int, row)) for row in self.confusion],
            "finalLoss": self.final_loss,
            "trainCount": self.train_count,
            "valCount": self.val_count,
            "extractorRevision": self.extractor_revision,
            "createdAt": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingMetrics":
        created = data.get("createdAt")
        try:
            when = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            when = datetime.now(timezone.utc)
        return cls(
            class_labels=[str(x) for x in (data.get("classLabels") or [])],
            train_accuracy=float(data.get("trainAccuracy", 0.0)),
            val_accuracy=float(data.get("valAccuracy", 0.0)),
            per_class=[ClassMetrics.from_dict(c) for c in (data.get("perClass") or [])],
            confusion=[[int(v) for v in row] for row in (data.get("confusion") or [])],
            final_loss=float(data.get("finalLoss", 0.0)),
            train_count=int(data.get("trainCount", 0)),
            val_count=int(data.get("valCount", 0)),
            extractor_revision=int(data.get("extractorRevision", 1)),
            created_at=when,
        )


def evaluate(y_true: Sequence[int], y_pred: Sequence[int],
             class_labels: Sequence[str]) -> tuple[float, list[ClassMetrics], list[list[int]]]:
    """Accuracy, per-class precision/recall/F1, and the confusion matrix.

    A class with no predictions gets precision 0 rather than NaN, so a collapsed
    model produces a readable report instead of a table of blanks.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = len(class_labels)

    confusion = np.zeros((k, k), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if 0 <= actual < k and 0 <= predicted < k:
            confusion[actual, predicted] += 1

    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0

    per_class: list[ClassMetrics] = []
    for i, label in enumerate(class_labels):
        true_positive = int(confusion[i, i])
        predicted_positive = int(confusion[:, i].sum())
        actual_positive = int(confusion[i, :].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class.append(ClassMetrics(label, precision, recall, f1, actual_positive))

    return accuracy, per_class, confusion.tolist()
