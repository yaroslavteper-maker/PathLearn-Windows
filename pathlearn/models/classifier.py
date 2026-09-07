"""The trained classifier and its `.cl` file — ``Models/MLClassifier.swift``.

Schema is ``02-DATA-FORMATS.md`` §2, reproduced exactly so files interchange
with macOS.  Every optional field falls back to a legacy default when absent,
which is what lets a `.paninmodel.json` written before the extractor system
existed still load.

WHAT A CLASSIFIER CARRIES BESIDES WEIGHTS
=========================================
* ``extractor_identity`` — the feature space it was trained in.  Predicting with
  a different extractor is meaningless, so this is checked, not trusted.
* ``aggregation`` — ``"meanmaxstd"`` means it expects one pooled vector per
  annotation rather than one per patch.
* ``feature_mean``/``feature_std`` — z-scoring applied at predict time.
* ``feature_source`` — ``"geometry"`` means the 14-D descriptor, not embeddings.
* ``null_reference``/``null_threshold`` — the learned lumen-exclusion filter
  (``04-DESIGN-DECISIONS.md`` §4), persisted so prediction filters exactly as
  training did.
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..core.logistic import softmax
from ..extractors.identity import LEGACY_VISION
from .metrics import TrainingMetrics

CLASSIFIER_VERSION = 1
CLASSIFIER_SUFFIX = ".cl"
#: The macOS build also wrote this older name; we read it, but never write it.
LEGACY_CLASSIFIER_SUFFIX = ".paninmodel.json"

KIND_LOGISTIC = "logistic"
KIND_CENTROID = "centroid"

AGGREGATION_MEAN_MAX_STD = "meanmaxstd"
SOURCE_GEOMETRY = "geometry"

#: Default cosine cutoff for null exclusion.
DEFAULT_NULL_THRESHOLD = 0.85
#: Cap on stored null reference vectors, matching the Swift.
MAX_NULL_REFERENCE = 128


class ClassifierError(ValueError):
    """Raised when a classifier file is malformed or incompatible."""


@dataclass(slots=True)
class MLClassifier:
    """A trained softmax classifier over one feature space."""

    class_labels: list[str]
    #: ``(feature_dim, class_count)`` row-major, matching the Swift layout.
    weights: np.ndarray
    biases: np.ndarray
    extractor_identity: str = LEGACY_VISION
    kind: str = KIND_LOGISTIC
    aggregation: str | None = None
    feature_source: str | None = None
    #: Set when output classes have been pooled or renamed after
    #: training. ``weights``/``biases`` stay over these *source* classes;
    #: ``class_labels`` are what the model now reports.
    source_labels: list[str] | None = None
    #: For each source column, the index into ``class_labels`` it feeds.
    class_groups: list[int] | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    null_reference: np.ndarray | None = None
    null_threshold: float | None = None
    metrics: TrainingMetrics | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding_snapshot: dict | None = None

    def __post_init__(self) -> None:
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        self.biases = np.ascontiguousarray(self.biases, dtype=np.float32)
        if self.weights.ndim != 2:
            raise ClassifierError(f"weights must be 2-D, got {self.weights.shape}")
        columns = self.weights.shape[1]
        if self.class_groups is not None:
            # Grouped: the weights span the source classes, and each
            # column names the output class it contributes to.
            if self.source_labels is None:
                raise ClassifierError(
                    "class_groups needs source_labels alongside it")
            if len(self.class_groups) != columns:
                raise ClassifierError(
                    f"class_groups has {len(self.class_groups)} entries "
                    f"but the weights have {columns} columns")
            if len(self.source_labels) != columns:
                raise ClassifierError(
                    f"source_labels has {len(self.source_labels)} entries "
                    f"but the weights have {columns} columns")
            if self.class_groups and (min(self.class_groups) < 0
                                      or max(self.class_groups)
                                      >= len(self.class_labels)):
                raise ClassifierError(
                    "class_groups points outside class_labels")
        elif columns != len(self.class_labels):
            raise ClassifierError(
                f"weights have {columns} columns but there are "
                f"{len(self.class_labels)} class labels")
        if self.biases.shape != (columns,):
            raise ClassifierError(
                f"biases must have one entry per weight column "
                f"({columns}), got {self.biases.shape}")

    # -- shape ------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        return int(self.weights.shape[0])

    @property
    def class_count(self) -> int:
        return len(self.class_labels)

    @property
    def is_pooled(self) -> bool:
        return self.aggregation == AGGREGATION_MEAN_MAX_STD

    @property
    def is_geometry(self) -> bool:
        return self.feature_source == SOURCE_GEOMETRY

    @property
    def uses_null_filter(self) -> bool:
        return self.null_reference is not None and self.null_threshold is not None

    def describe(self) -> str:
        bits = [f"{self.class_count} classes", f"{self.feature_dim}-d", self.kind]
        if self.is_pooled:
            bits.append("pooled")
        if self.is_geometry:
            bits.append("geometry")
        if self.is_regrouped:
            bits.append(f"regrouped from {len(self.training_labels)}")
        if self.uses_null_filter:
            bits.append(f"null<{self.null_threshold:g}")
        return f"{self.extractor_identity} · " + " · ".join(bits)

    # -- prediction -------------------------------------------------------

    def normalise(self, features: np.ndarray) -> np.ndarray:
        """Apply the stored z-scoring, if any."""
        array = np.atleast_2d(np.asarray(features, dtype=np.float32))
        if self.feature_mean is not None and self.feature_std is not None:
            array = (array - self.feature_mean) / self.feature_std
        return array.astype(np.float32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        array = self.normalise(features)
        if array.shape[1] != self.feature_dim:
            raise ClassifierError(
                f"Model expects {self.feature_dim}-d features, got {array.shape[1]}. "
                f"It was trained with {self.extractor_identity}.")
        probabilities = softmax(array @ self.weights + self.biases)
        if self.class_groups is None:
            return probabilities
        # Pooling a softmax means summing PROBABILITIES, not weights:
        # P(merged) = P(a) + P(b) is the exact marginal, whereas adding
        # weight columns is not the same function at all.
        pooled = np.zeros((probabilities.shape[0], len(self.class_labels)),
                          dtype=np.float32)
        for source, target in enumerate(self.class_groups):
            pooled[:, target] += probabilities[:, source]
        return pooled

    def predict(self, features: np.ndarray) -> tuple[str, np.ndarray]:
        """Label and probability vector for a single feature vector."""
        probs = self.predict_proba(np.asarray(features).ravel())[0]
        return self.class_labels[int(np.argmax(probs))], probs

    def predict_batch(self, features: np.ndarray) -> tuple[list[str], np.ndarray]:
        probs = self.predict_proba(features)
        return [self.class_labels[i] for i in np.argmax(probs, axis=1)], probs

    def is_null_like(self, features: np.ndarray) -> np.ndarray:
        """Which rows resemble the null reference closely enough to drop.

        Cosine similarity to the **nearest** null vector, compared against
        ``null_threshold``.  Computed on the raw (un-z-scored) vectors, which is
        what the Swift stored and compared.
        """
        array = np.atleast_2d(np.asarray(features, dtype=np.float32))
        if not self.uses_null_filter:
            return np.zeros(array.shape[0], dtype=bool)

        reference = np.asarray(self.null_reference, dtype=np.float32)
        a = array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-12)
        b = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12)
        return (a @ b.T).max(axis=1) >= self.null_threshold

    @property
    def is_regrouped(self) -> bool:
        """True when output classes have been pooled or renamed after training."""
        return self.class_groups is not None

    @property
    def training_labels(self) -> list[str]:
        """The classes the weights were actually fitted on."""
        return list(self.source_labels or self.class_labels)

    def with_class_mapping(self, mapping: dict[str, str],
                           name: str | None = None) -> "MLClassifier":
        """A copy whose classes are renamed and/or pooled.

        *mapping* takes each **training** label to the output label it should
        report as. Several training labels may map to the same output label,
        which pools them: the pooled probability is the sum of theirs, which is
        the exact marginal. Nothing is refitted — the weights are untouched —
        so this cannot invent accuracy it did not have. What it can do is stop
        counting a PanIN-2-called-PanIN-3 as an error, because a model that
        only reports "PanIN" was never asked to make that call.

        A label left out of *mapping* keeps its own name.
        """
        training = self.training_labels
        unknown = set(mapping) - set(training)
        if unknown:
            raise ClassifierError(
                f"These are not classes of this model: {', '.join(sorted(unknown))}. "
                f"It was trained on {', '.join(training)}.")

        outputs: list[str] = []
        groups: list[int] = []
        for label in training:
            target = (mapping.get(label) or label).strip() or label
            if target not in outputs:
                outputs.append(target)
            groups.append(outputs.index(target))

        if len(outputs) < 1:
            raise ClassifierError("A model needs at least one output class.")

        metrics = self.metrics.regrouped(outputs, groups) if self.metrics else None
        collapsed = groups == list(range(len(training)))
        return replace(
            self,
            id=uuid.uuid4(),          # a different model, not a revision
            class_labels=list(outputs),
            # Identity mapping after a pure rename: keep the file simple, since
            # nothing is being pooled.
            source_labels=None if collapsed else list(training),
            class_groups=None if collapsed else groups,
            metrics=metrics,
            created_at=datetime.now(timezone.utc),
        )

    def accepts(self, extractor_identity: str) -> bool:
        """Whether this model may be applied to features from *extractor_identity*."""
        if self.is_geometry:
            return True     # geometry descriptors have no extractor
        return self.extractor_identity == extractor_identity

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": CLASSIFIER_VERSION,
            "id": str(self.id),
            "createdAt": self.created_at.isoformat(),
            "classLabels": list(self.class_labels),
            "featureDim": self.feature_dim,
            # The number of weight COLUMNS, which is the training class
            # count — not len(class_labels), which a pooled model shrinks.
            # Writing the latter makes the file unloadable: the reshape on
            # the way back in uses this number.
            "classCount": int(self.weights.shape[1]),
            # Row-major float32, exactly as the Swift wrote it.
            "weightsBase64": _encode(self.weights),
            "biasesBase64": _encode(self.biases),
            "featureExtractorRevision": _revision_of(self.extractor_identity),
            "extractorIdentity": self.extractor_identity,
            "kind": self.kind,
            "aggregation": self.aggregation,
            "featureSource": self.feature_source,
            "sourceLabels": (list(self.source_labels)
                             if self.source_labels else None),
            "classGroups": (list(self.class_groups)
                            if self.class_groups is not None else None),
            "featureMean": _list_or_none(self.feature_mean),
            "featureStd": _list_or_none(self.feature_std),
            "nullReference": ([list(map(float, row)) for row in self.null_reference]
                              if self.null_reference is not None else None),
            "nullThreshold": self.null_threshold,
            "embeddingSnapshot": self.embedding_snapshot,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }

    def to_json(self) -> str:
        """The `.cl` document as text, for embedding without a file."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "MLClassifier":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ClassifierError(f"Malformed classifier JSON: {exc}") from exc

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "MLClassifier":
        if not isinstance(data, dict):
            raise ClassifierError("Classifier file must be a JSON object")
        try:
            labels = [str(x) for x in data["classLabels"]]
            feature_dim = int(data["featureDim"])
            class_count = int(data.get("classCount", len(labels)))
            weights = _decode(data["weightsBase64"]).reshape(feature_dim, class_count)
            biases = _decode(data["biasesBase64"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClassifierError(f"Malformed classifier: {exc}") from exc

        null_reference = data.get("nullReference")
        if null_reference is not None:
            null_reference = np.asarray(null_reference, dtype=np.float32)

        metrics = data.get("metrics")
        try:
            when = datetime.fromisoformat(str(data.get("createdAt")).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            when = datetime.now(timezone.utc)
        try:
            ident = uuid.UUID(str(data.get("id")))
        except (ValueError, TypeError):
            ident = uuid.uuid4()

        return cls(
            class_labels=labels,
            weights=weights,
            biases=biases,
            # Absent identity means a legacy Vision-print model.
            extractor_identity=str(data.get("extractorIdentity") or LEGACY_VISION),
            kind=str(data.get("kind") or KIND_LOGISTIC),
            aggregation=data.get("aggregation") or None,
            feature_source=data.get("featureSource") or None,
            source_labels=(list(data["sourceLabels"])
                           if data.get("sourceLabels") else None),
            class_groups=([int(g) for g in data["classGroups"]]
                          if data.get("classGroups") is not None else None),
            feature_mean=_array_or_none(data.get("featureMean")),
            feature_std=_array_or_none(data.get("featureStd")),
            null_reference=null_reference,
            null_threshold=(float(data["nullThreshold"])
                            if data.get("nullThreshold") is not None else None),
            metrics=TrainingMetrics.from_dict(metrics) if isinstance(metrics, dict) else None,
            id=ident,
            created_at=when,
            embedding_snapshot=data.get("embeddingSnapshot") or None,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLClassifier":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClassifierError(f"Could not read {path.name}: {exc}") from exc
        return cls.from_dict(data)


def _encode(array: np.ndarray) -> str:
    return base64.b64encode(
        np.ascontiguousarray(array, dtype=np.float32).tobytes()).decode("ascii")


def _decode(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32).copy()


def _list_or_none(array: np.ndarray | None) -> list | None:
    return None if array is None else [float(v) for v in np.asarray(array).ravel()]


def _array_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    return array if array.size else None


def _revision_of(identity: str) -> int:
    from ..extractors.identity import ExtractorIdentity
    try:
        return ExtractorIdentity.parse(identity).revision
    except ValueError:
        return 1
