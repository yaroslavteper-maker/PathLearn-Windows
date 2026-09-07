"""Classification classes and profiles — from ``Models/Classification.swift``.

A *profile* is the palette of classes the user annotates with.  Persisted as
``<name>.panin-profile.json`` (``02-DATA-FORMATS.md`` §8).

``is_null`` marks an **Exclude / null** class.  Patches of such a class never
train as a real class, and at predict time candidates that look like them are
dropped (cosine similarity to the nearest null vector >= threshold).  Profiles
written before the flag existed still load, defaulting it to ``False`` — the
same backward compatibility the Swift custom decoder provided.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .annotation import AnnotationColor

PROFILE_VERSION = 1

#: Profiles are no longer a file of their own — the class palette lives in a
#: saved state, and in settings between runs. The suffix remains only so a
#: profile written by an older build can still be opened by hand.
PROFILE_SUFFIX = ".panin-profile.json"


@dataclass(slots=True)
class Classification:
    """One annotatable class: a name, a colour, and the null flag."""

    name: str
    color: AnnotationColor
    is_null: bool = False
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "color": {"r": self.color.r, "g": self.color.g, "b": self.color.b},
            "isNull": self.is_null,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Classification":
        raw_color = data.get("color") or {}
        if isinstance(raw_color, dict):
            color = AnnotationColor(
                _int(raw_color.get("r"), 200),
                _int(raw_color.get("g"), 60),
                _int(raw_color.get("b"), 60),
            )
        else:  # tolerate a [r, g, b] list
            color = AnnotationColor.from_sequence(list(raw_color) + [0, 0, 0])
        try:
            ident = uuid.UUID(str(data.get("id")))
        except (ValueError, TypeError):
            ident = uuid.uuid4()
        return cls(
            name=str(data.get("name") or "Unlabeled"),
            color=color,
            # Absent in profiles written before the flag existed.
            is_null=bool(data.get("isNull", False)),
            id=ident,
        )


@dataclass(slots=True)
class ClassificationProfile:
    """A named set of classes."""

    name: str
    classes: list[Classification]

    def to_json(self) -> str:
        return json.dumps({
            "version": PROFILE_VERSION,
            "name": self.name,
            "classes": [c.to_dict() for c in self.classes],
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ClassificationProfile":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Profile must be a JSON object")
        raw_classes = data.get("classes")
        classes = [Classification.from_dict(c)
                   for c in (raw_classes or [])
                   if isinstance(c, dict)]
        return cls(name=str(data.get("name") or "Untitled"), classes=classes)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ClassificationProfile":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def by_name(self, name: str) -> Classification | None:
        for c in self.classes:
            if c.name == name:
                return c
        return None

    @property
    def null_class_names(self) -> set[str]:
        return {c.name for c in self.classes if c.is_null}

    @staticmethod
    def default() -> "ClassificationProfile":
        """The Swift ``ClassificationProfile.default``, reproduced exactly."""
        return ClassificationProfile(
            name="Pancreatic Pathology",
            classes=[
                Classification("PaNIN-1", AnnotationColor(80, 180, 80)),
                Classification("PaNIN-2", AnnotationColor(240, 180, 30)),
                Classification("PaNIN-3", AnnotationColor(220, 60, 60)),
                Classification("Normal", AnnotationColor(80, 160, 220)),
                Classification("Stroma", AnnotationColor(180, 130, 200)),
            ],
        )


def _int(value: Any, default: int) -> int:
    try:
        return max(0, min(255, int(round(float(value)))))
    except (TypeError, ValueError):
        return default
