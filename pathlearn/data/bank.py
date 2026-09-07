"""The patch bank — SQLite, replacing SwiftData (``02-DATA-FORMATS.md`` §5).

Stores one row per extracted patch: where it came from, what class it belongs
to, its feature vector, and **which extractor produced that vector**.  The
extractor identity is the guard that stops two feature spaces being mixed.

STORAGE LOCATION
================
``%LOCALAPPDATA%\\PathLearn\\bank.db`` — an app-specific path, never a shared or
default one.  ``04-DESIGN-DECISIONS.md`` §6 records why: the macOS build once
let SwiftData use its implicit default store and another app clobbered it,
destroying the patch bank.  The `.bank` JSON export is the recovery path, so it
stays a first-class format rather than an afterthought.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from ..extractors.identity import LEGACY_VISION, ExtractorIdentity

#: `featureElementType` values, matching the Swift.
ELEMENT_FLOAT32 = 1
ELEMENT_FLOAT16 = 2

_DTYPE_FOR_ELEMENT = {ELEMENT_FLOAT32: np.float32, ELEMENT_FLOAT16: np.float16}

#: Version stamped into `.bank` exports.
BANK_FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS patches (
    id                  TEXT PRIMARY KEY,
    slide_path          TEXT NOT NULL,
    slide_name          TEXT NOT NULL,
    annotation_id       TEXT NOT NULL,
    classification      TEXT NOT NULL,
    patch_x             INTEGER NOT NULL,
    patch_y             INTEGER NOT NULL,
    patch_level         INTEGER NOT NULL,
    patch_size_level    INTEGER NOT NULL,
    feature_data        BLOB NOT NULL,
    feature_dim         INTEGER NOT NULL,
    feature_element     INTEGER NOT NULL,
    extractor_identity  TEXT,
    extractor_revision  INTEGER NOT NULL DEFAULT 1,
    white_fraction      REAL NOT NULL DEFAULT 0,
    nucleus_count       INTEGER NOT NULL DEFAULT -1,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patches_extractor ON patches(extractor_identity);
CREATE INDEX IF NOT EXISTS idx_patches_class     ON patches(classification);
CREATE INDEX IF NOT EXISTS idx_patches_slide     ON patches(slide_path);
CREATE INDEX IF NOT EXISTS idx_patches_annot     ON patches(annotation_id);
"""


def default_bank_path() -> Path:
    directory = Path.home() / "AppData" / "Local" / "PathLearn"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "bank.db"


def encode_features(vector: np.ndarray, element_type: int = ELEMENT_FLOAT32) -> bytes:
    dtype = _DTYPE_FOR_ELEMENT[element_type]
    return np.ascontiguousarray(np.asarray(vector, dtype=dtype)).tobytes()


def decode_features(blob: bytes, dim: int, element_type: int) -> np.ndarray:
    """Always returns float32, whatever was stored."""
    dtype = _DTYPE_FOR_ELEMENT.get(element_type, np.float32)
    array = np.frombuffer(blob, dtype=dtype)
    if array.size != dim:
        raise ValueError(f"Feature blob holds {array.size} values, expected {dim}")
    return array.astype(np.float32)


@dataclass(slots=True)
class Patch:
    """One extracted patch.  Coordinates are level-0, top-left origin, Y down."""

    slide_path: str
    slide_name: str
    annotation_id: uuid.UUID
    classification: str
    patch_x: int
    patch_y: int
    patch_level: int
    patch_size_level: int
    features: np.ndarray
    extractor_identity: str
    white_fraction: float = 0.0
    nucleus_count: int = -1
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    element_type: int = ELEMENT_FLOAT32

    @property
    def feature_dim(self) -> int:
        return int(np.asarray(self.features).size)

    @property
    def extractor_revision(self) -> int:
        try:
            return ExtractorIdentity.parse(self.extractor_identity).revision
        except ValueError:
            return 1

    @property
    def size_level0(self) -> int:
        """Patch edge length in level-0 pixels."""
        return int(round(self.patch_size_level * (2 ** self.patch_level)))


@dataclass(frozen=True, slots=True)
class BankStats:
    total: int
    by_class: dict[str, int]
    by_extractor: dict[str, int]
    by_slide: dict[str, int]
    feature_dims: dict[str, int]

    @property
    def is_empty(self) -> bool:
        return self.total == 0

    @property
    def is_mixed(self) -> bool:
        """True when the bank holds more than one feature space."""
        return len(self.by_extractor) > 1

    def summary(self) -> str:
        if self.is_empty:
            return "Bank is empty."
        lines = [f"{self.total} patches, {len(self.by_class)} classes, "
                 f"{len(self.by_slide)} slides"]
        for identity, count in sorted(self.by_extractor.items()):
            dim = self.feature_dims.get(identity, "?")
            lines.append(f"  {identity}  {count} patches, dim {dim}")
        if self.is_mixed:
            lines.append("  NOTE: multiple extractors present — training must pick one.")
        return "\n".join(lines)


class PatchBank:
    """SQLite-backed store of extracted patches."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_bank_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        # WAL keeps reads working while a long extraction writes.
        if str(self.path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def reopen(self) -> None:
        """Reconnect after the database file has been replaced on disk.

        Restoring a saved state overwrites bank.db underneath this object;
        without reopening, every later query would use a handle to the file
        that used to be there.
        """
        self.close()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        if str(self.path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    def _require_conn(self) -> sqlite3.Connection:
        """Fail clearly on use-after-close.

        Without this the caller gets ``'NoneType' object has no attribute
        'execute'`` from somewhere deep in a query, which says nothing about
        what actually went wrong.
        """
        if self._conn is None:
            raise RuntimeError(
                f"Patch bank {self.path.name} is closed — reopen it before querying.")
        return self._conn

    def __enter__(self) -> "PatchBank":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._require_conn()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    # -- writing ----------------------------------------------------------

    def add(self, patch: Patch) -> None:
        self.add_many([patch])

    def add_many(self, patches: Iterable[Patch]) -> int:
        """Insert in one transaction.  Returns how many rows were written."""
        rows = [self._to_row(p) for p in patches]
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO patches VALUES
                   (:id, :slide_path, :slide_name, :annotation_id, :classification,
                    :patch_x, :patch_y, :patch_level, :patch_size_level,
                    :feature_data, :feature_dim, :feature_element,
                    :extractor_identity, :extractor_revision,
                    :white_fraction, :nucleus_count, :created_at)""",
                rows,
            )
        return len(rows)

    @staticmethod
    def _to_row(p: Patch) -> dict:
        return {
            "id": str(p.id),
            "slide_path": p.slide_path,
            "slide_name": p.slide_name,
            "annotation_id": str(p.annotation_id),
            "classification": p.classification,
            "patch_x": int(p.patch_x),
            "patch_y": int(p.patch_y),
            "patch_level": int(p.patch_level),
            "patch_size_level": int(p.patch_size_level),
            "feature_data": encode_features(p.features, p.element_type),
            "feature_dim": p.feature_dim,
            "feature_element": p.element_type,
            "extractor_identity": p.extractor_identity,
            "extractor_revision": p.extractor_revision,
            "white_fraction": float(p.white_fraction),
            "nucleus_count": int(p.nucleus_count),
            "created_at": p.created_at.isoformat(),
        }

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Patch:
        return Patch(
            id=uuid.UUID(row["id"]),
            slide_path=row["slide_path"],
            slide_name=row["slide_name"],
            annotation_id=uuid.UUID(row["annotation_id"]),
            classification=row["classification"],
            patch_x=row["patch_x"],
            patch_y=row["patch_y"],
            patch_level=row["patch_level"],
            patch_size_level=row["patch_size_level"],
            features=decode_features(row["feature_data"], row["feature_dim"],
                                     row["feature_element"]),
            extractor_identity=row["extractor_identity"] or LEGACY_VISION,
            white_fraction=row["white_fraction"],
            nucleus_count=row["nucleus_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            element_type=row["feature_element"],
        )

    # -- reading ----------------------------------------------------------

    def fetch(self, *, extractor_identity: str | None = None,
              classifications: Sequence[str] | None = None,
              slide_path: str | None = None,
              annotation_id: uuid.UUID | None = None,
              max_white_fraction: float | None = None,
              min_nuclei: int | None = None,
              limit: int | None = None) -> list[Patch]:
        """Patches matching every supplied filter.

        ``min_nuclei`` deliberately keeps rows with ``nucleus_count == -1``
        (never measured); excluding them would silently drop every patch
        extracted without nucleus counting enabled.
        """
        clauses, params = [], []
        if extractor_identity is not None:
            clauses.append("extractor_identity = ?")
            params.append(extractor_identity)
        if classifications:
            clauses.append(f"classification IN ({','.join('?' * len(classifications))})")
            params.extend(classifications)
        if slide_path is not None:
            clauses.append("slide_path = ?")
            params.append(slide_path)
        if annotation_id is not None:
            clauses.append("annotation_id = ?")
            params.append(str(annotation_id))
        if max_white_fraction is not None:
            clauses.append("white_fraction <= ?")
            params.append(float(max_white_fraction))
        if min_nuclei is not None:
            clauses.append("(nucleus_count >= ? OR nucleus_count < 0)")
            params.append(int(min_nuclei))

        sql = "SELECT * FROM patches"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        return [self._from_row(r) for r in self._require_conn().execute(sql, params)]

    def feature_matrix(self, patches: Sequence[Patch]) -> tuple[np.ndarray, list[str]]:
        """Stack patches into ``(n, d)`` plus their labels.

        Raises if the feature dimensions disagree — that means two extractors
        got mixed, and silently truncating would corrupt training.
        """
        if not patches:
            return np.zeros((0, 0), dtype=np.float32), []
        dims = {p.feature_dim for p in patches}
        if len(dims) != 1:
            identities = sorted({p.extractor_identity for p in patches})
            raise ValueError(
                f"Patches have mixed feature dimensions {sorted(dims)} "
                f"(extractors: {identities}). Filter by extractor first."
            )
        matrix = np.stack([p.features for p in patches]).astype(np.float32)
        return matrix, [p.classification for p in patches]

    def stats(self) -> BankStats:
        conn = self._require_conn()
        total = conn.execute("SELECT COUNT(*) FROM patches").fetchone()[0]
        def group(column: str) -> dict[str, int]:
            return {r[0]: r[1] for r in conn.execute(
                f"SELECT {column}, COUNT(*) FROM patches GROUP BY {column}")}
        dims = {r[0]: r[1] for r in conn.execute(
            "SELECT extractor_identity, MAX(feature_dim) FROM patches "
            "GROUP BY extractor_identity")}
        return BankStats(total, group("classification"), group("extractor_identity"),
                         group("slide_name"), dims)

    @property
    def extractor_identities(self) -> list[str]:
        return sorted(r[0] for r in self._require_conn().execute(
            "SELECT DISTINCT extractor_identity FROM patches") if r[0])

    def count(self, **filters) -> int:
        return len(self.fetch(**filters))

    def rename_class(self, old: str, new: str) -> int:
        """Relabel every patch of class *old* as *new*.  Returns the count.

        Renaming rather than deleting matters: these rows carry extracted
        features that cost real time to produce, and a palette rename that
        orphaned them would leave the bank training a class the user believes
        no longer exists.
        """
        if not old or not new or old == new:
            return 0
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE patches SET classification = ? WHERE classification = ?",
                (new, old))
        return cursor.rowcount

    # -- deleting ---------------------------------------------------------

    def delete_where(self, *, extractor_identity: str | None = None,
                     slide_path: str | None = None,
                     annotation_id: uuid.UUID | None = None,
                     classification: str | None = None) -> int:
        clauses, params = [], []
        for column, value in (("extractor_identity", extractor_identity),
                              ("slide_path", slide_path),
                              ("classification", classification)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if annotation_id is not None:
            clauses.append("annotation_id = ?")
            params.append(str(annotation_id))
        if not clauses:
            raise ValueError("delete_where needs at least one filter; "
                             "use clear() to empty the bank")
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM patches WHERE " + " AND ".join(clauses), params)
        return cursor.rowcount

    def delete_ids(self, ids: Sequence[uuid.UUID]) -> int:
        """Delete specific patches by id — the individual-row path."""
        values = [str(i) for i in ids]
        if not values:
            return 0
        removed = 0
        with self.transaction() as conn:
            # Chunked: SQLite caps host parameters per statement (999 by
            # default), and a selection of thousands of rows is plausible.
            for start in range(0, len(values), 500):
                chunk = values[start:start + 500]
                cursor = conn.execute(
                    f"DELETE FROM patches WHERE id IN ({','.join('?' * len(chunk))})",
                    chunk)
                removed += cursor.rowcount
        return removed

    def slides(self) -> list[tuple[str, str, int]]:
        """(slide_path, slide_name, patch count) for everything in the bank."""
        return [(r[0], r[1], r[2]) for r in self._require_conn().execute(
            "SELECT slide_path, slide_name, COUNT(*) FROM patches "
            "GROUP BY slide_path ORDER BY COUNT(*) DESC")]

    def clear(self) -> int:
        with self.transaction() as conn:
            cursor = conn.execute("DELETE FROM patches")
        return cursor.rowcount

    # -- .bank interop ----------------------------------------------------

    def export_bank(self, path: str | Path, patches: Sequence[Patch] | None = None) -> int:
        """Write a `.bank` JSON snapshot (``02-DATA-FORMATS.md`` §5).

        Field names match the macOS export so the files interchange.
        """
        rows = list(patches) if patches is not None else self.fetch()
        document = {
            "version": BANK_FORMAT_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "patchCount": len(rows),
            "patches": [{
                "id": str(p.id).upper(),
                "slidePath": p.slide_path,
                "slideName": p.slide_name,
                "annotationID": str(p.annotation_id).upper(),
                "classification": p.classification,
                "patchX": p.patch_x,
                "patchY": p.patch_y,
                "patchLevel": p.patch_level,
                "patchSizeLevel": p.patch_size_level,
                "featureBase64": base64.b64encode(
                    encode_features(p.features, p.element_type)).decode("ascii"),
                "featureDim": p.feature_dim,
                "featureElementType": p.element_type,
                "extractorIdentity": p.extractor_identity,
                "extractorRevision": p.extractor_revision,
                "whiteFraction": p.white_fraction,
                "nucleusCount": p.nucleus_count,
                "createdAt": p.created_at.isoformat().replace("+00:00", "Z"),
            } for p in rows],
        }
        Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")
        return len(rows)

    def import_bank(self, path: str | Path, replace: bool = False) -> int:
        """Read a `.bank` file.  With *replace*, the bank is emptied first."""
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = document.get("patches") or []
        patches = [self._patch_from_export(entry) for entry in raw
                   if isinstance(entry, dict)]
        patches = [p for p in patches if p is not None]
        if replace:
            self.clear()
        return self.add_many(patches)

    @staticmethod
    def _patch_from_export(entry: dict) -> Patch | None:
        try:
            element = int(entry.get("featureElementType", ELEMENT_FLOAT32))
            dim = int(entry["featureDim"])
            features = decode_features(
                base64.b64decode(entry["featureBase64"]), dim, element)
            created = entry.get("createdAt")
            return Patch(
                id=_uuid_or_new(entry.get("id")),
                slide_path=str(entry.get("slidePath", "")),
                slide_name=str(entry.get("slideName", "")),
                annotation_id=_uuid_or_new(entry.get("annotationID")),
                classification=str(entry.get("classification", "Unlabeled")),
                patch_x=int(entry.get("patchX", 0)),
                patch_y=int(entry.get("patchY", 0)),
                patch_level=int(entry.get("patchLevel", 0)),
                patch_size_level=int(entry.get("patchSizeLevel", 0)),
                features=features,
                # Absent identity means a legacy Vision-print bank.
                extractor_identity=str(entry.get("extractorIdentity") or LEGACY_VISION),
                white_fraction=float(entry.get("whiteFraction", 0.0)),
                nucleus_count=int(entry.get("nucleusCount", -1)),
                created_at=_parse_time(created),
                element_type=element,
            )
        except (KeyError, ValueError, TypeError):
            return None


def _uuid_or_new(value: object) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return uuid.uuid4()


def _parse_time(value: object) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)
