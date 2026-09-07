"""Background tile loading and caching for the slide canvas.

Replaces macOS ``CATiledLayer``: the canvas asks for the tiles covering the
visible rect, gets back whatever is already cached, and repaints when the
missing ones arrive off a worker pool.

Tiles are addressed in **pyramid-level** coordinates — ``(level, col, row)``
covers level-*L* pixels ``[col*TILE, row*TILE]`` to ``+TILE`` — while
``SlideImage.read_region`` wants a level-0 origin, so the loader converts.
That asymmetry is OpenSlide's and is the single easiest thing to get wrong.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap

from ..io.slide import SlideImage

#: Edge length of a tile, in its own level's pixels.
TILE_SIZE = 512
#: Roughly how many tiles to keep decoded.  512*512*4 B = 1 MB each.
DEFAULT_CACHE_TILES = 600


@dataclass(frozen=True, slots=True)
class TileKey:
    level: int
    col: int
    row: int


class _TileSignals(QObject):
    ready = Signal(object, object)  # TileKey, QImage | None


class _TileJob(QRunnable):
    """Reads one tile off the OpenSlide handle on a worker thread."""

    def __init__(self, slide: SlideImage, key: TileKey, signals: _TileSignals,
                 generation: int, is_current: "callable") -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._slide = slide
        self._key = key
        self._signals = signals
        self._generation = generation
        self._is_current = is_current

    def run(self) -> None:
        if not self._is_current(self._generation):
            return  # the slide was closed, or the user moved on
        key = self._key
        downsample = self._slide.level_downsamples[key.level]
        level_dims = self._slide.level_dimensions[key.level]

        # Clip against the level's edge so we never ask for phantom pixels.
        width = min(TILE_SIZE, level_dims.width - key.col * TILE_SIZE)
        height = min(TILE_SIZE, level_dims.height - key.row * TILE_SIZE)
        if width <= 0 or height <= 0:
            self._signals.ready.emit(key, None)
            return

        # x/y are level-0; w/h are level-local.
        x0 = int(round(key.col * TILE_SIZE * downsample))
        y0 = int(round(key.row * TILE_SIZE * downsample))
        try:
            rgb = self._slide.read_region(x0, y0, key.level, width, height)
        except Exception:  # a dead handle or a torn slide must not kill the pool
            self._signals.ready.emit(key, None)
            return

        if self._is_current(self._generation):
            self._signals.ready.emit(key, _to_qimage(rgb))


def _to_qimage(rgb: np.ndarray) -> QImage | None:
    """Wrap an ``(h, w, 3)`` uint8 array as an owned RGB888 QImage."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        return None
    rgb = np.ascontiguousarray(rgb)
    height, width, _ = rgb.shape
    # copy() detaches from the numpy buffer, which is about to be garbage.
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class TileManager(QObject):
    """An LRU cache of decoded tiles, filled by a background thread pool."""

    tile_ready = Signal()

    def __init__(self, max_tiles: int = DEFAULT_CACHE_TILES, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._slide: SlideImage | None = None
        self._cache: OrderedDict[TileKey, QPixmap] = OrderedDict()
        self._pending: set[TileKey] = set()
        self._max_tiles = max_tiles
        self._generation = 0
        self._lock = threading.Lock()

        self._pool = QThreadPool(self)
        # Leave headroom for the UI thread and for later ML work.
        self._pool.setMaxThreadCount(max(2, min(6, QThreadPool.globalInstance().maxThreadCount() - 2)))

        self._signals = _TileSignals()
        self._signals.ready.connect(self._on_tile_ready, Qt.ConnectionType.QueuedConnection)

    # -- lifecycle --------------------------------------------------------

    def set_slide(self, slide: SlideImage | None) -> None:
        """Point at a new slide, abandoning every in-flight and cached tile."""
        with self._lock:
            self._generation += 1
        self._slide = slide
        self._cache.clear()
        self._pending.clear()

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def shutdown(self) -> None:
        with self._lock:
            self._generation += 1
        self._pool.clear()
        self._pool.waitForDone(3000)
        self._cache.clear()
        self._pending.clear()

    # -- access -----------------------------------------------------------

    def tile(self, key: TileKey, *, request: bool = True) -> QPixmap | None:
        """Cached pixmap for *key*, queueing a load when absent."""
        pixmap = self._cache.get(key)
        if pixmap is not None:
            self._cache.move_to_end(key)
            return pixmap
        if request:
            self._request(key)
        return None

    def cached_only(self, key: TileKey) -> QPixmap | None:
        """Look up without queueing — used when scavenging other levels."""
        return self.tile(key, request=False)

    def _request(self, key: TileKey) -> None:
        slide = self._slide
        if slide is None or key in self._pending:
            return
        if not (0 <= key.level < slide.level_count):
            return
        self._pending.add(key)
        self._pool.start(_TileJob(slide, key, self._signals, self._generation, self._is_current))

    def _on_tile_ready(self, key: TileKey, image: QImage | None) -> None:
        self._pending.discard(key)
        if image is None or image.isNull():
            return
        self._cache[key] = QPixmap.fromImage(image)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_tiles:
            self._cache.popitem(last=False)
        self.tile_ready.emit()

    @property
    def pending_count(self) -> int:
        return len(self._pending)
