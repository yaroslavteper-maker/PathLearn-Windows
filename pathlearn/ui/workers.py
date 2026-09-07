"""Background work off the UI thread.

Extraction takes seconds to minutes; running it on the UI thread would freeze
the canvas and make the app look hung.  The macOS build used detached tasks with
progress callbacks and a main-actor hop for DB writes; Qt's equivalent is a
``QThread`` with signals, which marshal to the UI thread automatically.

Cancellation is cooperative: the worker sets a flag, the pipeline checks it
between batches, and the run unwinds after finishing whatever patch is in
flight.  Nothing is killed mid-write, so the bank is never left half-updated.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


class Worker(QObject):
    """Runs one callable on a worker thread, reporting progress and result.

    The callable receives ``progress`` and ``should_cancel`` keyword arguments
    when it accepts them, so pipeline functions need no Qt awareness.
    """

    progress = Signal(int, int, str)     # done, total, message
    finished = Signal(object)            # result, or None on failure
    failed = Signal(str)                 # human-readable message
    traceback_ready = Signal(str)        # full traceback, for the details pane

    def __init__(self, job: Callable[..., Any], **kwargs: Any) -> None:
        super().__init__()
        self._job = job
        self._kwargs = kwargs
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            result = self._job(
                progress=lambda done, total, message: self.progress.emit(done, total, message),
                should_cancel=lambda: self._cancelled,
                **self._kwargs,
            )
        except Exception as exc:
            self.traceback_ready.emit(traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            self.finished.emit(None)
            return
        self.finished.emit(result)


class BackgroundTask:
    """Owns a :class:`Worker` and the thread it runs on.

    Keep a reference for the task's lifetime — a garbage-collected ``QThread``
    takes its worker down with it, which is a classic PySide crash.
    """

    #: Tasks whose worker has reported but whose thread has not yet exited.
    #: A ``QThread`` collected while still running takes the process down, so
    #: the task must stay referenced until ``thread.finished`` actually fires —
    #: which is strictly later than ``worker.finished``.
    _in_flight: "set[BackgroundTask]" = set()

    def __init__(self, job: Callable[..., Any], **kwargs: Any) -> None:
        self.thread = QThread()
        self.worker = Worker(job, **kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self._retire)

    def _retire(self) -> None:
        BackgroundTask._in_flight.discard(self)

    def start(self) -> None:
        # Self-reference until the thread genuinely exits, so a caller that
        # drops its handle on completion cannot destroy a live QThread.
        BackgroundTask._in_flight.add(self)
        self.thread.start()

    @property
    def is_finished(self) -> bool:
        """True once the thread has fully exited, not merely reported a result."""
        return self.thread.isFinished()

    def cancel(self) -> None:
        self.worker.cancel()

    def wait(self, milliseconds: int = 30_000) -> bool:
        """Block until the thread exits.  Only for shutdown paths."""
        return self.thread.wait(milliseconds)

    @property
    def is_running(self) -> bool:
        return self.thread.isRunning()
