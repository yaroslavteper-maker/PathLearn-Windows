"""Application entry point.

Run with ``pathlearn`` (installed script) or ``python -m pathlearn``.
An optional slide path opens that slide immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .ui.main_window import APP_NAME, ORG_NAME, MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    if len(argv) > 1:
        candidate = Path(argv[1])
        if candidate.exists():
            window.open_slide(candidate)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
