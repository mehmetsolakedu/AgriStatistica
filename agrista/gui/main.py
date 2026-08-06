"""Agrista GUI giriş noktası (`agrista-gui`)."""

from __future__ import annotations

import sys


def main() -> None:
    from PySide6.QtWidgets import QApplication

    from agrista.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Agrista")
    pencere = MainWindow()
    pencere.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
