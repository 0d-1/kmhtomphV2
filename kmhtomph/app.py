"""
Point d’entrée de l’application kmh→mph OCR (Qt).

Lance une fenêtre `MainWindow`. Optionnellement, un chemin vidéo peut être
fourni en argument pour l’ouvrir directement :

    python -m kmhtomph.app /chemin/vers/video.mp4
"""

from __future__ import annotations

import sys
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from .ui.mainwindow import MainWindow
from .video.io import VideoReader


def _setup_qt_app(argv: list[str]) -> QApplication:
    # Haute-DPI : améliore le rendu sur écrans rétina/4K
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv)
    app.setApplicationName("kmh→mph OCR")
    app.setOrganizationName("kmhtomph")
    return app


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    app = _setup_qt_app(argv)
    win = MainWindow()
    win.show()

    # Argument optionnel : ouvrir une vidéo au démarrage
    if len(argv) >= 2:
        path = argv[1]
        try:
            reader = VideoReader(path)
            reader.open()
            # lire une frame pour initialiser l’affichage
            ok, frame = reader.read()
            if ok and frame is not None:
                win.reader = reader  # garder ce reader
                win.canvas.set_frame(frame)
                win.canvas.fit_roi_to_frame()
            else:
                reader.release()
        except Exception:
            # on ignore l’erreur d’ouverture silencieusement et on démarre vide
            pass

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
