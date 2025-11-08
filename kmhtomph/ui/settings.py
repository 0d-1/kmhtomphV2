"""
Boîte de dialogue simple pour configurer le chemin de Tesseract.

Expose :
- class SettingsDialog(QDialog)
    .tesseract_path -> str
Usage :
    dlg = SettingsDialog(self, initial_path=current_path)
    if dlg.exec_() == QDialog.Accepted:
        save(dlg.tesseract_path)
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox
)

from ..ocr.tesseract import auto_locate_tesseract


class SettingsDialog(QDialog):
    def __init__(self, parent=None, initial_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres OCR")
        self.setModal(True)

        self._edit = QLineEdit(self)
        if initial_path:
            self._edit.setText(initial_path)

        pick_btn = QPushButton("Parcourir…", self)
        test_btn = QPushButton("Tester", self)
        ok_btn = QPushButton("OK", self)
        cancel_btn = QPushButton("Annuler", self)

        pick_btn.clicked.connect(self._on_pick)
        test_btn.clicked.connect(self._on_test)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(QLabel("Chemin Tesseract :", self))
        row.addWidget(self._edit)
        row.addWidget(pick_btn)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(test_btn)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)

        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addLayout(btns)
        self.setLayout(lay)
        self.resize(600, 120)

    @property
    def tesseract_path(self) -> str:
        return self._edit.text().strip()

    def _on_pick(self):
        start = self.tesseract_path or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Sélectionner l'exécutable Tesseract", start)
        if path:
            self._edit.setText(path)

    def _on_test(self):
        path = self.tesseract_path or None
        try:
            auto_locate_tesseract(path)
            # si OK, pytesseract utilisera ce chemin au prochain appel ; on vérifie l'existence
            if path and not os.path.exists(path):
                raise FileNotFoundError(path)
            QMessageBox.information(self, "Test Tesseract", "Configuration appliquée.\nUn test réel se fera à la première lecture OCR.")
        except Exception as e:
            QMessageBox.warning(self, "Test Tesseract", f"Échec de configuration : {e}")
