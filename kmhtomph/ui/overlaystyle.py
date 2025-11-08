from __future__ import annotations

from dataclasses import replace
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QColorDialog,
    QFontComboBox,
    QWidget,
    QDoubleSpinBox,
)

from ..video.overlay import OverlayStyle, render_text_pane_qt


def _rgba_from_qcolor(color: QColor) -> tuple[int, int, int, int]:
    return color.red(), color.green(), color.blue(), color.alpha()


def _qcolor_from_rgba(rgba: tuple[int, int, int, int]) -> QColor:
    r, g, b, a = rgba
    return QColor(r, g, b, a)


class OverlayStyleDialog(QDialog):
    """Boîte de dialogue pour éditer le style de texte de la zone de sortie."""

    def __init__(self, parent: Optional[QWidget] = None, style: Optional[OverlayStyle] = None):
        super().__init__(parent)
        self.setWindowTitle("Style de la zone de sortie")
        self.setModal(True)

        self._initial_style = style or OverlayStyle()

        self._font_combo = QFontComboBox(self)
        self._font_combo.setCurrentFont(QFont(self._initial_style.font_family))

        self._font_size = QSpinBox(self)
        self._font_size.setRange(8, 128)
        self._font_size.setValue(self._initial_style.font_point_size)

        self._quality_spin = QDoubleSpinBox(self)
        self._quality_spin.setRange(1.0, 4.0)
        self._quality_spin.setSingleStep(0.25)
        self._quality_spin.setDecimals(2)
        self._quality_spin.setValue(max(1.0, min(4.0, getattr(self._initial_style, "quality_scale", 1.0))))

        self._text_color = _qcolor_from_rgba(self._initial_style.text_color_rgba)
        self._bg_color = _qcolor_from_rgba(self._initial_style.bg_color_rgba)

        self._text_btn = QPushButton(self)
        self._text_btn.setText("Couleur du texte…")
        self._text_btn.clicked.connect(self._pick_text_color)

        self._bg_btn = QPushButton(self)
        self._bg_btn.setText("Couleur du fond…")
        self._bg_btn.clicked.connect(self._pick_bg_color)

        self._opacity_slider = QSlider(Qt.Horizontal, self)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(int(round(self._initial_style.fill_opacity * 100)))

        self._opacity_label = QLabel(self)
        self._preview_label = QLabel(self)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumHeight(100)

        btn_box = QHBoxLayout()
        btn_box.addWidget(self._text_btn)
        btn_box.addWidget(self._bg_btn)

        font_box = QHBoxLayout()
        font_box.addWidget(QLabel("Police :", self))
        font_box.addWidget(self._font_combo, 1)
        font_box.addSpacing(10)
        font_box.addWidget(QLabel("Taille :", self))
        font_box.addWidget(self._font_size)

        quality_box = QHBoxLayout()
        quality_box.addWidget(QLabel("Netteté :", self))
        quality_box.addWidget(self._quality_spin)
        quality_box.addStretch(1)

        opacity_box = QHBoxLayout()
        opacity_box.addWidget(QLabel("Opacité du fond :", self))
        opacity_box.addWidget(self._opacity_slider, 1)
        opacity_box.addSpacing(8)
        opacity_box.addWidget(self._opacity_label)

        action_box = QHBoxLayout()
        action_box.addStretch(1)
        ok_btn = QPushButton("OK", self)
        cancel_btn = QPushButton("Annuler", self)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        action_box.addWidget(ok_btn)
        action_box.addWidget(cancel_btn)

        lay = QVBoxLayout(self)
        lay.addLayout(font_box)
        lay.addLayout(quality_box)
        lay.addLayout(btn_box)
        lay.addLayout(opacity_box)
        lay.addWidget(QLabel("Aperçu :", self))
        lay.addWidget(self._preview_label, 1)
        lay.addLayout(action_box)

        self._font_combo.currentFontChanged.connect(self._update_preview)
        self._font_size.valueChanged.connect(self._update_preview)
        self._quality_spin.valueChanged.connect(self._update_preview)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)

        self._update_buttons()
        self._on_opacity_changed(self._opacity_slider.value())
        self._update_preview()

        self.resize(460, 360)

    @property
    def result_style(self) -> OverlayStyle:
        return replace(
            self._initial_style,
            font_family=self._font_combo.currentFont().family(),
            font_point_size=self._font_size.value(),
            text_color_rgba=_rgba_from_qcolor(self._text_color),
            bg_color_rgba=_rgba_from_qcolor(self._bg_color),
            fill_opacity=max(0.0, min(1.0, self._opacity_slider.value() / 100.0)),
            quality_scale=float(self._quality_spin.value()),
        )

    def _on_opacity_changed(self, value: int) -> None:
        self._opacity_label.setText(f"{value}%")
        self._update_preview()

    def _pick_text_color(self) -> None:
        color = QColorDialog.getColor(self._text_color, self, "Choisir la couleur du texte", QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self._text_color = color
            self._update_buttons()
            self._update_preview()

    def _pick_bg_color(self) -> None:
        color = QColorDialog.getColor(self._bg_color, self, "Choisir la couleur du fond", QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self._bg_color = color
            self._update_buttons()
            self._update_preview()

    def _update_buttons(self) -> None:
        def set_button_color(btn: QPushButton, color: QColor) -> None:
            btn.setStyleSheet(
                "QPushButton {"
                f" background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()});"
                " border: 1px solid #444;"
                " padding: 6px;"
                "}"
            )

        set_button_color(self._text_btn, self._text_color)
        set_button_color(self._bg_btn, self._bg_color)

    def _update_preview(self) -> None:
        preview_style = self.result_style
        image = render_text_pane_qt("123 mph", preview_style)
        pix = QPixmap.fromImage(image)
        scaled = pix.scaled(
            self._preview_label.width() or pix.width(),
            self._preview_label.height() or pix.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_preview()
