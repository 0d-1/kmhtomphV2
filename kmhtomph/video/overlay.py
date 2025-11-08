"""
Rendu de l’overlay texte (Qt) et insertion dans des frames OpenCV.

Cette unité est indépendante de la GUI : elle ne crée aucune fenêtre ni widget.
Elle s’appuie sur Qt pour la typographie (antialiasing, outline, etc.) et
renvoie des images prêtes à coller dans une frame BGR (OpenCV).

Expose :
- OverlayStyle : paramètres de style
- render_text_pane_qt(text, style) -> QImage (premultiplied ARGB32)
- paste_text_rotated(frame_bgr, qimage, center, angle_deg) -> None (in-place)
- format_speed_text(value, unit="mph", decimals=0) -> str
- draw_speed_overlay(frame_bgr, value, center, angle_deg, style, unit="mph", decimals=0) -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import numpy as np
import cv2

from PyQt5.QtGui import (
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QColor,
    QPen,
    QBrush,
    QPainterPath,
)
from PyQt5.QtCore import Qt, QSize

from ..constants import (
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_POINT_SIZE,
    DEFAULT_TEXT_PADDING_PX,
    DEFAULT_OUTLINE_THICKNESS_PX,
    DEFAULT_FILL_OPACITY,
    DEFAULT_TEXT_COLOR_RGBA,
    DEFAULT_OUTLINE_COLOR_RGBA,
    DEFAULT_BG_COLOR_RGBA,
)


@dataclass(frozen=True)
class OverlayStyle:
    font_family: str = DEFAULT_FONT_FAMILY
    font_point_size: int = DEFAULT_FONT_POINT_SIZE
    text_color_rgba: Tuple[int, int, int, int] = DEFAULT_TEXT_COLOR_RGBA
    outline_color_rgba: Tuple[int, int, int, int] = DEFAULT_OUTLINE_COLOR_RGBA
    outline_thickness_px: int = DEFAULT_OUTLINE_THICKNESS_PX
    bg_color_rgba: Tuple[int, int, int, int] = DEFAULT_BG_COLOR_RGBA
    text_padding_px: int = DEFAULT_TEXT_PADDING_PX
    fill_opacity: float = DEFAULT_FILL_OPACITY  # 0..1
    quality_scale: float = 1.0  # ≥1.0 : suréchantillonnage pour la netteté


def _qt_color(rgba: Tuple[int, int, int, int]) -> QColor:
    r, g, b, a = rgba
    return QColor(r, g, b, a)


def _tight_text_metrics(font: QFont, text: str, padding: int) -> tuple[int, int, float, float]:
    """Calcule taille + origine de dessin à partir du rect "serré"."""
    metrics = QFontMetricsF(font)
    # tightBoundingRect inclut les dépassements négatifs éventuels (gauche/haut)
    tight = metrics.tightBoundingRect(text)
    width = tight.width()
    height = tight.height()
    left = tight.left()
    top = tight.top()

    advance = metrics.horizontalAdvance(text)
    if advance > width:
        width = advance
        left = min(left, 0.0)

    if height <= 0:
        ascent = metrics.ascent()
        descent = metrics.descent()
        height = ascent + descent
        top = -ascent

    width = max(1, int(math.ceil(width)) + padding * 2)
    height = max(1, int(math.ceil(height)) + padding * 2)
    origin_x = padding - left
    origin_y = padding - top
    return width, height, origin_x, origin_y


def render_text_pane_qt(text: str, style: OverlayStyle) -> QImage:
    """
    Rend le texte dans un QImage ARGB32 premultiplied, paddé et avec fond.
    - Retour : QImage (format=QImage.Format_ARGB32_Premultiplied)
    """
    if not text:
        text = " "  # éviter largeur/hauteur nulles

    # Préparer police & suréchantillonnage
    base_font = QFont(style.font_family)
    base_font.setPointSize(style.font_point_size)
    base_font.setStyleStrategy(QFont.PreferAntialias)

    pad_base = style.text_padding_px
    w_base, h_base, base_x, base_y = _tight_text_metrics(base_font, text, pad_base)

    quality = max(1.0, float(getattr(style, "quality_scale", 1.0)))

    if math.isclose(quality, 1.0, rel_tol=1e-6):
        font = base_font
        w = w_base
        h = h_base
        origin_x = base_x
        origin_y = base_y
    else:
        font = QFont(style.font_family)
        font.setPointSizeF(style.font_point_size * quality)
        font.setStyleStrategy(QFont.PreferAntialias)
        pad = int(math.ceil(pad_base * quality))
        w, h, origin_x, origin_y = _tight_text_metrics(font, text, pad)

    img = QImage(QSize(w, h), QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)

    painter = QPainter(img)
    painter.setRenderHints(
        QPainter.Antialiasing
        | QPainter.TextAntialiasing
        | QPainter.SmoothPixmapTransform,
        True,
    )
    painter.setFont(font)

    # Fond
    if style.fill_opacity > 0:
        bg = _qt_color(style.bg_color_rgba)
        bg.setAlphaF(max(0.0, min(1.0, style.fill_opacity)) * (bg.alphaF()))
        painter.fillRect(0, 0, w, h, QBrush(bg))

    # Texte avec outline via QPainterPath pour un meilleur rendu
    path = QPainterPath()
    path.addText(origin_x, origin_y, font, text)

    if style.outline_thickness_px > 0:
        outline_width = style.outline_thickness_px
        if not math.isclose(quality, 1.0, rel_tol=1e-6):
            outline_width *= quality
        painter.setPen(
            QPen(
                _qt_color(style.outline_color_rgba),
                outline_width,
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(_qt_color(style.text_color_rgba)))
    painter.drawPath(path)

    painter.end()

    if not math.isclose(quality, 1.0, rel_tol=1e-6):
        img = img.scaled(
            w_base,
            h_base,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )

    return img


def _qimage_to_bgra_numpy(img: QImage) -> np.ndarray:
    """Convertit un QImage en np.ndarray BGRA (uint8) avec couleurs prémultipliées."""
    assert img.format() in (
        QImage.Format_ARGB32,
        QImage.Format_ARGB32_Premultiplied,
        QImage.Format_RGBA8888,
        QImage.Format_RGBA8888_Premultiplied,
    ), "QImage doit être ARGB32/RGBA8888"
    if img.format() != QImage.Format_ARGB32_Premultiplied:
        img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    w = img.width()
    h = img.height()
    ptr = img.constBits()
    ptr.setsize(h * img.bytesPerLine())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, img.bytesPerLine() // 4, 4))[:, :w, :]
    # Format Qt : ARGB prémultiplié -> convertir en BGRA prémultiplié
    bgra = arr[:, :, [2, 1, 0, 3]].copy()
    return bgra


def _alpha_blend_bgra(dst_bgr: np.ndarray, src_bgra: np.ndarray, top_left_xy: Tuple[int, int]) -> None:
    """
    Colle src_bgra (BGRA) sur dst_bgr (BGR) en place avec alpha.
    top_left_xy : (x, y) destination
    """
    x, y = top_left_xy
    h, w = src_bgra.shape[:2]
    H, W = dst_bgr.shape[:2]

    # Clipping si dépasse l'image
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return

    src_roi = src_bgra[(y0 - y):(y1 - y), (x0 - x):(x1 - x), :]
    dst_roi = dst_bgr[y0:y1, x0:x1, :]

    alpha = src_roi[:, :, 3:4].astype(np.float32) / 255.0
    if not np.any(alpha > 0):
        return

    # Les composantes sont déjà prémultipliées par alpha (voir _qimage_to_bgra_numpy)
    src_rgb = src_roi[:, :, :3].astype(np.float32) / 255.0
    dst_rgb = dst_roi.astype(np.float32) / 255.0

    out = src_rgb + dst_rgb * (1.0 - alpha)
    dst_roi[:] = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def paste_text_rotated(
    frame_bgr: np.ndarray,
    text_qimage: QImage,
    center: Tuple[int, int],
    angle_deg: float,
    *,
    target_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Colle le QImage (texte) dans la frame BGR autour de `center` avec rotation.
    - frame_bgr : np.ndarray HxWx3 (BGR)
    - text_qimage : panneau texte rendu par render_text_pane_qt
    - center : (cx, cy) en pixels dans la frame
    - angle_deg : rotation horaire en degrés
    Modifie `frame_bgr` in-place.
    """
    pane_bgra = _qimage_to_bgra_numpy(text_qimage)  # HxWx4 (BGRA)
    h, w = pane_bgra.shape[:2]

    if target_size is not None:
        tw, th = target_size
        tw = int(round(tw))
        th = int(round(th))
        if tw > 0 and th > 0 and w > 0 and h > 0:
            scale = min(tw / float(w), th / float(h))
            scale = min(scale, 1.0)
            if scale > 0 and not math.isclose(scale, 1.0, rel_tol=1e-3, abs_tol=1e-3):
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                pane_bgra = cv2.resize(pane_bgra, (new_w, new_h), interpolation=interp)
                h, w = pane_bgra.shape[:2]

    # Construire image plus grande pour éviter le crop lors de la rotation
    diag = int(np.ceil(np.hypot(h, w)))
    canvas = np.zeros((diag, diag, 4), dtype=np.uint8)
    ox = (diag - w) // 2
    oy = (diag - h) // 2
    canvas[oy:oy + h, ox:ox + w, :] = pane_bgra

    # Rotation autour du centre du canvas
    M = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # Positionner le coin supérieur gauche pour que le centre coïncide
    top_left = (int(center[0] - rotated.shape[1] // 2), int(center[1] - rotated.shape[0] // 2))
    _alpha_blend_bgra(frame_bgr, rotated, top_left)


def format_speed_text(value: float, unit: str = "mph", decimals: int = 0) -> str:
    """Retourne la chaîne formatée pour une vitesse (ex: "54 mph")."""
    unit = unit.strip() or "mph"
    decimals = max(0, int(decimals))
    if decimals <= 0:
        return f"{int(round(value))} {unit}"
    formatted = f"{value:.{decimals}f}"
    return f"{formatted} {unit}"


def draw_speed_overlay(
    frame_bgr: np.ndarray,
    value: float,
    center: Tuple[int, int],
    angle_deg: float,
    style: OverlayStyle,
    unit: str = "mph",
    decimals: int = 0,
    text: Optional[str] = None,
    *,
    target_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Dessine la vitesse formatée sur la frame et retourne le texte affiché.

    Cette fonction encapsule ``render_text_pane_qt`` et ``paste_text_rotated`` pour
    garantir que l’export et l’aperçu emploient exactement le même rendu
    typographique que l’application Qt.

    Paramètres supplémentaires :
    - unit : suffixe unité (par défaut "mph").
    - decimals : nombre de décimales lorsque ``text`` n’est pas fourni.
    - text : chaîne déjà formatée à dessiner (permet de garantir 1:1 avec
      l’affichage en UI ou d’utiliser une autre langue/unité).
    """

    if text is None:
        text = format_speed_text(value, unit=unit, decimals=decimals)
    pane = render_text_pane_qt(text, style)
    paste_text_rotated(frame_bgr, pane, center, angle_deg, target_size=target_size)
    return text
