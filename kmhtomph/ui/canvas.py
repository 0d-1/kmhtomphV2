from __future__ import annotations

from typing import Optional, Tuple
import math

import numpy as np
import cv2

# Import modulaires Qt (meilleur pour les analyseurs & stubs)
from PyQt5 import QtCore, QtGui, QtWidgets

from ..video.overlay import OverlayStyle, render_text_pane_qt


def _bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    """Convertit un BGR OpenCV en QImage RGB888 (copie)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
    return qimg.copy()  # détacher du buffer numpy


class VideoCanvas(QtWidgets.QWidget):
    on_roi_changed = QtCore.pyqtSignal()
    on_overlay_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_bgr: Optional[np.ndarray] = None
        # ROI : centre, taille, angle (deg)
        self._cx = 200
        self._cy = 150
        self._w = 180
        self._h = 90
        self._angle = 0.0

        # Overlay (zone de sortie)
        self._overlay_cx = 200
        self._overlay_cy = 200
        self._overlay_w = 220
        self._overlay_h = 80
        self._overlay_angle = 0.0

        # Aperçu overlay
        self._overlay_text: Optional[str] = None
        self._overlay_style: Optional[OverlayStyle] = None
        self._overlay_qimage: Optional[QtGui.QImage] = None

        # Interaction
        self._dragging_move = False
        self._dragging_resize = False
        self._drag_anchor: Tuple[int, int] | None = None
        self._resize_corner = 0  # 0..3
        self._active_shape = "roi"  # "roi" | "overlay"

        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    # ------------- API publique -------------

    def sizeHint(self) -> QtCore.QSize:
        if self._frame_bgr is not None:
            h, w = self._frame_bgr.shape[:2]
            return QtCore.QSize(w, h)
        return super().sizeHint()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, "frame_bgr doit être BGR HxWx3"
        self._frame_bgr = frame_bgr.copy()
        self.update()

    def set_roi(self, cx: int, cy: int, w: int, h: int, angle_deg: float = 0.0) -> None:
        self._cx, self._cy, self._w, self._h, self._angle = int(cx), int(cy), int(w), int(h), float(angle_deg)
        self._normalize_roi()
        self.on_roi_changed.emit()
        self.update()

    def fit_roi_to_frame(self, margin: int = 20) -> None:
        if self._frame_bgr is None:
            return
        h, w = self._frame_bgr.shape[:2]
        self._cx, self._cy = w // 2, h // 2
        self._w = max(10, w - 2 * margin)
        self._h = max(10, h // 5)
        self._angle = 0.0
        self.on_roi_changed.emit()
        self.update()

    def get_roi(self) -> tuple[int, int, int, int, float]:
        return int(self._cx), int(self._cy), int(self._w), int(self._h), float(self._angle)

    def get_roi_corners(self) -> np.ndarray:
        """
        Retourne les 4 coins du ROI tourné en coordonnées image (float32),
        dans l'ordre : top-left, top-right, bottom-right, bottom-left.

        Convention : le rectangle affiché est tourné **horaire** de `self._angle`.
        """
        cx, cy, w, h, angle = self.get_roi()
        half_w, half_h = w / 2.0, h / 2.0

        # Coins locaux (rectangle centré)
        local = np.array(
            [
                [-half_w, -half_h],  # TL
                [ half_w, -half_h],  # TR
                [ half_w,  half_h],  # BR
                [-half_w,  half_h],  # BL
            ],
            dtype=np.float32,
        )

        # Rotation HORAIRE de `angle` en repère image (y vers le bas)
        # R_clock = [[ cos,  sin],
        #            [-sin,  cos]]
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        R_clock = np.array([[c, s], [-s, c]], dtype=np.float32)

        pts = (local @ R_clock.T) + np.array([cx, cy], dtype=np.float32)
        return pts  # (4,2) float32

    def set_overlay_rect(self, cx: int, cy: int, w: int, h: int, angle_deg: float = 0.0) -> None:
        self._overlay_cx, self._overlay_cy = int(cx), int(cy)
        self._overlay_w, self._overlay_h = int(w), int(h)
        self._overlay_angle = float(angle_deg)
        self._normalize_overlay()
        self.on_overlay_changed.emit()
        self.update()

    def get_overlay_rect(self) -> tuple[int, int, int, int, float]:
        return (
            int(self._overlay_cx),
            int(self._overlay_cy),
            int(self._overlay_w),
            int(self._overlay_h),
            float(self._overlay_angle),
        )

    def set_active_shape(self, shape: str) -> None:
        if shape not in {"roi", "overlay"}:
            raise ValueError("shape doit être 'roi' ou 'overlay'")
        if self._active_shape != shape:
            self._active_shape = shape
            self.update()

    def active_shape(self) -> str:
        return self._active_shape

    def set_overlay_preview(self, text: Optional[str], style: Optional[OverlayStyle]) -> None:
        self._overlay_text = text
        self._overlay_style = style
        if text and style:
            self._overlay_qimage = render_text_pane_qt(text, style)
        else:
            self._overlay_qimage = None
        self.update()

    def refresh_overlay_preview(self) -> None:
        if self._overlay_text and self._overlay_style:
            self._overlay_qimage = render_text_pane_qt(self._overlay_text, self._overlay_style)
        else:
            self._overlay_qimage = None
        self.update()

    # ------------- Dessin -------------

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QtGui.QColor(20, 20, 20))

        # Frame vidéo
        if self._frame_bgr is not None:
            qimg = _bgr_to_qimage(self._frame_bgr)
            p.drawImage(
                0, 0,
                qimg.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation),
            )

        # Transform image -> widget
        sx, sy, ox, oy = self._image_transform()

        roi_center = (ox + self._cx * sx, oy + self._cy * sy)
        roi_size = (self._w * sx, self._h * sy)
        overlay_center = (ox + self._overlay_cx * sx, oy + self._overlay_cy * sy)
        overlay_size = (self._overlay_w * sx, self._overlay_h * sy)

        # Dessiner ROI
        roi_color = QtGui.QColor(255, 180, 0)
        roi_pen = QtGui.QPen(roi_color, 2.5 if self._active_shape == "roi" else 2.0, QtCore.Qt.SolidLine)
        p.setPen(roi_pen)
        p.setBrush(QtCore.Qt.NoBrush)
        self._draw_rotated_rect(p, roi_center, roi_size, self._angle, handles=True, color=roi_color)

        # Dessiner zone overlay
        overlay_color = QtGui.QColor(90, 210, 255)
        overlay_pen_style = QtCore.Qt.SolidLine if self._active_shape == "overlay" else QtCore.Qt.DashLine
        overlay_pen = QtGui.QPen(overlay_color, 2.5 if self._active_shape == "overlay" else 2.0, overlay_pen_style)
        p.setPen(overlay_pen)
        p.setBrush(QtCore.Qt.NoBrush)
        self._draw_rotated_rect(
            p,
            overlay_center,
            overlay_size,
            self._overlay_angle,
            handles=False,
            color=overlay_color,
        )

        # Dessiner le texte overlay après avoir réinitialisé la transform
        if self._overlay_qimage is not None and self._overlay_qimage.width() > 0 and self._overlay_qimage.height() > 0:
            p.save()
            t = QtGui.QTransform()
            t.translate(*overlay_center)
            t.rotate(-self._overlay_angle)
            p.setTransform(t, combine=False)

            target_w = max(1.0, overlay_size[0])
            target_h = max(1.0, overlay_size[1])
            img_w = float(self._overlay_qimage.width())
            img_h = float(self._overlay_qimage.height())
            scale = min(target_w / img_w, target_h / img_h) if img_w > 0 and img_h > 0 else 1.0
            scale = min(scale, 1.0)
            scale = max(scale, 1e-3)
            draw_w = img_w * scale
            draw_h = img_h * scale
            rect = QtCore.QRectF(-draw_w / 2.0, -draw_h / 2.0, draw_w, draw_h)
            p.drawImage(rect, self._overlay_qimage)
            p.restore()

        if self._active_shape == "overlay":
            handle_pen = QtGui.QPen(overlay_color, 2.5, QtCore.Qt.SolidLine)
            p.setPen(handle_pen)
            self._draw_rotated_rect(
                p,
                overlay_center,
                overlay_size,
                self._overlay_angle,
                handles=True,
                color=overlay_color,
                draw_rect=False,
            )

        p.end()

    # ------------- Interaction souris -------------

    def mousePressEvent(self, ev) -> None:
        if self._frame_bgr is None:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            self._dragging_move = True
            self._drag_anchor = (ev.x(), ev.y())
        elif ev.button() == QtCore.Qt.RightButton:
            self._dragging_resize = True
            self._drag_anchor = (ev.x(), ev.y())
            self._resize_corner = self._closest_corner_for_shape(ev.pos(), self._active_shape)
        self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev) -> None:
        if self._frame_bgr is None or self._drag_anchor is None:
            return

        dx = ev.x() - self._drag_anchor[0]
        dy = ev.y() - self._drag_anchor[1]

        sx, sy, ox, oy = self._image_transform()
        ddx = dx / max(1e-6, sx)
        ddy = dy / max(1e-6, sy)

        if self._active_shape == "overlay":
            angle = self._overlay_angle
        else:
            angle = self._angle

        # rotation inverse pour delta local
        rad = math.radians(angle)
        cos, sin = math.cos(rad), math.sin(rad)
        local_dx = cos * ddx + sin * ddy
        local_dy = -sin * ddx + cos * ddy

        if self._dragging_move:
            if self._active_shape == "overlay":
                self._overlay_cx += local_dx
                self._overlay_cy += local_dy
            else:
                self._cx += local_dx
                self._cy += local_dy
        elif self._dragging_resize:
            kx = 1 if self._resize_corner in (1, 2) else -1
            ky = 1 if self._resize_corner in (2, 3) else -1
            if self._active_shape == "overlay":
                self._overlay_w += kx * 2 * local_dx
                self._overlay_h += ky * 2 * local_dy
            else:
                self._w += kx * 2 * local_dx
                self._h += ky * 2 * local_dy

        self._drag_anchor = (ev.x(), ev.y())
        self._normalize_shape(self._active_shape)
        self._emit_shape_changed(self._active_shape)
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._dragging_move = False
        self._dragging_resize = False
        self._drag_anchor = None
        self.setCursor(QtCore.Qt.ArrowCursor)

    def wheelEvent(self, ev) -> None:
        step = 1.0 if (ev.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.ControlModifier)) else 3.0
        delta = ev.angleDelta().y() / 120.0
        if self._active_shape == "overlay":
            self._overlay_angle = (self._overlay_angle + delta * step) % 360.0
        else:
            self._angle = (self._angle + delta * step) % 360.0
        self._emit_shape_changed(self._active_shape)
        self.update()

    def mouseDoubleClickEvent(self, ev) -> None:
        sx, sy, ox, oy = self._image_transform()
        ix = (ev.x() - ox) / max(1e-6, sx)
        iy = (ev.y() - oy) / max(1e-6, sy)
        if self._active_shape == "overlay":
            self._overlay_cx, self._overlay_cy = ix, iy
        else:
            self._cx, self._cy = ix, iy
        self._normalize_shape(self._active_shape)
        self._emit_shape_changed(self._active_shape)
        self.update()

    # ------------- Helpers internes -------------

    def _image_transform(self) -> Tuple[float, float, float, float]:
        """Retourne (sx, sy, ox, oy) pour le mapping image->widget."""
        if self._frame_bgr is None:
            return 1.0, 1.0, 0.0, 0.0
        h, w = self._frame_bgr.shape[:2]
        if w <= 0 or h <= 0:
            return 1.0, 1.0, 0.0, 0.0
        rw = self.width() / w
        rh = self.height() / h
        s = min(rw, rh)
        sx = sy = s
        ox = (self.width() - w * s) / 2.0
        oy = (self.height() - h * s) / 2.0
        return sx, sy, ox, oy

    def _normalize_roi(self) -> None:
        if self._frame_bgr is None:
            return
        h, w = self._frame_bgr.shape[:2]
        self._w = max(10, min(self._w, w))
        self._h = max(10, min(self._h, h))
        self._cx = float(max(0, min(self._cx, w)))
        self._cy = float(max(0, min(self._cy, h)))

    def _normalize_overlay(self) -> None:
        if self._frame_bgr is None:
            return
        h, w = self._frame_bgr.shape[:2]
        self._overlay_w = max(10, min(self._overlay_w, w))
        self._overlay_h = max(10, min(self._overlay_h, h))
        self._overlay_cx = float(max(0, min(self._overlay_cx, w)))
        self._overlay_cy = float(max(0, min(self._overlay_cy, h)))

    def _normalize_shape(self, shape: str) -> None:
        if shape == "overlay":
            self._normalize_overlay()
        else:
            self._normalize_roi()

    def _emit_shape_changed(self, shape: str) -> None:
        if shape == "overlay":
            self.on_overlay_changed.emit()
        else:
            self.on_roi_changed.emit()

    def _closest_corner(self, pos) -> int:
        """Renvoie l'index du coin le plus proche dans l'espace écran (0..3)."""
        return self._closest_corner_for_shape(pos, self._active_shape)

    # ------------- Helpers internes -------------

    def _draw_rotated_rect(
        self,
        painter: QtGui.QPainter,
        center: Tuple[float, float],
        size: Tuple[float, float],
        angle_deg: float,
        *,
        handles: bool,
        color: QtGui.QColor,
        draw_rect: bool = True,
    ) -> None:
        w, h = size
        if w <= 0 or h <= 0:
            return
        painter.save()
        t = QtGui.QTransform()
        t.translate(center[0], center[1])
        t.rotate(-angle_deg)
        painter.setTransform(t, combine=False)
        rect = QtCore.QRectF(-w / 2.0, -h / 2.0, w, h)
        if draw_rect:
            painter.drawRect(rect)
        if handles:
            painter.setBrush(QtGui.QBrush(color))
            r = 6.0
            for (px, py) in [
                (-w / 2.0, -h / 2.0),
                (w / 2.0, -h / 2.0),
                (w / 2.0, h / 2.0),
                (-w / 2.0, h / 2.0),
            ]:
                painter.drawEllipse(QtCore.QPointF(px, py), r, r)
            painter.setBrush(QtCore.Qt.NoBrush)
        painter.restore()

    def _closest_corner_for_shape(self, pos, shape: str) -> int:
        sx, sy, ox, oy = self._image_transform()
        if shape == "overlay":
            cx = ox + self._overlay_cx * sx
            cy = oy + self._overlay_cy * sy
            w = self._overlay_w * sx
            h = self._overlay_h * sy
            angle = self._overlay_angle
        else:
            cx = ox + self._cx * sx
            cy = oy + self._cy * sy
            w = self._w * sx
            h = self._h * sy
            angle = self._angle
        rad = math.radians(angle)
        cos, sin = math.cos(rad), math.sin(rad)
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        pts = []
        for (x, y) in corners:
            rx = cos * x - sin * y
            ry = sin * x + cos * y
            pts.append((cx + rx, cy + ry))
        px, py = pos.x(), pos.y()
        dists = [(i, (px - x)**2 + (py - y)**2) for i, (x, y) in enumerate(pts)]
        dists.sort(key=lambda t: t[1])
        return dists[0][0]
