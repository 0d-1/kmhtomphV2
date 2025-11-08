"""
Entrées vidéo : ouverture robuste, lecture frame-à-frame, seek approximatif.

Expose :
- open_capture(path_or_index) -> cv2.VideoCapture
- get_props(cap) -> dict(fps, frame_count, width, height, fourcc)
- seek_msec(cap, ms) -> bool       # tentative rapide (CV_CAP_PROP_POS_MSEC), sinon scan
- read_frame(cap) -> (ok:bool, frame_bgr:np.ndarray|None)
- VideoReader : façade pratique avec .read(), .seek_msec(), .fps, .frame_count…
               + get_pos_msec(), get_pos_frame(), set_pos_frame()

Notes :
- Le seek par millisecondes n’est pas fiable sur tous les backends; on tente puis on
  retombe sur un scan avant si nécessaire.
- Aucune dépendance à Qt ici.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import os
import sys
import math
import cv2
import numpy as np


# ---------------------------------
# Ouverture avec fallbacks backends
# ---------------------------------

def _try_open_with_backend(path_or_index, backend) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(path_or_index, backend) if backend is not None else cv2.VideoCapture(path_or_index)
    if cap is not None and cap.isOpened():
        return cap
    try:
        if cap:
            cap.release()
    except Exception:
        pass
    return None


def open_capture(path_or_index) -> cv2.VideoCapture:
    """
    Ouvre une capture vidéo depuis un chemin ou un index de caméra.
    Essaie plusieurs backends selon l’OS pour la robustesse.
    """
    backends = [None]  # laisser OpenCV choisir d'abord
    if sys.platform.startswith("win"):
        backends += [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_FFMPEG]
    elif sys.platform == "darwin":
        backends += [cv2.CAP_AVFOUNDATION, cv2.CAP_FFMPEG]
    else:
        backends += [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]

    for be in backends:
        cap = _try_open_with_backend(path_or_index, be)
        if cap is not None:
            return cap
    raise RuntimeError(f"Impossible d’ouvrir la source vidéo : {path_or_index!r}")


# ------------------------
# Propriétés et utilitaires
# ------------------------

def _safe_get(cap: cv2.VideoCapture, prop: int, default: float = 0.0) -> float:
    try:
        v = float(cap.get(prop))
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def get_props(cap: cv2.VideoCapture) -> Dict[str, float]:
    fps = _safe_get(cap, cv2.CAP_PROP_FPS, 0.0)
    frame_count = _safe_get(cap, cv2.CAP_PROP_FRAME_COUNT, 0.0)
    width = int(_safe_get(cap, cv2.CAP_PROP_FRAME_WIDTH, 0))
    height = int(_safe_get(cap, cv2.CAP_PROP_FRAME_HEIGHT, 0))
    fourcc = int(_safe_get(cap, cv2.CAP_PROP_FOURCC, 0))
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fourcc": fourcc,
    }


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    ok, frame = cap.read()
    if not ok:
        return False, None
    # Assurer BGR uint8
    if frame is None or frame.ndim < 2:
        return False, None
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    return True, frame


# -------------
# Seek / Position
# -------------

def _set_pos_msec(cap: cv2.VideoCapture, ms: float) -> bool:
    # Certains backends ignorent POS_MSEC. On essaie, puis on valide en lisant.
    return bool(cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(ms))))


def _approx_frame_from_ms(fps: float, ms: float) -> int:
    if fps <= 0:
        return 0
    return int(round((ms / 1000.0) * fps))


def _set_pos_frames(cap: cv2.VideoCapture, frame_idx: int) -> bool:
    return bool(cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx))))


def _get_pos_frames(cap: cv2.VideoCapture) -> int:
    return int(_safe_get(cap, cv2.CAP_PROP_POS_FRAMES, 0.0))


def seek_msec(cap: cv2.VideoCapture, ms: float) -> bool:
    """
    Tente de se placer à ~ms millisecondes.
    Stratégie :
    1) set POS_MSEC
    2) si échec, convertir en index de frame via FPS et set POS_FRAMES
    3) si encore fragile, reculer un peu et scanner en avant
    """
    ms = max(0.0, float(ms))
    props = get_props(cap)
    fps = props["fps"]

    # Essai direct POS_MSEC
    if _set_pos_msec(cap, ms):
        return True

    # Essai via frame index
    idx = _approx_frame_from_ms(fps, ms)
    if _set_pos_frames(cap, idx):
        return True

    # Fallback : reculer de N frames puis scanner
    back = max(10, int(round((fps if fps > 0 else 25) * 0.5)))
    start = max(0, idx - back)
    if not _set_pos_frames(cap, start):
        return False

    # Scanner jusqu’à dépasser ms
    for _ in range(back * 2):
        pos_ms = _safe_get(cap, cv2.CAP_PROP_POS_MSEC, 0.0)
        if pos_ms >= ms - 1.0:  # marge 1 ms
            return True
        ok, _ = read_frame(cap)
        if not ok:
            break
    return False


# -----------------
# Façade orientée objet
# -----------------

@dataclass
class VideoReader:
    source: str | int
    cap: Optional[cv2.VideoCapture] = None
    fps: float = 0.0
    frame_count: float = 0.0
    width: int = 0
    height: int = 0

    def open(self) -> None:
        self.cap = open_capture(self.source)
        props = get_props(self.cap)
        self.fps = props["fps"]
        self.frame_count = props["frame_count"]
        self.width = props["width"]
        self.height = props["height"]

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        assert self.cap is not None, "VideoReader non ouvert"
        return read_frame(self.cap)

    def seek_msec(self, ms: float) -> bool:
        assert self.cap is not None, "VideoReader non ouvert"
        return seek_msec(self.cap, ms)

    def set_pos_frame(self, idx: int) -> bool:
        assert self.cap is not None, "VideoReader non ouvert"
        return _set_pos_frames(self.cap, idx)

    def get_pos_msec(self) -> float:
        assert self.cap is not None, "VideoReader non ouvert"
        return _safe_get(self.cap, cv2.CAP_PROP_POS_MSEC, 0.0)

    def get_pos_frame(self) -> int:
        assert self.cap is not None, "VideoReader non ouvert"
        return _get_pos_frames(self.cap)

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
