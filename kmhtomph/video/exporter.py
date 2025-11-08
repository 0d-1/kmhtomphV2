"""
Export vidéo : lit depuis un VideoReader et écrit un fichier avec overlay optionnel.

Expose :
- ExportParams : fps/size/codec configurables (avec détection auto)
- open_writer_with_fallback(path, size, fps, prefer_mp4=True) -> cv2.VideoWriter
- export_video(reader, out_path, text_supplier, draw_overlay=None, on_progress=None, params=ExportParams())

Contrats :
- `reader` : instance ouverte de VideoReader (video.io)
- `text_supplier(idx:int, frame_bgr:np.ndarray) -> Optional[str]`
    Retourne la chaîne à incruster (ex: "54 mph") pour cette frame, ou None pour rien.
- `draw_overlay(frame_bgr, text:str)` :
    Modifie la frame in-place pour y poser le texte (peut utiliser video.overlay).
    Si None, l’export écrit la frame originale (sans overlay).
- `on_progress(done_frames:int, total_frames:int|None)` :
    Call-back optionnel pour suivre l’avancement.

Notes :
- Cette unité ne dépend pas de Qt.
- Le writer est ouvert en essayant plusieurs FOURCC compatibles courants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Iterable
import os

import cv2
import numpy as np

from .io import VideoReader, get_props


# -------------------------
# Paramètres d'export
# -------------------------

@dataclass(frozen=True)
class ExportParams:
    fps: Optional[float] = None         # si None, reprendre celui de la source
    size: Optional[Tuple[int, int]] = None  # (width, height). Si None, reprendre celui de la source
    prefer_mp4: bool = True             # si True, on privilégie un codec compatible .mp4


# -------------------------
# Ouverture du writer
# -------------------------

def _try_open_writer(path: str, fourcc: str, fps: float, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(path, code, fps, size)
    if writer is not None and writer.isOpened():
        return writer
    try:
        if writer:
            writer.release()
    except Exception:
        pass
    return None


def open_writer_with_fallback(
    path: str,
    size: Tuple[int, int],
    fps: float,
    prefer_mp4: bool = True,
) -> cv2.VideoWriter:
    """
    Ouvre un VideoWriter en essayant une liste de FOURCC.
    - Pour .mp4 : 'mp4v', 'avc1', 'H264' (selon OS/build)
    - Pour .avi  : 'XVID', 'MJPG'

    Retourne un writer ouvert ou lève RuntimeError.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    is_mp4_ext = os.path.splitext(path)[1].lower() in (".mp4", ".m4v", ".mov")
    mp4_first = prefer_mp4 or is_mp4_ext

    mp4_candidates = ["mp4v", "avc1", "H264"]
    avi_candidates = ["XVID", "MJPG"]

    order = mp4_candidates + avi_candidates if mp4_first else avi_candidates + mp4_candidates

    # Toujours ajouter un dernier essai sans fourcc explicite (backend default)
    for fourcc in order:
        w = _try_open_writer(path, fourcc, fps, size)
        if w is not None:
            return w

    # Essai avec fourcc=0 (laisser OpenCV choisir)
    writer = cv2.VideoWriter(path, 0, fps, size)
    if writer is not None and writer.isOpened():
        return writer

    raise RuntimeError(f"Impossible d’ouvrir le writer vidéo pour {path!r} ({size[0]}x{size[1]} @ {fps} fps)")


# -------------------------
# Export
# -------------------------

def export_video(
    reader: VideoReader,
    out_path: str,
    text_supplier: Callable[[int, np.ndarray], Optional[str]],
    draw_overlay: Optional[Callable[[np.ndarray, str], None]] = None,
    on_progress: Optional[Callable[[int, Optional[int]], None]] = None,
    params: ExportParams = ExportParams(),
) -> None:
    """
    Parcourt les frames de `reader` et écrit un fichier avec overlay optionnel.

    - Les frames sont lues séquentiellement depuis la position courante.
    - Si `draw_overlay` est fourni et `text_supplier` retourne une chaîne, elle est incrustée.
    - Si `on_progress` est fourni, il est appelé régulièrement.

    Lève une exception en cas d’erreur d’ouverture du writer ou de lecture.
    """
    assert reader.is_opened(), "VideoReader doit être ouvert avant l’export"

    props = get_props(reader.cap)  # type: ignore[arg-type]
    fps = float(params.fps) if params.fps and params.fps > 0 else float(props["fps"] or 25.0)
    width = int(props["width"])
    height = int(props["height"])
    if params.size is not None:
        width, height = params.size

    writer = open_writer_with_fallback(out_path, (width, height), fps, prefer_mp4=params.prefer_mp4)

    total_frames = int(props["frame_count"]) if props["frame_count"] > 0 else None
    idx = 0

    try:
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                break

            # Adapter la taille si nécessaire
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            text = text_supplier(idx, frame)
            if draw_overlay is not None and text:
                draw_overlay(frame, text)

            writer.write(frame)

            idx += 1
            if on_progress is not None and (idx % 10 == 0 or (total_frames and idx == total_frames)):
                on_progress(idx, total_frames)
    finally:
        try:
            writer.release()
        except Exception:
            pass
