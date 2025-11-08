"""Backend de lecture vidéo inspiré du projet video-subtitle-extractor.

Ce module expose :

- :class:`SubtitleExtractorVideoReader` : utilise ``ffprobe`` et ``ffmpeg`` pour
  obtenir les métadonnées vidéo et décoder les frames en BGR via un pipe.

L’objectif est d’offrir une alternative à :mod:`cv2.VideoCapture` plus robuste
pour les fichiers (certaines vidéos MP4 posaient problème avec OpenCV). Le
projet `video-subtitle-extractor <https://github.com/YaoFANGUK/video-subtitle-extractor>`_
décode les vidéos en s’appuyant sur ffmpeg ; cette implémentation s’inspire de
son approche en démarrant un processus ``ffmpeg`` et en lisant les pixels bruts.

Le lecteur fournit les attributs ``fps``, ``frame_count``, ``width`` et
``height`` une fois ouvert. La méthode :meth:`seek_msec` redémarre le processus
``ffmpeg`` avec un décalage temporel (`-ss`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import IO, Optional, Tuple

import json
import math
import subprocess
import threading
import shutil

import numpy as np


def _ensure_executable_available(name: str) -> str:
    """Retourne le chemin vers l’exécutable ``name`` ou lève ``RuntimeError``."""

    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"L’exécutable {name!r} est requis pour la lecture vidéo (backend ffmpeg)."
        )
    return path


def _parse_fraction(value: str) -> float:
    try:
        frac = Fraction(value)
        if frac.denominator == 0:
            return 0.0
        return float(frac)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _spawn_reader_process(
    ffmpeg_path: str,
    source: str,
    start_sec: float,
) -> subprocess.Popen[bytes]:
    """Démarre un processus ``ffmpeg`` qui émet des frames BGR sur stdout."""

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{max(0.0, start_sec):.6f}",
        "-i",
        source,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-an",
        "-sn",
        "-dn",
        "-",
    ]

    # ``stderr`` est redirigé vers un thread pour éviter les blocages (ffmpeg
    # écrit des logs sur stderr).
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.stdout is not None  # pour mypy
    assert proc.stderr is not None

    def _drain_stderr(pipe: IO[bytes]) -> None:
        try:
            for _ in iter(lambda: pipe.readline(), b""):
                pass
        finally:
            pipe.close()

    drain_thread = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
    drain_thread.start()

    return proc


@dataclass
class SubtitleExtractorVideoReader:
    """Lecture vidéo via ffmpeg, inspirée de video-subtitle-extractor."""

    source: str
    ffmpeg_path: str = field(default="ffmpeg")
    ffprobe_path: str = field(default="ffprobe")
    fps: float = field(init=False, default=0.0)
    frame_count: float = field(init=False, default=0.0)
    width: int = field(init=False, default=0)
    height: int = field(init=False, default=0)

    _process: Optional[subprocess.Popen[bytes]] = field(init=False, default=None)
    _frame_size: int = field(init=False, default=0)
    _current_frame_index: int = field(init=False, default=0)
    _start_frame_index: int = field(init=False, default=0)
    _start_time_sec: float = field(init=False, default=0.0)

    def open(self) -> None:
        ffmpeg = _ensure_executable_available(self.ffmpeg_path)
        ffprobe = _ensure_executable_available(self.ffprobe_path)

        self._load_metadata(ffprobe)
        start_sec = self._start_frame_index / self.fps if self.fps > 0 else 0.0
        self._start_process(ffmpeg, start_sec=start_sec)

    # -------------------
    # Métadonnées (ffprobe)
    # -------------------

    def _load_metadata(self, ffprobe: str) -> None:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of",
            "json",
            self.source,
        ]

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffprobe a échoué pour {self.source!r}: {exc.output.decode(errors='ignore')}") from exc

        data = json.loads(output.decode("utf-8"))
        streams = data.get("streams") or []
        if not streams:
            raise RuntimeError(f"Aucun flux vidéo détecté dans {self.source!r}")

        stream = streams[0]
        self.width = int(stream.get("width") or 0)
        self.height = int(stream.get("height") or 0)
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError("Dimensions vidéo invalides (width/height)")

        r_frame_rate = stream.get("r_frame_rate")
        fps = _parse_fraction(r_frame_rate) if r_frame_rate else 0.0
        if fps <= 0:
            duration = _parse_float(stream.get("duration"))
            # Fallback : approx fps via durée et nb_frames si disponible
            nb_frames_val = stream.get("nb_frames")
            nb_frames = float(nb_frames_val) if nb_frames_val not in (None, "0", 0) else 0.0
            if duration > 0 and nb_frames > 0:
                fps = nb_frames / duration
        self.fps = fps if fps > 0 else 0.0

        nb_frames_raw = stream.get("nb_frames")
        nb_frames = _parse_float(nb_frames_raw) if nb_frames_raw not in (None, "0", 0) else 0.0
        if nb_frames <= 0 and self.fps > 0:
            duration = _parse_float(stream.get("duration"))
            if duration > 0:
                nb_frames = math.floor(duration * self.fps)
        self.frame_count = float(nb_frames) if nb_frames > 0 else 0.0

        self._frame_size = self.width * self.height * 3
        self._current_frame_index = 0
        self._start_frame_index = 0
        self._start_time_sec = 0.0

    # ----------------
    # Processus ffmpeg
    # ----------------

    def _start_process(self, ffmpeg_path: str, start_sec: float = 0.0) -> None:
        self._close_process()
        self._current_frame_index = 0
        self._start_time_sec = start_sec
        self._process = _spawn_reader_process(ffmpeg_path, self.source, start_sec)

    def _close_process(self) -> None:
        if self._process is not None:
            try:
                if self._process.stdout:
                    self._process.stdout.close()
            finally:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    # --------
    # Lecture
    # --------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._process is None or self._process.stdout is None:
            return False, None

        buffer = self._process.stdout.read(self._frame_size)
        if len(buffer) < self._frame_size:
            self._close_process()
            return False, None

        frame = np.frombuffer(buffer, dtype=np.uint8).reshape((self.height, self.width, 3))
        self._current_frame_index += 1
        return True, frame

    # ------
    # État
    # ------

    def is_opened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def seek_msec(self, ms: float) -> bool:
        target_ms = max(0.0, float(ms))
        start_sec = target_ms / 1000.0
        if self.fps > 0:
            self._start_frame_index = int(round(start_sec * self.fps))
        else:
            self._start_frame_index = 0
        try:
            ffmpeg = _ensure_executable_available(self.ffmpeg_path)
        except RuntimeError:
            return False
        self._start_process(ffmpeg, start_sec=start_sec)
        return True

    def set_pos_frame(self, idx: int) -> bool:
        if idx < 0:
            idx = 0
        target_sec = idx / self.fps if self.fps > 0 else 0.0
        self._start_frame_index = idx
        try:
            ffmpeg = _ensure_executable_available(self.ffmpeg_path)
        except RuntimeError:
            return False
        self._start_process(ffmpeg, start_sec=target_sec)
        return True

    def get_pos_frame(self) -> int:
        return self._start_frame_index + self._current_frame_index

    def get_pos_msec(self) -> float:
        if self.fps <= 0:
            return self._start_time_sec * 1000.0
        return (self.get_pos_frame() / self.fps) * 1000.0

    def release(self) -> None:
        self._close_process()

    # -------------
    # Contexte mgr
    # -------------

    def __enter__(self) -> "SubtitleExtractorVideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


__all__ = ["SubtitleExtractorVideoReader"]

