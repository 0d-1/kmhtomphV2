"""
Fenêtre principale : ouverture vidéo, lecture, OCR temps réel, export,
barre de progression + raccourcis, extraction ROI par homographie exacte.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Optional, List

import copy
import math

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer, QSettings
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QAction, QFileDialog, QMessageBox, QApplication,
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QProgressDialog, QSlider, QGroupBox, QFormLayout, QDoubleSpinBox, QSizePolicy,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtGui import QImage, QPixmap, QColor

from pytesseract import TesseractNotFoundError

from ..constants import (
    KMH_TO_MPH,
    DEFAULT_SHOW_DEBUG_THUMB,
    DEFAULT_DEBUG_THUMB_SIZE,
    AntiJitterConfig,
)
from ..ocr import OCRPipeline
from ..ocr.tesseract import auto_locate_tesseract
from ..video.io import VideoReader
from ..video.exporter import export_video, ExportParams
from ..video.overlay import OverlayStyle, draw_speed_overlay, format_speed_text
from .canvas import VideoCanvas
from .settings import SettingsDialog
from .overlaystyle import OverlayStyleDialog


def _extract_roi_from_corners(
    frame_bgr: np.ndarray,
    corners_xy: np.ndarray,  # shape (4,2) float32 : TL,TR,BR,BL en coordonnées image
    w: int,
    h: int,
) -> np.ndarray:
    """Extrait exactement la zone définie par `corners_xy` vers un patch (h, w)."""
    w, h = int(w), int(h)
    if w <= 1 or h <= 1:
        return np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)

    src_pts = np.asarray(corners_xy, dtype=np.float32)
    dst_pts = np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]],
        dtype=np.float32,
    )
    Hmat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    patch = cv2.warpPerspective(
        frame_bgr, Hmat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch


@dataclass
class SpeedTimelineEntry:
    time_sec: float
    kmh: Optional[float]
    manual: bool = False

    def mph(self) -> Optional[float]:
        if self.kmh is None:
            return None
        return float(self.kmh) * KMH_TO_MPH


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("kmh→mph OCR")
        self.setFocusPolicy(Qt.StrongFocus)  # capter les raccourcis

        # --- état ---
        self.reader: Optional[VideoReader] = None
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self.ocr = OCRPipeline()
        self.ocr_mode = "auto"  # "auto" | "sevenseg" | "tesseract"

        self.overlay_style = OverlayStyle()
        self._last_overlay_text: Optional[str] = None
        self._loading_overlay_settings = False
        self.show_debug_thumb = DEFAULT_SHOW_DEBUG_THUMB
        self.debug_thumb_size = DEFAULT_DEBUG_THUMB_SIZE
        self._last_debug_bgr: Optional[np.ndarray] = None
        self._loading_smoothing_settings = False

        self._settings = QSettings("kmhtomph", "kmh_to_mph")
        self._tesseract_error_shown = False

        self._loading_range_settings = False
        self.transcription_range_enabled = bool(
            self._settings.value("range/enabled", False, type=bool)
        )
        self.transcription_start_sec = self._coerce_float(
            self._settings.value("range/start_sec"),
            0.0,
        )
        self.transcription_end_sec = self._coerce_float(
            self._settings.value("range/end_sec"),
            0.0,
        )

        self._tesseract_path: Optional[str] = None
        stored_path = self._settings.value("tesseract/path", type=str)
        if stored_path:
            self._tesseract_path = stored_path
        self._apply_tesseract_path(initial=True)

        self._status_tasks: dict[str, dict] = {}

        # --- timeline ---
        self.timeline_step_seconds: float = 0.5
        self.speed_timeline: List[SpeedTimelineEntry] = []
        self._timeline_context: Optional[dict] = None
        self._timeline_total_duration: float = 0.0
        self._timeline_status_reason: Optional[str] = "Aucune analyse effectuée."
        self._timeline_table_loading: bool = False

        # --- UI ---
        self.canvas = VideoCanvas(self)
        self.canvas.on_roi_changed.connect(self._on_canvas_roi_changed)

        self._loading_overlay_settings = True
        self.overlay_style = self._load_overlay_style(self.overlay_style)
        self._set_overlay_text(None, allow_placeholder=True)
        self._load_overlay_rect()
        self._loading_overlay_settings = False

        self.lbl_kmh = QLabel("-- km/h", self)
        self.lbl_mph = QLabel("-- mph", self)
        for l in (self.lbl_kmh, self.lbl_mph):
            l.setStyleSheet("font-size: 18px;")

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["auto", "sevenseg", "tesseract"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        self.btn_open = QPushButton("Ouvrir…", self)
        self.btn_play = QPushButton("Lecture", self)
        self.btn_export = QPushButton("Exporter…", self)
        self.btn_overlay = QPushButton("Tracer zone de sortie", self)
        self.btn_overlay.setCheckable(True)
        self.btn_overlay_style = QPushButton("Style du texte…", self)

        self.chk_debug = QCheckBox("Vignette debug", self)
        self.chk_debug.setChecked(self.show_debug_thumb)
        self.chk_debug.stateChanged.connect(self._on_toggle_debug)

        self.spin_dbg = QSpinBox(self)
        self.spin_dbg.setRange(64, 512)
        self.spin_dbg.setValue(self.debug_thumb_size)
        self.spin_dbg.setSuffix(" px")
        self.spin_dbg.valueChanged.connect(self._on_thumb_size)

        # --- Barre de progression + temps ---
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setEnabled(False)
        self.slider.setRange(0, 0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(25)

        self.lbl_time = QLabel("00:00 / 00:00", self)

        self._seeking = False
        self._was_playing_before_seek = False

        # layout
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_open)
        top_bar.addWidget(self.btn_play)
        top_bar.addWidget(self.btn_export)
        top_bar.addWidget(self.btn_overlay)
        top_bar.addWidget(self.btn_overlay_style)
        top_bar.addSpacing(20)
        top_bar.addWidget(QLabel("Mode OCR:", self))
        top_bar.addWidget(self.mode_combo)
        top_bar.addStretch(1)
        top_bar.addWidget(self.lbl_kmh)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.lbl_mph)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.chk_debug)
        top_bar.addWidget(self.spin_dbg)

        prog_bar = QHBoxLayout()
        prog_bar.addWidget(self.slider, 1)
        prog_bar.addSpacing(8)
        prog_bar.addWidget(self.lbl_time)

        self.side_panel = self._create_side_panel()
        self._configure_range_controls(0.0)
        self._update_debug_preview_geometry()
        self._load_smoothing_settings()
        self._refresh_debug_thumbnail()

        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.addLayout(top_bar)

        content = QHBoxLayout()
        content.addWidget(self.canvas, 1)
        content.addWidget(self.side_panel)
        lay.addLayout(content, 1)

        lay.addLayout(prog_bar)
        self.setCentralWidget(central)

        # actions
        self.btn_open.clicked.connect(self._on_open)
        self.btn_play.clicked.connect(self._on_toggle_play)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_overlay.toggled.connect(self._on_toggle_overlay_mode)
        self.btn_overlay_style.clicked.connect(self._on_edit_overlay_style)

        # slider signals
        self.slider.sliderPressed.connect(self._on_seek_start)
        self.slider.sliderReleased.connect(self._on_seek_end)
        self.slider.valueChanged.connect(self._on_seek_change)

        self.canvas.on_overlay_changed.connect(self._on_overlay_rect_changed)

        # menu
        self._create_menu()

        self._update_overlay_mode_button()

        self.resize(1000, 740)

        self._refresh_timeline_table()

    # ------------- Menu -------------

    def _create_side_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setMinimumWidth(260)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(12)

        smoothing_group = QGroupBox("Lissage", panel)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        smoothing_group.setLayout(form)

        self.spin_smooth_past = QSpinBox(smoothing_group)
        self.spin_smooth_past.setRange(0, 31)
        self.spin_smooth_past.setValue(int(self.ocr.anti_jitter.median_past_frames))
        self.spin_smooth_past.setToolTip("Nombre d'images en amont utilisées pour la médiane centrée.")
        form.addRow("Passé (frames)", self.spin_smooth_past)

        self.spin_smooth_future = QSpinBox(smoothing_group)
        self.spin_smooth_future.setRange(0, 31)
        self.spin_smooth_future.setValue(int(self.ocr.anti_jitter.median_future_frames))
        self.spin_smooth_future.setToolTip("Nombre d'images en aval utilisées pour la médiane centrée.")
        form.addRow("Futur (frames)", self.spin_smooth_future)

        self.spin_smooth_delta = QDoubleSpinBox(smoothing_group)
        self.spin_smooth_delta.setDecimals(1)
        self.spin_smooth_delta.setSingleStep(0.1)
        self.spin_smooth_delta.setRange(0.5, 25.0)
        self.spin_smooth_delta.setValue(float(self.ocr.anti_jitter.max_delta_kmh))
        self.spin_smooth_delta.setSuffix(" km/h")
        self.spin_smooth_delta.setToolTip("Variation maximale autorisée entre deux mesures successives.")
        form.addRow("Delta max.", self.spin_smooth_delta)

        self.spin_smooth_conf = QDoubleSpinBox(smoothing_group)
        self.spin_smooth_conf.setDecimals(2)
        self.spin_smooth_conf.setSingleStep(0.05)
        self.spin_smooth_conf.setRange(0.0, 1.0)
        self.spin_smooth_conf.setValue(float(self.ocr.anti_jitter.min_confidence))
        self.spin_smooth_conf.setToolTip("Score minimal requis pour accepter une valeur.")
        form.addRow("Confiance min.", self.spin_smooth_conf)

        self.spin_smooth_hold = QSpinBox(smoothing_group)
        self.spin_smooth_hold.setRange(0, 120)
        self.spin_smooth_hold.setValue(int(self.ocr.anti_jitter.hold_max_gap_frames))
        self.spin_smooth_hold.setToolTip("Nombre de frames durant lesquelles conserver la dernière valeur fiable.")
        form.addRow("Maintien (frames)", self.spin_smooth_hold)

        self.spin_smooth_past.valueChanged.connect(self._on_smoothing_params_changed)
        self.spin_smooth_future.valueChanged.connect(self._on_smoothing_params_changed)
        self.spin_smooth_delta.valueChanged.connect(self._on_smoothing_params_changed)
        self.spin_smooth_conf.valueChanged.connect(self._on_smoothing_params_changed)
        self.spin_smooth_hold.valueChanged.connect(self._on_smoothing_params_changed)

        layout.addWidget(smoothing_group)

        range_group = QGroupBox("Plage de transcription", panel)
        range_layout = QVBoxLayout(range_group)
        range_layout.setContentsMargins(9, 9, 9, 9)
        range_layout.setSpacing(6)

        self.chk_range_enable = QCheckBox("Limiter la transcription à une plage", range_group)
        self.chk_range_enable.stateChanged.connect(self._on_range_toggle)
        range_layout.addWidget(self.chk_range_enable)

        range_form = QFormLayout()
        range_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.spin_range_start = QDoubleSpinBox(range_group)
        self.spin_range_start.setDecimals(2)
        self.spin_range_start.setSingleStep(0.1)
        self.spin_range_start.setRange(0.0, 0.0)
        self.spin_range_start.setSuffix(" s")
        self.spin_range_start.valueChanged.connect(self._on_range_start_changed)

        self.spin_range_end = QDoubleSpinBox(range_group)
        self.spin_range_end.setDecimals(2)
        self.spin_range_end.setSingleStep(0.1)
        self.spin_range_end.setRange(0.0, 0.0)
        self.spin_range_end.setSuffix(" s")
        self.spin_range_end.valueChanged.connect(self._on_range_end_changed)

        range_form.addRow("Début (s)", self.spin_range_start)
        range_form.addRow("Fin (s)", self.spin_range_end)
        range_layout.addLayout(range_form)

        layout.addWidget(range_group)

        timeline_group = self._create_timeline_group(panel)
        layout.addWidget(timeline_group)

        status_group = QGroupBox("Statut", panel)
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(9, 9, 9, 9)
        status_layout.setSpacing(6)

        self.status_empty_label = QLabel("Aucune tâche en cours", status_group)
        self.status_empty_label.setStyleSheet("color: #888; font-style: italic;")
        status_layout.addWidget(self.status_empty_label)

        self._status_layout = status_layout
        layout.addWidget(status_group)

        layout.addStretch(1)

        self.debug_label_title = QLabel("Vignette debug", panel)
        self.debug_label_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.debug_label_title)

        self.debug_preview = QLabel(panel)
        self.debug_preview.setAlignment(Qt.AlignCenter)
        self.debug_preview.setStyleSheet("background-color: #111; border: 1px solid #333; color: #888;")
        self.debug_preview.setWordWrap(True)
        self.debug_preview.setText("Aucune vignette")
        layout.addWidget(self.debug_preview, 0, Qt.AlignBottom)

        return panel

    def _create_timeline_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Analyse des vitesses", parent)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(9, 9, 9, 9)
        group_layout.setSpacing(6)

        self.btn_analyze_timeline = QPushButton("Analyser les vitesses", group)
        self.btn_analyze_timeline.clicked.connect(self._on_analyze_timeline)
        group_layout.addWidget(self.btn_analyze_timeline)

        self.timeline_status_label = QLabel(self._timeline_status_reason or "", group)
        self.timeline_status_label.setWordWrap(True)
        self.timeline_status_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self.timeline_status_label)

        self.timeline_table = QTableWidget(group)
        self.timeline_table.setColumnCount(3)
        self.timeline_table.setHorizontalHeaderLabels(["Temps (s)", "km/h", "mph"])
        header = self.timeline_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.timeline_table.verticalHeader().setVisible(False)
        self.timeline_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.timeline_table.setSelectionMode(QTableWidget.SingleSelection)
        self.timeline_table.setAlternatingRowColors(True)
        self.timeline_table.setEditTriggers(
            QTableWidget.DoubleClicked | QTableWidget.SelectedClicked | QTableWidget.EditKeyPressed
        )
        self.timeline_table.itemChanged.connect(self._on_timeline_item_changed)
        group_layout.addWidget(self.timeline_table, 1)

        return group

    def _refresh_timeline_table(self) -> None:
        if not hasattr(self, "timeline_table"):
            return
        self._timeline_table_loading = True
        try:
            self.timeline_table.clearContents()
            row_count = len(self.speed_timeline)
            self.timeline_table.setRowCount(row_count)
            if row_count > 0:
                default_color = self.timeline_table.palette().color(self.timeline_table.backgroundRole())
                for row, entry in enumerate(self.speed_timeline):
                    time_item = QTableWidgetItem(f"{entry.time_sec:.2f}")
                    time_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    time_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.timeline_table.setItem(row, 0, time_item)

                    kmh_text = "" if entry.kmh is None else f"{entry.kmh:.1f}"
                    kmh_item = QTableWidgetItem(kmh_text)
                    kmh_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.timeline_table.setItem(row, 1, kmh_item)

                    mph_val = entry.mph()
                    mph_text = "" if mph_val is None else f"{mph_val:.1f}"
                    mph_item = QTableWidgetItem(mph_text)
                    mph_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    mph_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.timeline_table.setItem(row, 2, mph_item)

                    self._apply_timeline_row_style(row, entry.manual, default_color)
        finally:
            self._timeline_table_loading = False

        self._refresh_timeline_status_label()

    def _refresh_timeline_status_label(self) -> None:
        if not hasattr(self, "timeline_status_label"):
            return
        if not self.speed_timeline:
            message = self._timeline_status_reason or "Aucune analyse effectuée."
            self.timeline_status_label.setStyleSheet("color: #888; font-size: 11px;")
            self.timeline_status_label.setText(message)
            return

        manual_count = sum(1 for entry in self.speed_timeline if entry.manual)
        base = f"{len(self.speed_timeline)} valeurs analysées (pas {self.timeline_step_seconds:.1f} s)."
        if self._timeline_total_duration > 0.0:
            base += f" Durée couverte : {self._format_hms(self._timeline_total_duration)}."
        if manual_count:
            base += f" {manual_count} valeur(s) modifiée(s) manuellement."
        self.timeline_status_label.setStyleSheet("color: #555; font-size: 11px;")
        self.timeline_status_label.setText(base)

    def _apply_timeline_row_style(
        self,
        row: int,
        manual: bool,
        default_color: Optional[QColor] = None,
    ) -> None:
        if not hasattr(self, "timeline_table"):
            return
        if default_color is None:
            default_color = self.timeline_table.palette().color(self.timeline_table.backgroundRole())
        color = QColor("#FFF4D6") if manual else default_color
        for col in range(self.timeline_table.columnCount()):
            item = self.timeline_table.item(row, col)
            if item is not None:
                item.setBackground(color)

    def _update_timeline_row(self, row: int) -> None:
        if not hasattr(self, "timeline_table"):
            return
        if row < 0 or row >= len(self.speed_timeline):
            return
        entry = self.speed_timeline[row]
        default_color = self.timeline_table.palette().color(self.timeline_table.backgroundRole())
        self._timeline_table_loading = True
        try:
            kmh_item = self.timeline_table.item(row, 1)
            if kmh_item is None:
                kmh_item = QTableWidgetItem()
                self.timeline_table.setItem(row, 1, kmh_item)
            kmh_item.setText("" if entry.kmh is None else f"{entry.kmh:.1f}")
            kmh_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            mph_item = self.timeline_table.item(row, 2)
            if mph_item is None:
                mph_item = QTableWidgetItem()
                mph_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.timeline_table.setItem(row, 2, mph_item)
            mph_val = entry.mph()
            mph_item.setText("" if mph_val is None else f"{mph_val:.1f}")
            mph_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        finally:
            self._timeline_table_loading = False

        self._apply_timeline_row_style(row, entry.manual, default_color)
        self._refresh_timeline_status_label()

    def _has_manual_timeline_edits(self) -> bool:
        return any(entry.manual for entry in self.speed_timeline)

    def _on_timeline_item_changed(self, item: QTableWidgetItem) -> None:
        if self._timeline_table_loading:
            return
        if item.column() != 1:
            return
        row = item.row()
        if row < 0 or row >= len(self.speed_timeline):
            return

        text = item.text().strip()
        if text:
            text = text.replace(",", ".")
        entry = self.speed_timeline[row]
        if not text:
            entry.kmh = None
            entry.manual = True
            self._update_timeline_row(row)
            return
        try:
            value = float(text)
        except ValueError:
            self._timeline_table_loading = True
            try:
                item.setText("" if entry.kmh is None else f"{entry.kmh:.1f}")
            finally:
                self._timeline_table_loading = False
            return

        entry.kmh = value
        entry.manual = True
        self._timeline_status_reason = None
        self._update_timeline_row(row)

    def _on_analyze_timeline(self) -> None:
        if not self.reader or not self.reader.is_opened():
            QMessageBox.information(self, "Analyse", "Ouvrez d’abord une vidéo avant de lancer l’analyse.")
            return
        if self._has_manual_timeline_edits():
            answer = QMessageBox.question(
                self,
                "Analyse",
                "L’analyse va écraser les modifications manuelles existantes. Continuer ?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._perform_speed_analysis()

    def _perform_speed_analysis(self) -> bool:
        if not self.reader or not self.reader.is_opened():
            QMessageBox.information(self, "Analyse", "Ouvrez d’abord une vidéo avant de lancer l’analyse.")
            return False

        src = self.reader.source
        if not src:
            QMessageBox.warning(self, "Analyse", "Impossible de déterminer la source vidéo.")
            return False

        cx, cy, w, h, _ = self.canvas.get_roi()
        if w <= 1 or h <= 1:
            QMessageBox.warning(self, "Analyse", "La zone de lecture (ROI) est trop petite.")
            return False
        base_corners = self.canvas.get_roi_corners().copy()

        precalc_ocr = copy.deepcopy(self.ocr)
        precalc_ocr.reset()

        analysis_reader = VideoReader(src)
        try:
            analysis_reader.open()
        except Exception as e:
            QMessageBox.critical(self, "Analyse", f"Impossible de préparer la vidéo : {e}")
            return False

        fps_source = float(analysis_reader.fps or self._fps())
        total_precalc = int(analysis_reader.frame_count) if analysis_reader.frame_count > 0 else None
        progress = QProgressDialog("Analyse des vitesses…", "Annuler", 0, total_precalc or 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        analysis_task_id = "timeline-analysis"
        self._status_begin_task(analysis_task_id, "Analyse des vitesses", total_precalc)

        mph_sequence: List[Optional[float]] = []
        error: Optional[Exception] = None
        canceled = False
        idx = 0
        try:
            while True:
                ok_calc, frame_calc = analysis_reader.read()
                if not ok_calc or frame_calc is None:
                    break
                roi_calc = _extract_roi_from_corners(frame_calc, base_corners, w, h)
                try:
                    kmh_val, _, _, _ = precalc_ocr.read_kmh(roi_calc, mode=self.ocr_mode)
                except (TesseractNotFoundError, FileNotFoundError) as e:
                    self._handle_tesseract_error(e)
                    error = e
                    break
                mph_sequence.append(kmh_val * KMH_TO_MPH if kmh_val is not None else None)

                if total_precalc:
                    progress.setMaximum(total_precalc)
                    progress.setValue(idx)
                    self._status_update_task(analysis_task_id, idx, total_precalc)
                else:
                    progress.setValue(idx % 100)
                    self._status_update_task(analysis_task_id, idx)
                QApplication.processEvents()
                if progress.wasCanceled():
                    canceled = True
                    break
                idx += 1

            if not canceled and error is None and total_precalc:
                progress.setMaximum(total_precalc)
                progress.setValue(total_precalc)
                self._status_update_task(analysis_task_id, total_precalc, total_precalc)
        finally:
            progress.close()
            try:
                analysis_reader.release()
            except Exception:
                pass
            if canceled:
                self._status_finish_task(analysis_task_id, message="Analyse annulée", success=False)
            elif error is not None:
                self._status_finish_task(
                    analysis_task_id,
                    message=f"Erreur : {error}",
                    success=False,
                )
            else:
                self._status_finish_task(analysis_task_id, message="Analyse terminée")

        if canceled:
            self._timeline_status_reason = "Analyse annulée."
            self._refresh_timeline_table()
            return False
        if error is not None:
            if not isinstance(error, (TesseractNotFoundError, FileNotFoundError)):
                QMessageBox.warning(self, "Analyse", f"Échec de l’analyse : {error}")
            return False

        if not mph_sequence:
            self.speed_timeline = []
            self._timeline_total_duration = 0.0
            self._timeline_context = self._current_timeline_context()
            self._timeline_status_reason = "Aucune donnée détectée."
            self._refresh_timeline_table()
            return True

        timeline_entries = self._build_timeline_from_sequence(mph_sequence, fps_source)
        self.speed_timeline = timeline_entries
        self._timeline_context = self._current_timeline_context()
        self._timeline_total_duration = (len(mph_sequence) / fps_source) if fps_source > 0 else 0.0
        self._timeline_status_reason = None
        self._refresh_timeline_table()
        return True

    def _build_timeline_from_sequence(
        self,
        mph_sequence: List[Optional[float]],
        fps: float,
    ) -> List[SpeedTimelineEntry]:
        fps = float(fps) if fps and fps > 0 else float(self._fps())
        if fps <= 0:
            fps = 25.0
        total_frames = len(mph_sequence)
        if total_frames <= 0:
            return []
        total_seconds = total_frames / fps
        steps = int(math.ceil(total_seconds / self.timeline_step_seconds))
        entries: List[SpeedTimelineEntry] = []
        for step_idx in range(steps):
            start_time = step_idx * self.timeline_step_seconds
            end_time = min(total_seconds, start_time + self.timeline_step_seconds)
            start_frame = int(math.floor(start_time * fps + 1e-6))
            end_frame = int(math.floor(end_time * fps + 1e-6))
            if end_frame <= start_frame:
                end_frame = min(start_frame + 1, total_frames)
            segment = mph_sequence[start_frame:end_frame]
            valid = [float(v) for v in segment if v is not None and np.isfinite(v)]
            if valid:
                mph_val = float(np.median(valid))
                kmh_val: Optional[float] = mph_val / KMH_TO_MPH
            else:
                kmh_val = None
            entries.append(
                SpeedTimelineEntry(
                    time_sec=float(round(start_time, 3)),
                    kmh=kmh_val,
                    manual=False,
                )
            )
        return entries

    def _invalidate_speed_timeline(self, reason: Optional[str] = None) -> None:
        had_timeline = bool(self.speed_timeline)
        self.speed_timeline = []
        self._timeline_context = None
        self._timeline_total_duration = 0.0
        if reason and had_timeline:
            self._timeline_status_reason = f"Recalcul requis ({reason})."
        elif not had_timeline and self._timeline_status_reason is None:
            self._timeline_status_reason = "Aucune analyse effectuée."
        elif not reason:
            self._timeline_status_reason = "Aucune analyse effectuée."
        self._refresh_timeline_table()

    def _on_canvas_roi_changed(self) -> None:
        if self.speed_timeline:
            self._invalidate_speed_timeline("ROI modifié")

    def _current_timeline_context(self) -> Optional[dict]:
        if not self.reader or not self.reader.is_opened():
            return None
        cx, cy, w, h, ang = self.canvas.get_roi()
        corners = tuple(round(float(v), 3) for v in self.canvas.get_roi_corners().flatten())
        cfg = self.ocr.anti_jitter
        return {
            "source": self.reader.source,
            "mode": self.ocr_mode,
            "roi": (int(cx), int(cy), int(w), int(h), round(float(ang), 3), corners),
            "smoothing": (
                int(cfg.median_past_frames),
                int(cfg.median_future_frames),
                float(cfg.max_delta_kmh),
                float(cfg.min_confidence),
                int(cfg.hold_max_gap_frames),
            ),
        }

    def _timeline_mph_for_frame(self, frame_idx: int, fps: float) -> Optional[float]:
        if not self.speed_timeline:
            return None
        if fps <= 0:
            fps = self._fps()
        fps = max(float(fps), 1e-6)
        time_sec = float(frame_idx) / fps
        if self.timeline_step_seconds <= 0:
            row = 0
        else:
            row = int(time_sec / self.timeline_step_seconds)
        row = max(0, min(row, len(self.speed_timeline) - 1))
        return self.speed_timeline[row].mph()

    def _ensure_speed_timeline_ready_for_export(self) -> bool:
        if not self.reader or not self.reader.is_opened():
            return False

        if not self.speed_timeline:
            answer = QMessageBox.question(
                self,
                "Export",
                "Aucune analyse des vitesses n’a été réalisée. Voulez-vous la lancer maintenant ?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return False
            if not self._perform_speed_analysis():
                return False

        ctx = self._current_timeline_context()
        if self._timeline_context is None or ctx != self._timeline_context:
            answer = QMessageBox.question(
                self,
                "Export",
                "Les paramètres ont changé depuis la dernière analyse. Relancer l’analyse maintenant ?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return False
            if not self._perform_speed_analysis():
                return False
        return bool(self.speed_timeline)

    def _status_begin_task(self, task_id: str, title: str, total: Optional[int]) -> None:
        if not getattr(self, "_status_layout", None):
            return
        if task_id in self._status_tasks:
            self._status_finish_task(task_id, immediate=True)

        container = QWidget(self)
        box_layout = QVBoxLayout(container)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(4)

        title_label = QLabel(title, container)
        title_label.setStyleSheet("font-weight: bold;")
        box_layout.addWidget(title_label)

        progress = QProgressBar(container)
        progress.setMinimum(0)
        total_value = int(total) if total is not None and total > 0 else None
        if total_value:
            progress.setMaximum(total_value)
        else:
            progress.setRange(0, 0)
        box_layout.addWidget(progress)

        eta_label = QLabel("Temps restant estimé : calcul…", container)
        eta_label.setStyleSheet("color: #888; font-size: 11px;")
        box_layout.addWidget(eta_label)

        self._status_layout.addWidget(container)
        if getattr(self, "status_empty_label", None) is not None:
            self.status_empty_label.hide()

        self._status_tasks[task_id] = {
            "widget": container,
            "progress": progress,
            "eta": eta_label,
            "start": time.monotonic(),
            "total": total_value,
            "title": title,
        }

    def _status_update_task(self, task_id: str, value: int, total: Optional[int] = None) -> None:
        info = self._status_tasks.get(task_id)
        if not info:
            return

        progress: QProgressBar = info["progress"]
        if total is not None:
            total_value = int(total) if total and total > 0 else None
            info["total"] = total_value
            if total_value:
                progress.setRange(0, total_value)
            else:
                progress.setRange(0, 0)

        total_frames = info.get("total")
        if total_frames:
            clamped = max(0, min(int(value), total_frames))
            progress.setValue(clamped)
            if clamped > 0:
                elapsed = max(0.0, time.monotonic() - info["start"])
                rate = elapsed / clamped if clamped else None
                if rate and np.isfinite(rate):
                    remaining = max(0.0, (total_frames - clamped) * rate)
                    eta_txt = self._format_hms(remaining)
                    info["eta"].setStyleSheet("color: #888; font-size: 11px;")
                    info["eta"].setText(f"Temps restant estimé : {eta_txt}")
                else:
                    info["eta"].setText("Temps restant estimé : calcul…")
            else:
                info["eta"].setText("Temps restant estimé : calcul…")
        else:
            # Indeterminate task
            progress.setRange(0, 0)
            info["eta"].setStyleSheet("color: #888; font-size: 11px;")
            info["eta"].setText("Temps restant estimé : indéterminé")

    def _status_finish_task(
        self,
        task_id: str,
        *,
        message: Optional[str] = None,
        success: bool = True,
        immediate: bool = False,
    ) -> None:
        info = self._status_tasks.get(task_id)
        if not info:
            return

        progress: QProgressBar = info["progress"]
        if progress.maximum() == 0:
            progress.setRange(0, 1)
        progress.setValue(progress.maximum())

        if message is None:
            message = "Terminé" if success else "Interrompu"

        color = "#2e7d32" if success else "#c62828"
        info["eta"].setStyleSheet(f"color: {color}; font-size: 11px;")
        info["eta"].setText(message)

        def cleanup():
            widget = info["widget"]
            widget.setParent(None)
            widget.deleteLater()
            self._status_tasks.pop(task_id, None)
            if not self._status_tasks and getattr(self, "status_empty_label", None) is not None:
                self.status_empty_label.show()

        if immediate:
            cleanup()
        else:
            QTimer.singleShot(2000, cleanup)

    def _collect_smoothing_config(self) -> AntiJitterConfig:
        return AntiJitterConfig(
            median_past_frames=int(self.spin_smooth_past.value()),
            median_future_frames=int(self.spin_smooth_future.value()),
            max_delta_kmh=float(self.spin_smooth_delta.value()),
            min_confidence=float(self.spin_smooth_conf.value()),
            hold_max_gap_frames=int(self.spin_smooth_hold.value()),
        )

    def _load_smoothing_settings(self) -> None:
        if not hasattr(self, "spin_smooth_past"):
            return

        self._loading_smoothing_settings = True
        try:
            cfg = self.ocr.anti_jitter
            past = self._settings.value("smoothing/past_frames", cfg.median_past_frames, type=int)
            future = self._settings.value("smoothing/future_frames", cfg.median_future_frames, type=int)
            legacy_window = self._settings.value("smoothing/window_size", None, type=int)
            delta = self._settings.value("smoothing/max_delta", cfg.max_delta_kmh, type=float)
            conf = self._settings.value("smoothing/min_confidence", cfg.min_confidence, type=float)
            hold = self._settings.value("smoothing/hold_frames", cfg.hold_max_gap_frames, type=int)

            if past is None:
                past = cfg.median_past_frames
            if future is None:
                future = cfg.median_future_frames
            if legacy_window is not None and past == cfg.median_past_frames and future == cfg.median_future_frames:
                legacy_window = max(1, int(legacy_window))
                legacy_past = max(0, (legacy_window - 1) // 2)
                legacy_future = max(0, legacy_window - 1 - legacy_past)
                past = legacy_past
                future = legacy_future
            if delta is None:
                delta = cfg.max_delta_kmh
            if conf is None:
                conf = cfg.min_confidence
            if hold is None:
                hold = cfg.hold_max_gap_frames

            self.spin_smooth_past.blockSignals(True)
            self.spin_smooth_past.setValue(int(past))
            self.spin_smooth_past.blockSignals(False)

            self.spin_smooth_future.blockSignals(True)
            self.spin_smooth_future.setValue(int(future))
            self.spin_smooth_future.blockSignals(False)

            self.spin_smooth_delta.blockSignals(True)
            self.spin_smooth_delta.setValue(float(delta))
            self.spin_smooth_delta.blockSignals(False)

            self.spin_smooth_conf.blockSignals(True)
            self.spin_smooth_conf.setValue(float(conf))
            self.spin_smooth_conf.blockSignals(False)

            self.spin_smooth_hold.blockSignals(True)
            self.spin_smooth_hold.setValue(int(hold))
            self.spin_smooth_hold.blockSignals(False)
        finally:
            self._loading_smoothing_settings = False

        config = self._collect_smoothing_config()
        self.ocr.set_anti_jitter_config(config)

    def _save_smoothing_settings(self, config: AntiJitterConfig) -> None:
        self._settings.setValue("smoothing/past_frames", int(config.median_past_frames))
        self._settings.setValue("smoothing/future_frames", int(config.median_future_frames))
        self._settings.setValue("smoothing/max_delta", float(config.max_delta_kmh))
        self._settings.setValue("smoothing/min_confidence", float(config.min_confidence))
        self._settings.setValue("smoothing/hold_frames", int(config.hold_max_gap_frames))
        self._settings.sync()

    def _on_smoothing_params_changed(self):
        if self._loading_smoothing_settings:
            return
        config = self._collect_smoothing_config()
        self.ocr.set_anti_jitter_config(config)
        self._save_smoothing_settings(config)
        if self.speed_timeline:
            self._invalidate_speed_timeline("Paramètres de lissage modifiés")

    def _save_range_settings(self) -> None:
        self._settings.setValue("range/enabled", bool(self.transcription_range_enabled))
        self._settings.setValue("range/start_sec", float(self.transcription_start_sec))
        self._settings.setValue("range/end_sec", float(self.transcription_end_sec))
        self._settings.sync()

    def _configure_range_controls(self, total_seconds: float) -> None:
        if not hasattr(self, "spin_range_start"):
            return

        total_seconds = max(0.0, float(total_seconds))
        self._loading_range_settings = True
        try:
            for spin in (self.spin_range_start, self.spin_range_end):
                spin.setMinimum(0.0)
                spin.setMaximum(total_seconds)

            if total_seconds <= 0.0:
                self.spin_range_start.setValue(0.0)
                self.spin_range_end.setValue(0.0)
                self.spin_range_start.setEnabled(False)
                self.spin_range_end.setEnabled(False)
                self.chk_range_enable.setChecked(False)
                self.transcription_range_enabled = False
                self.transcription_start_sec = 0.0
                self.transcription_end_sec = 0.0
            else:
                # Clamp stored values within the duration
                start = min(max(0.0, self.transcription_start_sec), total_seconds)
                if self.transcription_end_sec <= 0.0:
                    end = total_seconds
                else:
                    end = min(max(self.transcription_end_sec, start), total_seconds)

                self.transcription_start_sec = start
                self.transcription_end_sec = max(start, end)

                self.spin_range_start.setValue(self.transcription_start_sec)
                self.spin_range_end.setValue(self.transcription_end_sec)

                self.chk_range_enable.setChecked(bool(self.transcription_range_enabled))
                self.spin_range_start.setEnabled(self.transcription_range_enabled)
                self.spin_range_end.setEnabled(self.transcription_range_enabled)
        finally:
            self._loading_range_settings = False

        self._save_range_settings()
        self._refresh_overlay_for_range()

    def _on_range_toggle(self, state: int) -> None:
        if not hasattr(self, "spin_range_start"):
            return
        enabled = state == Qt.Checked and self.spin_range_start.maximum() > 0.0
        self.transcription_range_enabled = enabled
        self.spin_range_start.setEnabled(enabled)
        self.spin_range_end.setEnabled(enabled)
        if not self._loading_range_settings:
            self._save_range_settings()
            self._refresh_overlay_for_range()

    def _on_range_start_changed(self, value: float) -> None:
        if self._loading_range_settings:
            return
        start = max(0.0, float(value))
        end = float(self.spin_range_end.value())
        if start > end:
            self.spin_range_end.blockSignals(True)
            self.spin_range_end.setValue(start)
            self.spin_range_end.blockSignals(False)
            end = start
        self.transcription_start_sec = start
        self.transcription_end_sec = max(start, end)
        self._save_range_settings()
        self._refresh_overlay_for_range()

    def _on_range_end_changed(self, value: float) -> None:
        if self._loading_range_settings:
            return
        end = max(0.0, float(value))
        start = float(self.spin_range_start.value())
        if end < start:
            self.spin_range_start.blockSignals(True)
            self.spin_range_start.setValue(end)
            self.spin_range_start.blockSignals(False)
            start = end
        self.transcription_end_sec = end
        self.transcription_start_sec = min(start, end)
        self._save_range_settings()
        self._refresh_overlay_for_range()

    def _has_transcription_range(self) -> bool:
        return bool(
            self.transcription_range_enabled
            and self.transcription_end_sec > self.transcription_start_sec
            and self._total_frames() > 0
        )

    def _transcription_range_frame_bounds(
        self,
        *,
        fps: Optional[float] = None,
        total_frames: Optional[int] = None,
    ) -> tuple[int, int]:
        if total_frames is None:
            total_frames = self._total_frames()
        total = max(0, int(total_frames))
        if total <= 0:
            return 0, -1

        if fps is None:
            fps = self._fps()
        fps = max(float(fps), 1e-6)

        if not self.transcription_range_enabled:
            return 0, total - 1

        start_sec = max(0.0, self.transcription_start_sec)
        end_sec = self.transcription_end_sec
        if end_sec <= 0.0:
            end_sec = total / fps
        end_sec = max(start_sec, end_sec)

        start_idx = int(math.floor(start_sec * fps + 1e-6))
        end_idx = int(math.floor(end_sec * fps + 1e-6))
        start_idx = max(0, min(start_idx, total - 1))
        end_idx = max(0, min(end_idx, total - 1))
        if end_idx < start_idx:
            end_idx = start_idx
        return start_idx, end_idx

    def _frame_in_transcription_range(self, frame_idx: int) -> bool:
        if not self.transcription_range_enabled:
            return True
        start, end = self._transcription_range_frame_bounds()
        if end < start:
            return False
        return start <= frame_idx <= end

    def _current_frame_index(self) -> Optional[int]:
        if not self.reader or not self.reader.is_opened():
            return None
        try:
            return int(self.reader.get_pos_frame())
        except Exception:
            fps = self._fps()
            if fps <= 0:
                return 0
            ms = self.reader.get_pos_msec()
            return int(round((ms / 1000.0) * fps))

    def _refresh_overlay_for_range(self) -> None:
        if not hasattr(self, "canvas"):
            return
        if not self.reader or not self.reader.is_opened():
            if self.transcription_range_enabled:
                self._set_overlay_text(None, allow_placeholder=True, force_clear=True)
            return
        frame_idx = self._current_frame_index()
        if frame_idx is None:
            return
        if self._frame_in_transcription_range(frame_idx):
            if self._last_overlay_text:
                self._set_overlay_text(self._last_overlay_text)
            else:
                self._set_overlay_text(None, allow_placeholder=True)
        else:
            self._set_overlay_text(None, force_clear=True)

    def _update_debug_preview_geometry(self) -> None:
        if not hasattr(self, "debug_preview"):
            return
        size = max(64, int(self.debug_thumb_size))
        self.debug_preview.setFixedSize(size, size)

    def _bgr_to_pixmap(self, debug_bgr: np.ndarray) -> Optional[QPixmap]:
        if debug_bgr.ndim == 2:
            rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_GRAY2RGB)
            fmt = QImage.Format_RGB888
        elif debug_bgr.ndim == 3 and debug_bgr.shape[2] == 3:
            rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
        elif debug_bgr.ndim == 3 and debug_bgr.shape[2] == 4:
            rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGRA2RGBA)
            fmt = QImage.Format_RGBA8888
        else:
            return None
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], fmt)
        return QPixmap.fromImage(qimg.copy())

    def _set_debug_thumbnail(self, debug_bgr: Optional[np.ndarray]) -> None:
        self._last_debug_bgr = None if debug_bgr is None else debug_bgr.copy()
        self._refresh_debug_thumbnail()

    def _refresh_debug_thumbnail(self) -> None:
        if not hasattr(self, "debug_preview"):
            return
        if not self.show_debug_thumb:
            self.debug_preview.setPixmap(QPixmap())
            self.debug_preview.setText("Vignette désactivée")
            return
        if self._last_debug_bgr is None or self._last_debug_bgr.size == 0:
            self.debug_preview.setPixmap(QPixmap())
            self.debug_preview.setText("Aucune vignette")
            return
        pixmap = self._bgr_to_pixmap(self._last_debug_bgr)
        if pixmap is None or pixmap.isNull():
            self.debug_preview.setPixmap(QPixmap())
            self.debug_preview.setText("Aucune vignette")
            return
        scaled = pixmap.scaled(self.debug_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.debug_preview.setPixmap(scaled)
        self.debug_preview.setText("")

    def _create_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&Fichier")
        act_open = QAction("Ouvrir…", self)
        act_export = QAction("Exporter…", self)
        act_quit = QAction("Quitter", self)
        act_open.triggered.connect(self._on_open)
        act_export.triggered.connect(self._on_export)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_open)
        file_menu.addAction(act_export)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        edit_menu = menubar.addMenu("&Édition")
        act_settings = QAction("Paramètres OCR…", self)
        act_settings.triggered.connect(self._on_settings)
        edit_menu.addAction(act_settings)

        view_menu = menubar.addMenu("&Affichage")
        act_fit_roi = QAction("Ajuster le ROI à l’image", self)
        act_fit_roi.triggered.connect(self.canvas.fit_roi_to_frame)
        view_menu.addAction(act_fit_roi)

    # ------------- Utilitaires -------------

    def _on_toggle_overlay_mode(self, checked: bool) -> None:
        self.canvas.set_active_shape("overlay" if checked else "roi")
        self._update_overlay_mode_button()

    def _update_overlay_mode_button(self) -> None:
        if self.btn_overlay.isChecked():
            self.btn_overlay.setText("Zone de sortie (édition)")
        else:
            self.btn_overlay.setText("Tracer zone de sortie")

    def _on_edit_overlay_style(self) -> None:
        dlg = OverlayStyleDialog(self, style=self.overlay_style)
        if dlg.exec_():
            self.overlay_style = dlg.result_style
            self._save_overlay_style()
            self._set_overlay_text(self._last_overlay_text, allow_placeholder=True)

    def _set_overlay_text(
        self,
        text: Optional[str],
        *,
        allow_placeholder: bool = False,
        force_clear: bool = False,
    ) -> None:
        if force_clear:
            display = "-- mph" if allow_placeholder else None
        else:
            display = text
            if text:
                self._last_overlay_text = text
            else:
                display = self._last_overlay_text
                if display is None and allow_placeholder:
                    display = "-- mph"
        self.canvas.set_overlay_preview(display, self.overlay_style)

    @staticmethod
    def _rgba_to_string(rgba: tuple[int, int, int, int]) -> str:
        return "#{:02X}{:02X}{:02X}{:02X}".format(*rgba)

    @staticmethod
    def _string_to_rgba(value: Optional[str]) -> Optional[tuple[int, int, int, int]]:
        if not value:
            return None
        txt = value.strip()
        if not txt:
            return None
        if txt.startswith("#"):
            txt = txt[1:]
        try:
            if len(txt) == 6:
                r = int(txt[0:2], 16)
                g = int(txt[2:4], 16)
                b = int(txt[4:6], 16)
                a = 255
            elif len(txt) == 8:
                r = int(txt[0:2], 16)
                g = int(txt[2:4], 16)
                b = int(txt[4:6], 16)
                a = int(txt[6:8], 16)
            else:
                return None
        except ValueError:
            return None
        return r, g, b, a

    @staticmethod
    def _coerce_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _load_overlay_style(self, base: OverlayStyle) -> OverlayStyle:
        style = base
        family = self._settings.value("overlay/font_family", type=str)
        if family:
            style = replace(style, font_family=family)
        size = self._settings.value("overlay/font_point_size", type=int)
        if size:
            style = replace(style, font_point_size=int(size))
        text_color = self._string_to_rgba(self._settings.value("overlay/text_color", type=str))
        if text_color:
            style = replace(style, text_color_rgba=text_color)
        bg_color = self._string_to_rgba(self._settings.value("overlay/bg_color", type=str))
        if bg_color:
            style = replace(style, bg_color_rgba=bg_color)
        fill_value = self._settings.value("overlay/fill_opacity")
        if fill_value is not None:
            try:
                fill = float(fill_value)
            except (TypeError, ValueError):
                fill = style.fill_opacity
            else:
                fill = max(0.0, min(1.0, fill))
            style = replace(style, fill_opacity=fill)
        quality_value = self._settings.value("overlay/quality_scale")
        if quality_value is not None:
            try:
                quality = float(quality_value)
            except (TypeError, ValueError):
                quality = style.quality_scale
            else:
                quality = max(1.0, min(4.0, quality))
            style = replace(style, quality_scale=quality)
        return style

    def _save_overlay_style(self) -> None:
        self._settings.setValue("overlay/font_family", self.overlay_style.font_family)
        self._settings.setValue("overlay/font_point_size", self.overlay_style.font_point_size)
        self._settings.setValue("overlay/text_color", self._rgba_to_string(self.overlay_style.text_color_rgba))
        self._settings.setValue("overlay/bg_color", self._rgba_to_string(self.overlay_style.bg_color_rgba))
        self._settings.setValue("overlay/fill_opacity", float(self.overlay_style.fill_opacity))
        self._settings.setValue("overlay/quality_scale", float(getattr(self.overlay_style, "quality_scale", 1.0)))
        self._settings.sync()

    def _load_overlay_rect(self) -> None:
        rect_str = self._settings.value("overlay/rect", type=str)
        if not rect_str:
            return
        parts = rect_str.split(",")
        if len(parts) != 5:
            return
        try:
            cx, cy, w, h, ang = [float(p) for p in parts]
        except ValueError:
            return
        self.canvas.set_overlay_rect(int(round(cx)), int(round(cy)), int(round(w)), int(round(h)), float(ang))

    def _save_overlay_rect(self) -> None:
        cx, cy, w, h, ang = self.canvas.get_overlay_rect()
        rect_str = f"{cx},{cy},{w},{h},{ang}"
        self._settings.setValue("overlay/rect", rect_str)
        self._settings.sync()

    def _on_overlay_rect_changed(self) -> None:
        if self._loading_overlay_settings:
            return
        self._save_overlay_rect()

    def _format_hms(self, seconds: float) -> str:
        if seconds < 0 or not np.isfinite(seconds):
            return "00:00"
        s = int(round(seconds))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _fps(self) -> float:
        return float(self.reader.fps or 25.0) if self.reader else 25.0

    def _total_frames(self) -> int:
        return int(self.reader.frame_count) if (self.reader and self.reader.frame_count > 0) else 0

    # ------------- Raccourcis clavier -------------

    def keyPressEvent(self, ev) -> None:
        if not self.reader or not self.reader.is_opened():
            return super().keyPressEvent(ev)
        key = ev.key()

        if key == Qt.Key_Space:
            self._on_toggle_play(); ev.accept(); return
        if key == Qt.Key_Left:
            self._step_frames(-1); ev.accept(); return
        if key == Qt.Key_Right:
            self._step_frames(+1); ev.accept(); return
        if key == Qt.Key_PageUp:
            self._step_seconds(+1.0); ev.accept(); return
        if key == Qt.Key_PageDown:
            self._step_seconds(-1.0); ev.accept(); return
        if key == Qt.Key_Home:
            self._seek_to_frame(0); ev.accept(); return
        if key == Qt.Key_End:
            self._seek_to_frame(max(0, self._total_frames() - 1)); ev.accept(); return

        super().keyPressEvent(ev)

    def _step_frames(self, delta_frames: int) -> None:
        if not self.reader:
            return
        was_playing = self.playing
        if was_playing:
            self._on_toggle_play()  # pause
        cur_ms = self.reader.get_pos_msec()
        cur = int(round((cur_ms / 1000.0) * self._fps()))
        target = int(np.clip(cur + delta_frames, 0, max(0, self._total_frames() - 1)))
        self._seek_to_frame(target)
        if was_playing:
            self._on_toggle_play()

    def _step_seconds(self, delta_seconds: float) -> None:
        frames = int(round(delta_seconds * self._fps()))
        self._step_frames(frames)

    def _seek_to_frame(self, target_frame: int) -> None:
        if not self.reader:
            return
        self._seeking = True
        try:
            if not self.reader.set_pos_frame(target_frame):
                ms = (target_frame / self._fps()) * 1000.0
                self.reader.seek_msec(ms)
            ok, frame = self.reader.read()
            if ok and frame is not None:
                self.canvas.set_frame(frame)

            if self.slider.isEnabled():
                self.slider.blockSignals(True)
                self.slider.setValue(max(0, min(target_frame, max(0, self._total_frames() - 1))))
                self.slider.blockSignals(False)

            total_seconds = (self._total_frames() / self._fps()) if self._total_frames() > 0 else 0.0
            self.lbl_time.setText(f"{self._format_hms(target_frame / self._fps())} / {self._format_hms(total_seconds)}")
            self._refresh_overlay_for_range()
        finally:
            self._seeking = False

    # ------------- Slots -------------

    def _on_mode_changed(self, txt: str):
        self.ocr_mode = txt
        self.ocr.reset()
        self._invalidate_speed_timeline("Mode OCR modifié")

    def _on_toggle_debug(self, state: int):
        self.show_debug_thumb = (state == Qt.Checked)
        self._refresh_debug_thumbnail()

    def _on_thumb_size(self, v: int):
        self.debug_thumb_size = int(v)
        self._update_debug_preview_geometry()
        self._refresh_debug_thumbnail()

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une vidéo", "", "Vidéos (*.mp4 *.m4v *.mov *.avi *.mkv);;Tous les fichiers (*)")
        if not path:
            return
        try:
            if self.reader is not None:
                self.reader.release()
            self.reader = VideoReader(path)
            self.reader.open()
        except Exception as e:
            QMessageBox.critical(self, "Ouverture", f"Échec : {e}")
            self.reader = None
            return

        ok, frame = self.reader.read()
        if not ok or frame is None:
            QMessageBox.warning(self, "Ouverture", "Impossible de lire la vidéo.")
            self.reader.release()
            self.reader = None
            return

        total_frames = self._total_frames()
        fps = self._fps()
        self.slider.setEnabled(total_frames > 0)
        self.slider.setRange(0, max(0, total_frames - 1))
        self.slider.setValue(0)
        total_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0
        self.lbl_time.setText(f"00:00 / {self._format_hms(total_seconds)}")

        self._configure_range_controls(total_seconds)

        self.canvas.set_frame(frame)
        self.canvas.fit_roi_to_frame()
        self.lbl_kmh.setText("-- km/h")
        self.lbl_mph.setText("-- mph")
        self._last_overlay_text = None
        self._set_overlay_text(None, allow_placeholder=True)
        self.ocr.reset()
        self._set_debug_thumbnail(None)
        self._invalidate_speed_timeline("Nouvelle vidéo")
        self._refresh_overlay_for_range()

    def _on_toggle_play(self):
        if not self.reader or not self.reader.is_opened():
            return
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Lecture")
        if self.playing:
            self.timer.start(int(1000 / max(1.0, self._fps())))
        else:
            self.timer.stop()

    def _on_tick(self):
        if not self.reader:
            return
        ok, frame = self.reader.read()
        if not ok or frame is None:
            self._on_toggle_play()
            return

        self.canvas.set_frame(frame)

        # OCR sur le ROI courant — coins exacts du canvas
        cx, cy, w, h, ang = self.canvas.get_roi()
        corners = self.canvas.get_roi_corners()  # (4,2) TL,TR,BR,BL en coords image
        roi = _extract_roi_from_corners(frame, corners, w, h)

        try:
            kmh, debug_bgr, score, details = self.ocr.read_kmh(roi, mode=self.ocr_mode)
        except (TesseractNotFoundError, FileNotFoundError) as e:
            self._handle_tesseract_error(e)
            return

        if debug_bgr is not None and debug_bgr.size > 0:
            self._set_debug_thumbnail(debug_bgr)
        else:
            self._set_debug_thumbnail(None)

        mph_text: Optional[str] = None
        if kmh is not None:
            mph = kmh * KMH_TO_MPH
            mph_text = format_speed_text(mph)
            self.lbl_kmh.setText(f"{kmh:.0f} km/h")
            self.lbl_mph.setText(mph_text)
        else:
            self.lbl_kmh.setText("-- km/h")
            self.lbl_mph.setText("-- mph")

        # --- progression ---
        cur_ms = self.reader.get_pos_msec()
        fps = self._fps()
        cur_frame = int(round((cur_ms / 1000.0) * fps)) if fps > 0 else 0
        total_frames = self._total_frames()

        in_range = self._frame_in_transcription_range(cur_frame)
        if in_range:
            if mph_text:
                self._set_overlay_text(mph_text)
            else:
                self._set_overlay_text(None, allow_placeholder=True)
        else:
            self._set_overlay_text(None, force_clear=True)

        if not self._seeking and self.slider.isEnabled():
            self.slider.blockSignals(True)
            self.slider.setValue(max(0, min(cur_frame, max(0, total_frames - 1))))
            self.slider.blockSignals(False)

        total_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0
        self.lbl_time.setText(f"{self._format_hms(cur_ms/1000.0)} / {self._format_hms(total_seconds)}")

    def _on_settings(self):
        dlg = SettingsDialog(self, initial_path=self._tesseract_path)
        if dlg.exec_():
            prev_path = self._tesseract_path
            self._tesseract_path = dlg.tesseract_path or None
            if not self._apply_tesseract_path():
                self._tesseract_path = prev_path
                self._apply_tesseract_path(initial=True)

    def _apply_tesseract_path(self, *, initial: bool = False) -> bool:
        path = self._tesseract_path or None
        try:
            auto_locate_tesseract(path)
        except FileNotFoundError as e:
            if not initial:
                QMessageBox.warning(
                    self,
                    "Tesseract introuvable",
                    "Impossible de trouver l'exécutable Tesseract.\n"
                    "Vérifiez l'installation ou choisissez le bon fichier dans les Paramètres OCR.\n"
                    f"Détail : {e}",
                )
            return False
        else:
            self._tesseract_error_shown = False
            if path:
                self._save_tesseract_path()
            elif not initial:
                self._settings.remove("tesseract/path")
                self._settings.sync()
            return True

    def _save_tesseract_path(self) -> None:
        if self._tesseract_path:
            self._settings.setValue("tesseract/path", self._tesseract_path)
        else:
            self._settings.remove("tesseract/path")
        self._settings.sync()

    def _handle_tesseract_error(self, err: Exception) -> None:
        if self._tesseract_error_shown:
            return
        self._tesseract_error_shown = True
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.btn_play.setText("Lecture")
        self._set_debug_thumbnail(None)
        self.lbl_kmh.setText("-- km/h")
        self.lbl_mph.setText("-- mph")
        QMessageBox.critical(
            self,
            "Erreur Tesseract",
            "Tesseract n'a pas pu être exécuté.\n"
            "Veuillez vérifier l'installation et configurer le chemin dans Paramètres OCR…\n"
            f"Détail : {err}",
        )

    # ------------- Export -------------

    def _on_export(self):
        if not self.reader or not self.reader.is_opened():
            QMessageBox.information(self, "Export", "Ouvrez d’abord une vidéo.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Exporter la vidéo", "", "MP4 (*.mp4);;AVI (*.avi);;Tous les fichiers (*)")
        if not out_path:
            return

        if not self._ensure_speed_timeline_ready_for_export():
            return

        src = self.reader.source
        new_reader = VideoReader(src)
        try:
            new_reader.open()
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Impossible de rouvrir la source : {e}")
            return

        style = self.overlay_style
        out_cx, out_cy, out_w, out_h, out_ang = self.canvas.get_overlay_rect()

        fps_for_range = float(new_reader.fps or self._fps())
        known_total_frames = int(new_reader.frame_count) if new_reader.frame_count > 0 else None
        start_frame, end_frame = self._transcription_range_frame_bounds(
            fps=fps_for_range,
            total_frames=known_total_frames,
        )
        range_enabled = (
            self.transcription_range_enabled
            and known_total_frames is not None
            and end_frame >= start_frame
            and end_frame >= 0
        )

        last_mph_value = {"value": None}
        last_text_value = {"text": None}

        def _reset_last_values() -> None:
            last_mph_value["value"] = None
            last_text_value["text"] = None

        export_speed_decimals = 1

        def text_supplier(idx: int, frame_bgr: np.ndarray) -> Optional[str]:
            if range_enabled and (idx < start_frame or idx > end_frame):
                _reset_last_values()
                return None
            mph_val = self._timeline_mph_for_frame(idx, fps_for_range)
            if mph_val is None:
                cached_text = last_text_value["text"]
                if cached_text:
                    return cached_text
                return None
            last_mph_value["value"] = mph_val
            text = format_speed_text(mph_val, decimals=export_speed_decimals)
            last_text_value["text"] = text
            return text

        def draw_overlay(frame_bgr: np.ndarray, text: str) -> None:
            value = last_mph_value["value"]
            if value is None:
                try:
                    value = float(text.split()[0])
                except (ValueError, IndexError):
                    return
            draw_speed_overlay(
                frame_bgr,
                value,
                (out_cx, out_cy),
                out_ang,
                style,
                text=text,
                target_size=(out_w, out_h),
            )

        total = int(new_reader.frame_count) if new_reader.frame_count > 0 else None
        progress = QProgressDialog("Export en cours…", "Annuler", 0, total or 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        export_task_id = "export"
        self._status_begin_task(export_task_id, "Export vidéo", total)

        canceled = {"flag": False}
        export_error: Optional[Exception] = None

        def on_progress(done: int, total_opt: Optional[int]):
            if total_opt:
                progress.setMaximum(total_opt)
                progress.setValue(done)
                self._status_update_task(export_task_id, done, total_opt)
            else:
                progress.setValue(done % 100)
                self._status_update_task(export_task_id, done)
            QApplication.processEvents()
            if progress.wasCanceled():
                canceled["flag"] = True
                raise RuntimeError("Export annulé")

        try:
            export_video(
                new_reader,
                out_path,
                text_supplier=text_supplier,
                draw_overlay=draw_overlay,
                on_progress=on_progress,
                params=ExportParams(),
            )
            progress.setValue(progress.maximum())
            if total:
                self._status_update_task(export_task_id, total, total)
            QMessageBox.information(self, "Export", f"Fichier écrit :\n{out_path}")
            self._status_finish_task(export_task_id, message="Export terminé")
        except Exception as e:
            if not canceled["flag"]:
                export_error = e
                QMessageBox.warning(self, "Export", f"Échec : {e}")
        finally:
            try:
                new_reader.release()
            except Exception:
                pass
            if canceled["flag"]:
                self._status_finish_task(export_task_id, message="Export annulé", success=False)
            elif export_error is not None:
                self._status_finish_task(export_task_id, message="Échec de l'export", success=False)

    # ------------- Seek / Slider -------------

    def _on_seek_start(self):
        if not self.reader or not self.slider.isEnabled():
            return
        self._seeking = True
        self._was_playing_before_seek = self.playing
        if self.playing:
            self._on_toggle_play()  # pause

    def _on_seek_change(self, value: int):
        if not self.reader:
            return
        fps = self._fps()
        total_frames = self._total_frames()
        total_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0
        cur_seconds = (value / fps) if fps > 0 else 0.0
        self.lbl_time.setText(f"{self._format_hms(cur_seconds)} / {self._format_hms(total_seconds)}")

    def _on_seek_end(self):
        if not self.reader or not self.slider.isEnabled():
            self._seeking = False
            return
        target_frame = int(self.slider.value())
        try:
            if not self.reader.set_pos_frame(target_frame):
                ms = (target_frame / self._fps()) * 1000.0
                self.reader.seek_msec(ms)
            ok, frame = self.reader.read()
            if ok and frame is not None:
                self.canvas.set_frame(frame)
                self._refresh_overlay_for_range()
            else:
                self.reader.set_pos_frame(max(0, target_frame - 1))
                ok, frame = self.reader.read()
                if ok and frame is not None:
                    self.canvas.set_frame(frame)
                    self._refresh_overlay_for_range()
        finally:
            self._seeking = False
            if self._was_playing_before_seek:
                self._on_toggle_play()  # reprendre
