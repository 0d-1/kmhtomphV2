
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

from .tesseract import tesseract_ocr, TesseractParams, DEFAULT_PARAMS as DEFAULT_TESS_PARAMS
from .sevenseg import sevenseg_ocr
from .chooser import AntiJitterState, choose_best_kmh, reset as reset_state, update_config as update_state_config
from ..constants import DEFAULT_ANTI_JITTER, AntiJitterConfig, MAX_REASONABLE_KMH


@dataclass
class OCRPipeline:
    anti_jitter: AntiJitterConfig = field(default_factory=lambda: DEFAULT_ANTI_JITTER)
    state: AntiJitterState = field(default_factory=lambda: AntiJitterState())
    tesseract_params: TesseractParams = field(default_factory=lambda: DEFAULT_TESS_PARAMS)
    auto_prefer_7seg_confidence: float = 0.75
    auto_prefer_7seg_min_digits: int = 2
    auto_prefer_7seg_delta_factor: float = 1.2

    def __post_init__(self):
        update_state_config(self.state, self.anti_jitter)

    def reset(self):
        reset_state(self.state)

    def set_anti_jitter_config(self, config: AntiJitterConfig) -> None:
        self.anti_jitter = config
        update_state_config(self.state, config)

    def read_kmh(self, roi_bgr: np.ndarray, mode: str = "auto"):
        """
        Returns (kmh_value:Optional[float], debug_bgr:Optional[np.ndarray], score:float, details:str)
        """
        candidates: List[Tuple[Optional[float], float, str, Optional[np.ndarray]]] = []
        dbg_best = None

        def _sanitize(raw: Optional[float]) -> Optional[float]:
            if raw is None:
                return None
            if not np.isfinite(raw):
                return None
            if raw < 0 or raw > float(MAX_REASONABLE_KMH):
                return None
            return float(raw)

        run_tesseract = mode in ("tesseract", "auto")
        prefer_7seg = False

        if mode in ("sevenseg", "auto"):
            txt7, conf7, dbg7 = sevenseg_ocr(roi_bgr)
            txt7_clean = txt7.strip() if txt7 else ""
            v7 = _sanitize(float(txt7_clean)) if txt7_clean else None
            candidates.append((v7, float(conf7), "7seg", dbg7))

            if (
                mode == "auto"
                and v7 is not None
                and conf7 >= float(self.auto_prefer_7seg_confidence)
                and len(txt7_clean) >= int(self.auto_prefer_7seg_min_digits)
            ):
                last = self.state.last_kmh
                delta_ok = (
                    last is None
                    or abs(float(v7) - float(last))
                    <= float(self.anti_jitter.max_delta_kmh) * float(self.auto_prefer_7seg_delta_factor)
                )
                if delta_ok:
                    run_tesseract = False
                    prefer_7seg = True

        if run_tesseract:
            txt, conf, dbg = tesseract_ocr(roi_bgr, self.tesseract_params)
            v = _sanitize(float(txt)) if txt is not None and txt.strip() else None
            candidates.append((v, float(conf), "tess", dbg))

        simple = [(v, c, s) for (v, c, s, d) in candidates]
        chosen = choose_best_kmh(simple, self.state, self.anti_jitter)

        if chosen is None:
            # choose a dbg to show
            for v,c,s,d in candidates:
                if s == "tess" and d is not None:
                    dbg_best = d; break
                if d is not None:
                    dbg_best = d
            return None, dbg_best, 0.0, "no-accept"

        # find confidence from the chosen source
        conf_src = 0.0
        src_used: Optional[str] = None
        for v,c,s,d in candidates:
            if v is not None and abs(v - chosen) < 1e-6:
                conf_src = float(c)
                src_used = s
                if d is not None: dbg_best = d
                break
        if dbg_best is None and candidates:
            dbg_best = candidates[0][3]

        detail = "ok"
        if src_used == "7seg" or (src_used is None and prefer_7seg):
            detail = "ok:7seg"
        elif src_used == "tess":
            detail = "ok:tess"
        return float(chosen), dbg_best, conf_src, detail
