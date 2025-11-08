"""
Constantes et paramètres par défaut pour kmhtomph.
Séparées du reste pour éviter les imports circulaires et centraliser le tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Conversion
KMH_TO_MPH: float = 0.621371192237334  # 1 km/h = 0.62137 mph

# Valeurs aberrantes
MAX_REASONABLE_KMH: float = 500.0

# Texte incrusté (overlay) par défaut
DEFAULT_FONT_FAMILY: str = "DejaVu Sans"
DEFAULT_FONT_POINT_SIZE: int = 28
DEFAULT_TEXT_PADDING_PX: int = 6
DEFAULT_OUTLINE_THICKNESS_PX: int = 2
DEFAULT_FILL_OPACITY: float = 0.75  # 0..1 pour l'arrière-plan

# Couleurs RGBA (0..255) pour l'overlay texte
DEFAULT_TEXT_COLOR_RGBA: Tuple[int, int, int, int] = (255, 255, 255, 255)
DEFAULT_OUTLINE_COLOR_RGBA: Tuple[int, int, int, int] = (0, 0, 0, 255)
DEFAULT_BG_COLOR_RGBA: Tuple[int, int, int, int] = (0, 0, 0, 180)

# Anti-jitter / sélection de la meilleure valeur OCR
@dataclass(frozen=True)
class AntiJitterConfig:
    """
    Paramètres de lissage/anti-sauts pour la vitesse.

    - median_past_frames: nombre de frames en amont utilisées pour la médiane centrée
    - median_future_frames: nombre de frames en aval utilisées pour la médiane centrée
    - max_delta_kmh: saut max accepté d’une frame à l’autre (km/h), sinon on rejette
    - min_confidence: score mini pour accepter la valeur (0..1)
    - hold_max_gap_frames: nb de frames à “tenir” la dernière valeur quand tout rate
    """

    median_past_frames: int = 3
    median_future_frames: int = 3
    # Ajusté pour des FPS typiques (25–60) : on privilégie des transitions plus douces.
    max_delta_kmh: float = 2.0
    min_confidence: float = 0.55
    # NEW: maintien configurable de la dernière valeur quand on a des trous
    hold_max_gap_frames: int = 8

    @property
    def total_window(self) -> int:
        """Retourne la taille totale de la fenêtre médiane (passé + présent + futur)."""
        past = max(0, int(self.median_past_frames))
        future = max(0, int(self.median_future_frames))
        return past + future + 1


DEFAULT_ANTI_JITTER = AntiJitterConfig()

# Débogage
DEFAULT_SHOW_DEBUG_THUMB: bool = True  # affiche la vignette OCR quand dispo
DEFAULT_DEBUG_THUMB_SIZE: int = 160    # pixels (carré)
