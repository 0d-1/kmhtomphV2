
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Sequence

import numpy as np

from ..constants import DEFAULT_ANTI_JITTER, AntiJitterConfig


def _reset_memory(state: AntiJitterState) -> None:
    """Oublie la dernière valeur acceptée et vide la fenêtre médiane."""
    state.last_kmh = None
    state.win = _MedianWindow(state.config.median_past_frames)


class _MedianWindow:
    def __init__(self, size: int):
        self.size = max(0, int(size))
        self.buf: List[float] = []

    def add(self, v: float) -> None:
        self.buf.append(float(v))
        if len(self.buf) > self.size:
            self.buf.pop(0)

    def median(self) -> Optional[float]:
        if not self.buf:
            return None
        return float(np.median(self.buf))


@dataclass
class AntiJitterState:
    config: AntiJitterConfig = field(default_factory=lambda: DEFAULT_ANTI_JITTER)
    last_kmh: Optional[float] = None
    miss_streak: int = 0
    win: _MedianWindow = field(init=False)

    def __post_init__(self):
        self.win = _MedianWindow(self.config.median_past_frames)


def reset(state: AntiJitterState) -> None:
    state.last_kmh = None
    state.miss_streak = 0
    state.win = _MedianWindow(state.config.median_past_frames)


def update_config(state: AntiJitterState, config: AntiJitterConfig) -> None:
    """Met à jour la configuration de lissage tout en conservant l'état utile."""
    state.config = config
    last_value = state.last_kmh
    state.win = _MedianWindow(config.median_past_frames)
    if last_value is not None:
        state.win.add(float(last_value))
    state.miss_streak = min(state.miss_streak, int(config.hold_max_gap_frames))


def _score_candidate(value: float, conf: float, median_ref: Optional[float]) -> float:
    base = float(conf)
    if median_ref is None:
        return base
    bonus = max(0.0, 1.0 - (abs(float(value) - float(median_ref)) / 10.0))
    return base + 0.1 * bonus


def centered_median_smoothing(
    values: Sequence[Optional[float]],
    past: int,
    future: int,
) -> List[Optional[float]]:
    """Applique un filtre médian centré en tenant compte du passé et du futur."""

    past = max(0, int(past))
    future = max(0, int(future))
    n = len(values)
    smoothed: List[Optional[float]] = [None] * n

    for idx in range(n):
        start = max(0, idx - past)
        end = min(n, idx + future + 1)
        window = [v for v in values[start:end] if v is not None]
        if not window:
            smoothed[idx] = None
        else:
            smoothed[idx] = float(np.median(window))

    return smoothed


def choose_best_kmh(
    candidates: List[Tuple[Optional[float], float, str]],
    state: AntiJitterState,
    config: AntiJitterConfig = DEFAULT_ANTI_JITTER
) -> Optional[float]:
    filtered = [(v, c, s) for (v, c, s) in candidates if (v is not None and float(c) >= float(config.min_confidence))]
    if not filtered:
        state.miss_streak += 1
        if state.last_kmh is not None and state.miss_streak <= int(config.hold_max_gap_frames):
            return state.last_kmh
        _reset_memory(state)
        return None

    median_ref = state.win.median()
    scored = [(float(v), float(c), s, _score_candidate(v, c, median_ref)) for (v, c, s) in filtered]
    scored.sort(key=lambda t: (t[3], t[1]), reverse=True)
    best_v, best_c, best_s, _ = scored[0]

    if state.last_kmh is not None and abs(best_v - float(state.last_kmh)) > float(config.max_delta_kmh):
        for v, c, s, sc in scored[1:]:
            if abs(float(v) - float(state.last_kmh)) <= float(config.max_delta_kmh):
                best_v, best_c, best_s, _ = v, c, s, sc
                break
        else:
            state.miss_streak += 1
            if state.last_kmh is not None and state.miss_streak <= int(config.hold_max_gap_frames):
                return state.last_kmh
            _reset_memory(state)
            return None

    state.win.add(float(best_v))
    median_after = state.win.median()
    smoothed = float(median_after if median_after is not None else best_v)
    state.last_kmh = smoothed
    state.miss_streak = 0
    return smoothed
