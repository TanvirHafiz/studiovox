"""Stage 6: the finishing chain. Runs in the app env (pure numpy/scipy, no GPU)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.dsp import filters
from app.dsp.breath import reduce_breaths
from app.dsp.dynamics import compressor, de_esser
from app.dsp.loudness import normalize_and_limit

LOUDNESS_TARGETS = {
    "podcast": (-16.0, -1.0),
    "youtube": (-14.0, -1.0),
    "broadcast": (-23.0, -1.0),
    "stem": (None, None),  # no normalization, peak at -6 dBFS
}


@dataclass
class FinishingParams:
    high_pass_hz: float = 80.0
    high_pass_db_per_oct: int = 18
    mud_cut_hz: float = 300.0
    mud_cut_db: float = 0.0
    presence_hz: float = 4000.0
    presence_db: float = 0.0
    air_shelf_hz: float = 10000.0
    air_shelf_db: float = 0.0
    deesser_enabled: bool = True
    deesser_low_hz: float = 5000.0
    deesser_high_hz: float = 9000.0
    deesser_threshold_db: float = -24.0
    compressor_enabled: bool = True
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 3.0
    compressor_attack_ms: float = 15.0
    compressor_release_ms: float = 120.0
    breath_reduction_enabled: bool = False
    breath_attenuation_db: float = -8.0
    loudness_target: str = "youtube"  # key into LOUDNESS_TARGETS
    stem_peak_dbfs: float = -6.0


def run_finishing_chain(x: np.ndarray, sr: int, params: FinishingParams) -> np.ndarray:
    y = x.astype(np.float32)

    y = filters.high_pass(y, sr, params.high_pass_hz, params.high_pass_db_per_oct)
    y = filters.peaking_eq(y, sr, params.mud_cut_hz, params.mud_cut_db, q=1.0)
    y = filters.peaking_eq(y, sr, params.presence_hz, params.presence_db, q=1.0)
    y = filters.high_shelf(y, sr, params.air_shelf_hz, params.air_shelf_db)

    if params.deesser_enabled:
        y = de_esser(
            y,
            sr,
            low_hz=params.deesser_low_hz,
            high_hz=params.deesser_high_hz,
            threshold_db=params.deesser_threshold_db,
        )

    if params.compressor_enabled:
        y = compressor(
            y,
            sr,
            threshold_db=params.compressor_threshold_db,
            ratio=params.compressor_ratio,
            attack_ms=params.compressor_attack_ms,
            release_ms=params.compressor_release_ms,
        )

    if params.breath_reduction_enabled:
        y = reduce_breaths(y, sr, attenuation_db=params.breath_attenuation_db)

    target_lufs, ceiling_dbtp = LOUDNESS_TARGETS.get(params.loudness_target, (None, None))
    if target_lufs is not None:
        y = normalize_and_limit(y, sr, target_lufs, ceiling_dbtp)
    else:
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        target_peak = 10 ** (params.stem_peak_dbfs / 20)
        if peak > 0:
            y = y * (target_peak / peak)

    return y.astype(np.float32)
