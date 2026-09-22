"""De-esser and compressor: simple feed-forward dynamics processors."""

from __future__ import annotations

import numpy as np

from app.dsp.filters import band_pass


def _envelope(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000)) if attack_ms > 0 else 0.0
    release_coef = np.exp(-1.0 / (sr * release_ms / 1000)) if release_ms > 0 else 0.0
    env = np.zeros_like(x, dtype=np.float64)
    level = 0.0
    rectified = np.abs(x).astype(np.float64)
    for i in range(rectified.size):
        coef = attack_coef if rectified[i] > level else release_coef
        level = coef * level + (1 - coef) * rectified[i]
        env[i] = level
    return env


def compressor(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 15.0,
    release_ms: float = 120.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    env = _envelope(x, sr, attack_ms, release_ms)
    env_db = 20 * np.log10(np.maximum(env, 1e-8))
    over = env_db - threshold_db
    gain_reduction_db = np.where(over > 0, over * (1.0 / ratio - 1.0), 0.0)
    gain = 10 ** ((gain_reduction_db + makeup_db) / 20)
    return (x * gain).astype(np.float32)


def de_esser(
    x: np.ndarray,
    sr: int,
    low_hz: float = 5000.0,
    high_hz: float = 9000.0,
    threshold_db: float = -24.0,
    ratio: float = 4.0,
    attack_ms: float = 3.0,
    release_ms: float = 60.0,
) -> np.ndarray:
    """Dynamic band de-esser: compress only the sibilance band and add the reduced band back."""
    band = band_pass(x, sr, low_hz, high_hz)
    env = _envelope(band, sr, attack_ms, release_ms)
    env_db = 20 * np.log10(np.maximum(env, 1e-8))
    over = env_db - threshold_db
    gain_reduction_db = np.where(over > 0, over * (1.0 / ratio - 1.0), 0.0)
    gain = 10 ** (gain_reduction_db / 20)
    reduced_band = band * gain
    return (x - band + reduced_band).astype(np.float32)
