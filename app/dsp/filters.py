"""Biquad EQ filters (RBJ cookbook) and a cascaded high-pass."""

from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt, tf2sos


def _biquad_peaking(freq: float, sr: int, gain_db: float, q: float = 1.0) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * a
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return tf2sos(b, a)


def _biquad_low_shelf(freq: float, sr: int, gain_db: float, q: float = 0.707) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(a)

    b0 = a * ((a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = 2 * a * ((a - 1) - (a + 1) * cos_w0)
    b2 = a * ((a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = -2 * ((a - 1) + (a + 1) * cos_w0)
    a2 = (a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return tf2sos(b, a)


def _biquad_high_shelf(freq: float, sr: int, gain_db: float, q: float = 0.707) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(a)

    b0 = a * ((a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cos_w0)
    b2 = a * ((a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cos_w0)
    a2 = (a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return tf2sos(b, a)


def _biquad_highpass(freq: float, sr: int, q: float = 0.707) -> np.ndarray:
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return tf2sos(b, a)


def high_pass(x: np.ndarray, sr: int, freq: float, order_db_per_oct: int = 18) -> np.ndarray:
    """Cascaded 2nd-order high-pass sections to approximate the requested slope (12 dB/oct each)."""
    stages = max(1, round(order_db_per_oct / 12))
    y = x
    for _ in range(stages):
        sos = _biquad_highpass(freq, sr)
        y = sosfilt(sos, y)
    return y.astype(np.float32)


def peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    if abs(gain_db) < 0.01:
        return x
    sos = _biquad_peaking(freq, sr, gain_db, q)
    return sosfilt(sos, x).astype(np.float32)


def high_shelf(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 0.707) -> np.ndarray:
    if abs(gain_db) < 0.01:
        return x
    sos = _biquad_high_shelf(freq, sr, gain_db, q)
    return sosfilt(sos, x).astype(np.float32)


def low_shelf(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 0.707) -> np.ndarray:
    if abs(gain_db) < 0.01:
        return x
    sos = _biquad_low_shelf(freq, sr, gain_db, q)
    return sosfilt(sos, x).astype(np.float32)


def band_pass(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Used by the de-esser to isolate the sibilance band."""
    y = high_pass(x, sr, low, order_db_per_oct=12)
    sos_lp = tf2sos(*_lowpass_ba(high, sr))
    return sosfilt(sos_lp, y).astype(np.float32)


def _lowpass_ba(freq: float, sr: int, q: float = 0.707):
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a
