"""Unit tests for the EQ/dynamics building blocks."""

import numpy as np

from app.dsp.filters import high_pass, peaking_eq, high_shelf, low_shelf
from app.dsp.dynamics import compressor, de_esser


def _tone(freq, sr=48000, seconds=1.0, amp=0.5):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_high_pass_attenuates_low_frequency():
    sr = 48000
    low = _tone(40, sr)
    y = high_pass(low, sr, freq=80)
    in_rms = np.sqrt(np.mean(low[sr // 2 :] ** 2))
    out_rms = np.sqrt(np.mean(y[sr // 2 :] ** 2))
    assert out_rms < in_rms * 0.3


def test_high_pass_passes_mid_frequency():
    sr = 48000
    mid = _tone(1000, sr)
    y = high_pass(mid, sr, freq=80)
    in_rms = np.sqrt(np.mean(mid[sr // 2 :] ** 2))
    out_rms = np.sqrt(np.mean(y[sr // 2 :] ** 2))
    assert out_rms > in_rms * 0.9


def test_low_shelf_boosts_low_frequency():
    sr = 48000
    low = _tone(100, sr)
    y = low_shelf(low, sr, freq=150, gain_db=6.0)
    in_rms = np.sqrt(np.mean(low[sr // 2 :] ** 2))
    out_rms = np.sqrt(np.mean(y[sr // 2 :] ** 2))
    assert out_rms > in_rms * 1.5  # roughly +6dB should about double amplitude


def test_low_shelf_leaves_high_frequency_unaffected():
    sr = 48000
    high = _tone(8000, sr)
    y = low_shelf(high, sr, freq=150, gain_db=6.0)
    in_rms = np.sqrt(np.mean(high[sr // 2 :] ** 2))
    out_rms = np.sqrt(np.mean(y[sr // 2 :] ** 2))
    assert abs(out_rms - in_rms) < in_rms * 0.1


def test_no_nan_inf_from_filters():
    sr = 48000
    rng = np.random.default_rng(0)
    x = 0.3 * rng.standard_normal(sr).astype(np.float32)
    y = high_pass(x, sr, 80)
    y = peaking_eq(y, sr, 300, -2.0)
    y = high_shelf(y, sr, 10000, 1.0)
    assert np.all(np.isfinite(y))


def test_compressor_reduces_peak_above_threshold():
    sr = 48000
    x = _tone(300, sr, amp=0.9)
    y = compressor(x, sr, threshold_db=-12, ratio=4.0)
    # Skip the attack transient at the very start; steady-state level should be reduced.
    assert np.max(np.abs(y[sr // 2 :])) < np.max(np.abs(x[sr // 2 :]))


def test_deesser_reduces_sibilance_band_energy():
    sr = 48000
    x = _tone(7000, sr, amp=0.5)  # inside the 5-9kHz sibilance band
    y = de_esser(x, sr, threshold_db=-40, ratio=8.0)
    in_rms = np.sqrt(np.mean(x[sr // 2 :] ** 2))
    out_rms = np.sqrt(np.mean(y[sr // 2 :] ** 2))
    assert out_rms < in_rms
