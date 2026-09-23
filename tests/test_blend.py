"""Unit tests for Stage 5 alignment/blend: verify the alignment sign convention empirically
(rather than trusting cross-correlation lag-sign reasoning), and that band blend avoids the
comb filtering that a naive misaligned time blend produces.
"""

import numpy as np

from app.dsp.blend import (
    align_to,
    apply_lag,
    band_blend,
    detect_comb_filtering,
    estimate_alignment_lag,
    time_blend,
)


def _noise(sr, seconds, seed=0):
    rng = np.random.default_rng(seed)
    return (0.3 * rng.standard_normal(int(sr * seconds))).astype(np.float32)


def test_apply_lag_shifts_in_the_documented_direction():
    x = np.arange(10, dtype=np.float32)
    # apply_lag(x, lag) => out[n] = x[n+lag]
    np.testing.assert_array_equal(apply_lag(x, 2), np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(apply_lag(x, -2), np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32))


def test_estimate_alignment_lag_recovers_a_known_shift():
    sr = 48000
    faithful = _noise(sr, 2.0)
    known_shift = 137  # generative "arrives late": generative[n + known_shift] = faithful[n]
    generative = np.zeros_like(faithful)
    generative[known_shift:] = faithful[: faithful.size - known_shift]

    lag = estimate_alignment_lag(faithful, generative, sr)
    assert lag == known_shift


def test_align_to_recovers_near_perfect_correlation():
    sr = 48000
    faithful = _noise(sr, 2.0)
    generative = np.zeros_like(faithful)
    shift = 250
    generative[shift:] = faithful[: faithful.size - shift]

    aligned, lag = align_to(faithful, generative, sr)
    assert lag == shift
    # Ignore the zero-padded edge introduced by the shift.
    correlation = np.corrcoef(faithful[shift:-1], aligned[shift:-1])[0, 1]
    assert correlation > 0.99


def test_band_blend_reconstructs_faithful_at_wet_zero():
    sr = 48000
    faithful = _noise(sr, 1.0)
    generative = _noise(sr, 1.0, seed=1)  # unrelated signal
    out = band_blend(faithful, generative, sr, crossover_hz=4000, wet=0.0)
    # LR4 split+recombine has small filter-edge error; should still track the input closely.
    assert np.corrcoef(faithful[1000:-1000], out[1000:-1000])[0, 1] > 0.97


def test_naive_misaligned_time_blend_triggers_comb_detection():
    sr = 48000
    faithful = _noise(sr, 2.0)
    # A short, un-aligned delay of a correlated copy of itself is the textbook comb-filter case.
    delay_samples = 20  # ~0.4ms at 48kHz -> notches every ~2.4kHz, well inside detection range
    delayed = np.zeros_like(faithful)
    delayed[delay_samples:] = faithful[: faithful.size - delay_samples]

    blended = time_blend(faithful, delayed, wet=0.5)
    flagged, details = detect_comb_filtering(blended, sr)
    assert flagged, details


def test_band_blend_does_not_trigger_comb_detection_on_the_same_case():
    sr = 48000
    faithful = _noise(sr, 2.0)
    delay_samples = 20
    delayed = np.zeros_like(faithful)
    delayed[delay_samples:] = faithful[: faithful.size - delay_samples]

    blended = band_blend(faithful, delayed, sr, crossover_hz=4000, wet=1.0)
    flagged, details = detect_comb_filtering(blended, sr)
    assert not flagged, details
