"""Stage 5 generative-restore support: phase/timing alignment, time blend, band blend,
and a comb-filtering guard.

Generative restoration models (Resemble Enhance) are not sample-accurate or phase-coherent
with their input: they can shift timing slightly and rebuild the waveform from scratch. Mixing
a shifted, phase-independent signal directly with the original produces comb filtering (evenly
spaced spectral notches from destructive interference). This module removes the gross timing
offset via cross-correlation, then offers two ways to combine the two signals:
  - time blend: a simple aligned mix, full-band. Still relies on sub-sample alignment being
    good enough to avoid audible comb filtering, so callers should check `detect_comb_filtering`.
  - band blend (default, safer): a Linkwitz-Riley crossover keeps the faithful signal below the
    crossover and only mixes in the generative signal above it, where any residual misalignment
    matters far less perceptually and there is much less spectral overlap to cancel against.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, correlate, correlation_lags, sosfiltfilt


def estimate_alignment_lag(faithful: np.ndarray, generative: np.ndarray, sr: int, max_lag_seconds: float = 0.5) -> int:
    """Returns the number of samples 'generative' must be shifted left (positive) or right
    (negative) to align with 'faithful', found via cross-correlation over a bounded search
    window (generative models do not introduce multi-second drift, so this stays cheap).
    """
    n = min(faithful.size, generative.size)
    if n == 0:
        return 0
    f = faithful[:n].astype(np.float64)
    g = generative[:n].astype(np.float64)

    max_lag = min(int(max_lag_seconds * sr), n - 1)
    corr = correlate(f, g, mode="full")
    lags = correlation_lags(f.size, g.size, mode="full")
    window = (lags >= -max_lag) & (lags <= max_lag)
    # correlate(f, g)'s peak lag L means f best matches g shifted right by L (f[n] ~ g[n-L]);
    # apply_lag's convention is out[n] = x[n+lag], i.e. a left shift, so the sign is inverted
    # to align g to f. Verified empirically in tests/test_blend.py against a known shift.
    raw_lag = int(lags[window][np.argmax(corr[window])])
    return -raw_lag


def apply_lag(x: np.ndarray, lag: int) -> np.ndarray:
    """Shifts x so that x_out[n] = x[n + lag], padding with zeros at the freed end.
    Pairs with estimate_alignment_lag: apply_lag(generative, lag) aligns it to 'faithful'.
    """
    if lag == 0:
        return x
    if lag > 0:
        return np.concatenate([x[lag:], np.zeros(lag, dtype=x.dtype)])
    return np.concatenate([np.zeros(-lag, dtype=x.dtype), x[:lag]])


def align_to(faithful: np.ndarray, generative: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Aligns generative to faithful and matches its length. Returns (aligned, lag_samples)."""
    lag = estimate_alignment_lag(faithful, generative, sr)
    aligned = apply_lag(generative, lag)
    if aligned.size < faithful.size:
        aligned = np.pad(aligned, (0, faithful.size - aligned.size))
    else:
        aligned = aligned[: faithful.size]
    return aligned.astype(np.float32), lag


def time_blend(faithful: np.ndarray, generative_aligned: np.ndarray, wet: float) -> np.ndarray:
    """Simple full-band aligned mix. See module docstring for the comb-filtering caveat."""
    return (wet * generative_aligned + (1 - wet) * faithful).astype(np.float32)


def _lr4_split(x: np.ndarray, sr: int, crossover_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """4th-order Linkwitz-Riley split: two cascaded 2nd-order Butterworth sections per band,
    applied zero-phase (sosfiltfilt) so low+high sum back to the original with no ripple or
    added phase distortion - the property that makes the crossover itself comb-filter-safe.
    """
    sos_lp = butter(2, crossover_hz, btype="low", fs=sr, output="sos")
    sos_hp = butter(2, crossover_hz, btype="high", fs=sr, output="sos")
    low = sosfiltfilt(sos_lp, sosfiltfilt(sos_lp, x))
    high = sosfiltfilt(sos_hp, sosfiltfilt(sos_hp, x))
    return low.astype(np.float32), high.astype(np.float32)


def band_blend(
    faithful: np.ndarray,
    generative_aligned: np.ndarray,
    sr: int,
    crossover_hz: float = 4000.0,
    wet: float = 1.0,
) -> np.ndarray:
    """Below crossover_hz: always the faithful signal. Above it: a wet/dry mix of the
    generative and faithful highs. At wet=0 this reconstructs the faithful signal exactly
    (up to the LR4 filter's own negligible reconstruction error); at wet=1 it is the full
    plan-described band blend (faithful lows, generative highs).
    """
    faithful_low, faithful_high = _lr4_split(faithful, sr, crossover_hz)
    _, generative_high = _lr4_split(generative_aligned, sr, crossover_hz)
    blended_high = wet * generative_high + (1 - wet) * faithful_high
    return (faithful_low + blended_high).astype(np.float32)


def detect_comb_filtering(
    x: np.ndarray,
    sr: int,
    periodicity_threshold: float = 0.35,
    min_lag_hz_spacing: float = 50.0,
) -> tuple[bool, dict]:
    """Comb filtering (destructive interference from summing a signal with a delayed,
    phase-independent copy of itself) produces a magnitude spectrum with strictly periodic
    ripple - notches evenly spaced by 1/delay Hz. That periodicity is what to detect, not
    individual dips: a per-bin envelope threshold fires constantly on ordinary spectral
    texture (verified empirically - it flagged thousands of "notches" with no consistent
    spacing on real audio). The right tool is autocorrelation of the ripple left after
    removing the coarse spectral envelope: true comb ripple produces one strong, sharp
    autocorrelation peak at the lag corresponding to the notch spacing; incidental spectral
    texture does not autocorrelate with itself at any single lag.
    """
    if x.size < sr // 10:
        return False, {"reason": "clip too short to analyze"}

    n = x.size
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(n)))
    spectrum_db = 20 * np.log10(spectrum + 1e-9)

    smooth_window = max(9, len(spectrum_db) // 20)
    envelope = maximum_filter1d(spectrum_db, size=smooth_window | 1)  # force odd window
    ripple = spectrum_db - envelope
    ripple = ripple - ripple.mean()

    autocorr = np.correlate(ripple, ripple, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    autocorr = autocorr / (autocorr[0] + 1e-9)

    bin_hz = sr / n
    min_lag = max(2, int(min_lag_hz_spacing / bin_hz))
    if min_lag >= autocorr.size:
        return False, {"reason": "clip too short for the minimum spacing checked"}

    search = autocorr[min_lag:]
    peak_offset = int(np.argmax(search))
    peak_lag = peak_offset + min_lag
    peak_value = float(autocorr[peak_lag])

    flagged = peak_value > periodicity_threshold
    return flagged, {
        "periodicity_strength": round(peak_value, 3),
        "notch_spacing_hz": round(peak_lag * bin_hz, 1),
    }
