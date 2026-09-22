"""Breath reduction: detect low-energy, spectrally flat regions between phrases and attenuate them."""

from __future__ import annotations

import numpy as np


def reduce_breaths(
    x: np.ndarray,
    sr: int,
    attenuation_db: float = -8.0,
    frame_ms: float = 30.0,
    energy_percentile: float = 35.0,
    flatness_threshold: float = 0.35,
) -> np.ndarray:
    frame_len = max(256, int(sr * frame_ms / 1000))
    hop = frame_len // 2
    n_frames = max(0, (x.size - frame_len) // hop + 1)
    if n_frames < 2:
        return x

    gain_curve = np.ones(x.size, dtype=np.float64)
    window = np.hanning(frame_len)

    energies = np.zeros(n_frames)
    flatness = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + frame_len].astype(np.float64) * window
        energies[i] = np.sqrt(np.mean(frame**2) + 1e-12)
        spectrum = np.abs(np.fft.rfft(frame)) + 1e-12
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness[i] = geo_mean / arith_mean

    energy_thresh = np.percentile(energies, energy_percentile)
    target_gain = 10 ** (attenuation_db / 20)

    for i in range(n_frames):
        is_breath = energies[i] < energy_thresh and energies[i] > 1e-4 and flatness[i] > flatness_threshold
        if is_breath:
            start = i * hop
            end = min(x.size, start + frame_len)
            gain_curve[start:end] = np.minimum(gain_curve[start:end], target_gain)

    # Smooth the gain curve to avoid abrupt jumps.
    smooth_len = max(1, int(sr * 0.01))
    kernel = np.ones(smooth_len) / smooth_len
    gain_curve = np.convolve(gain_curve, kernel, mode="same")

    return (x * gain_curve).astype(np.float32)
