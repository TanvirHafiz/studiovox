"""Loudness normalization (ITU-R BS.1770 via pyloudnorm) and a true-peak limiter."""

from __future__ import annotations

import numpy as np
import pyloudnorm as pyln


def measure_lufs(x: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(x.astype(np.float64)))


def normalize_lufs(x: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    current = measure_lufs(x, sr)
    if not np.isfinite(current):
        return x
    gain_db = target_lufs - current
    gain = 10 ** (gain_db / 20)
    return (x * gain).astype(np.float32)


def true_peak_limiter(
    x: np.ndarray,
    sr: int,
    ceiling_dbtp: float = -1.0,
    oversample: int = 4,
    lookahead_ms: float = 5.0,
    release_ms: float = 60.0,
) -> np.ndarray:
    """Lookahead brickwall limiter with per-sample gain reduction, not a static gain scale.

    A single static gain sized to the loudest true peak would pull the whole track down to
    that one transient, which can miss the loudness target by many LU on peaky material. This
    only pulls gain down around the transients that actually exceed the ceiling, using a
    lookahead so the reduction starts just before the peak (no clicks) and a release so gain
    recovers smoothly afterward.
    """
    import soxr
    from scipy.ndimage import minimum_filter1d

    if x.size == 0:
        return x

    ceiling = 10 ** (ceiling_dbtp / 20)
    n = x.size

    oversampled = soxr.resample(x.astype(np.float32), sr, sr * oversample, quality="HQ")
    env_over = np.abs(oversampled)
    m = env_over.size // oversample
    env = env_over[: m * oversample].reshape(m, oversample).max(axis=1)
    if env.size < n:
        env = np.pad(env, (0, n - env.size), mode="edge")
    else:
        env = env[:n]

    target_gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))

    lookahead = max(1, int(sr * lookahead_ms / 1000))
    gain_lookahead = minimum_filter1d(target_gain, size=lookahead * 2 + 1, mode="nearest")

    # The attack/release envelope follower is inherently sequential, so it is computed on small
    # blocks (not per sample) to keep long files fast; each block takes the worst-case (minimum)
    # gain within it, so this never under-reduces relative to the per-sample envelope.
    block = max(1, int(sr * 0.002))  # ~2ms blocks
    n_blocks = int(np.ceil(gain_lookahead.size / block))
    padded = np.pad(gain_lookahead, (0, n_blocks * block - gain_lookahead.size), mode="edge")
    block_min = padded.reshape(n_blocks, block).min(axis=1)

    release_coef = np.exp(-block / (sr * release_ms / 1000))
    smoothed_blocks = np.empty(n_blocks, dtype=np.float64)
    level = 1.0
    for i in range(n_blocks):
        g = block_min[i]
        level = g if g < level else release_coef * level + (1 - release_coef) * g
        smoothed_blocks[i] = level

    # Linear interpolation (not repeat) between block centers avoids a stepped gain curve,
    # which would otherwise multiply a per-block discontinuity straight into the audio.
    block_centers = np.arange(n_blocks) * block + block / 2.0
    sample_positions = np.arange(gain_lookahead.size)
    smoothed = np.interp(sample_positions, block_centers, smoothed_blocks)

    y = x * smoothed
    # Safety clip for any residual inter-sample overs the envelope model missed.
    y = np.clip(y, -ceiling, ceiling)
    return y.astype(np.float32)


def normalize_and_limit(
    x: np.ndarray,
    sr: int,
    target_lufs: float,
    ceiling_dbtp: float = -1.0,
    max_iterations: int = 10,
    tolerance_lu: float = 0.3,
) -> np.ndarray:
    """Normalize to target_lufs then apply the lookahead limiter, iterating a few times.

    Limiting a peaky signal reduces its average loudness too (that is the point), so a single
    normalize-then-limit pass can undershoot the target. Each further iteration measures the
    actual result and applies the remaining correction, converging within a couple of rounds
    since the residual gain needed shrinks each time.
    """
    y = x.astype(np.float32)
    for _ in range(max_iterations):
        current = measure_lufs(y, sr)
        if not np.isfinite(current):
            break
        if abs(current - target_lufs) <= tolerance_lu:
            break
        gain_db = target_lufs - current
        y = (y * (10 ** (gain_db / 20))).astype(np.float32)
        y = true_peak_limiter(y, sr, ceiling_dbtp)
    return y
