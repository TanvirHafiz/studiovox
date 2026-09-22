"""Stage 1 (optional): classic declipper. Interpolates clipped runs with a constrained cubic fit.

This is a partial repair only, never a full recovery of clipped information.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def declip(x: np.ndarray, threshold: float = 0.99, max_run_samples: int = 400) -> np.ndarray:
    """PCHIP (shape-preserving Hermite) interpolation, not a plain cubic spline: an
    unconstrained cubic spline overshoots past the surrounding sample range right where
    a clipped run resumes (Runge's phenomenon), which can push the reconstructed peak
    higher than the original clipping - actively worse than doing nothing. PCHIP never
    overshoots the data it interpolates between.
    """
    y = x.copy()
    clipped = np.abs(x) >= threshold
    if not np.any(clipped):
        return y

    runs = _find_runs(clipped)
    for start, end in runs:
        length = end - start
        if length > max_run_samples:
            continue  # too long to guess reliably, leave as-is
        ctx = max(8, length)
        lo = max(0, start - ctx)
        hi = min(x.size, end + ctx)
        anchor_idx = np.concatenate([np.arange(lo, start), np.arange(end, hi)])
        if anchor_idx.size < 4:
            continue
        anchor_val = x[anchor_idx]
        try:
            interp = PchipInterpolator(anchor_idx, anchor_val)
            y[start:end] = interp(np.arange(start, end))
        except Exception:
            continue
    return y.astype(np.float32)


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, mask.size))
    return runs
