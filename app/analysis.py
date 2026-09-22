"""Stage 0: analyze ingested audio and produce the metrics stored in job.json."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pyloudnorm as pyln


@dataclass
class AnalysisResult:
    duration_seconds: float
    sample_rate: int
    peak_dbfs: float
    lufs: float
    clipping_percent: float
    noise_floor_dbfs: float
    bandwidth_hz: float
    mode_guess: str  # "speech" | "singing"
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _peak_dbfs(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return -120.0
    return 20.0 * np.log10(peak)


def _clipping_percent(x: np.ndarray, threshold: float = 0.999) -> float:
    if x.size == 0:
        return 0.0
    clipped = np.abs(x) >= threshold
    return 100.0 * float(np.count_nonzero(clipped)) / x.size


def _noise_floor_dbfs(x: np.ndarray, sr: int, frame_ms: float = 50.0) -> float:
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = x.size // frame_len
    if n_frames == 0:
        return -120.0
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    rms = rms[rms > 0]
    if rms.size == 0:
        return -120.0
    quietest_count = max(1, int(rms.size * 0.10))
    quietest = np.sort(rms)[:quietest_count]
    mean_rms = float(np.mean(quietest))
    if mean_rms <= 0:
        return -120.0
    return 20.0 * np.log10(mean_rms)


def _estimate_bandwidth_hz(x: np.ndarray, sr: int, rolloff_fraction: float = 0.99) -> float:
    """Spectral rolloff: the frequency below which rolloff_fraction of the energy lies."""
    if x.size == 0:
        return 0.0
    n = min(x.size, sr * 20)  # cap analysis window to 20s for speed
    segment = x[:n]
    window = np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(segment * window))
    power = spectrum**2
    total = np.sum(power)
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, rolloff_fraction * total))
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sr)
    idx = min(idx, freqs.size - 1)
    return float(freqs[idx])


def _guess_mode(x: np.ndarray, sr: int) -> str:
    """Rough speech-vs-singing heuristic: sustained-pitch fraction via autocorrelation periodicity."""
    frame_len = int(sr * 0.05)
    hop = frame_len // 2
    if x.size < frame_len * 4:
        return "speech"
    n_frames = (x.size - frame_len) // hop
    periodic_count = 0
    voiced_count = 0
    prev_period = None
    sustained_runs = 0
    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + frame_len]
        energy = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        if energy < 1e-4:
            prev_period = None
            continue
        frame = frame * np.hanning(frame.size)
        corr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
        min_lag = int(sr / 500)
        max_lag = int(sr / 70)
        if max_lag >= corr.size:
            continue
        segment = corr[min_lag:max_lag]
        if segment.size == 0 or corr[0] <= 0:
            continue
        peak_idx = int(np.argmax(segment)) + min_lag
        strength = corr[peak_idx] / corr[0]
        if strength > 0.55:
            voiced_count += 1
            if prev_period is not None and abs(peak_idx - prev_period) / peak_idx < 0.03:
                sustained_runs += 1
            prev_period = peak_idx
        else:
            prev_period = None
    if voiced_count == 0:
        return "speech"
    sustained_fraction = sustained_runs / voiced_count
    return "singing" if sustained_fraction > 0.5 else "speech"


def analyze(x: np.ndarray, sr: int) -> AnalysisResult:
    """x: mono float32 array in range [-1, 1]."""
    warnings: list[str] = []

    peak = _peak_dbfs(x)
    clip_pct = _clipping_percent(x)
    noise_floor = _noise_floor_dbfs(x, sr)
    bandwidth = _estimate_bandwidth_hz(x, sr)
    mode = _guess_mode(x, sr)
    try:
        lufs = float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
        if not np.isfinite(lufs):
            lufs = -70.0
    except Exception:  # noqa: BLE001
        lufs = -70.0

    if clip_pct > 0.1:
        warnings.append(f"Clipping detected ({clip_pct:.2f}% of samples). Cannot be fully repaired.")
    if bandwidth < 8000:
        warnings.append(f"Low effective bandwidth (~{bandwidth:.0f} Hz). Source may be phone quality.")
    if peak < -30:
        warnings.append(f"Very low level (peak {peak:.1f} dBFS). Consider gain staging before re-recording.")

    return AnalysisResult(
        duration_seconds=x.size / sr,
        sample_rate=sr,
        peak_dbfs=round(peak, 2),
        lufs=round(lufs, 2),
        clipping_percent=round(clip_pct, 4),
        noise_floor_dbfs=round(noise_floor, 2),
        bandwidth_hz=round(bandwidth, 1),
        mode_guess=mode,
        warnings=warnings,
    )
