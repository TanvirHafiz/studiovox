"""Phase 5 acceptance test: sustained notes and vibrato must survive the singing
pipeline (vocal isolation + light dereverb), verified via pitch contour comparison
rather than just listening, per the plan's stated acceptance method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio_io import read_wav_mono
from app.engines import get_engine
from app.orchestrator import run_job

REPO_ROOT = Path(__file__).resolve().parent.parent
SINGING_FILE = REPO_ROOT / "tests" / "audio" / "10_singing_test.wav"


def _installed(name: str) -> bool:
    try:
        return get_engine(name).installed
    except KeyError:
        return False


def _track_pitch(x: np.ndarray, sr: int, frame_ms: float = 30.0, hop_ms: float = 10.0,
                  fmin: float = 150.0, fmax: float = 500.0) -> np.ndarray:
    """Simple autocorrelation-based F0 tracker. Returns one F0 estimate per hop (NaN where
    the frame isn't voiced/periodic enough to trust), good enough to compare contour shape
    between two versions of the same performance rather than for precise pitch measurement.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = max(0, (x.size - frame_len) // hop_len + 1)
    f0 = np.full(n_frames, np.nan)

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    window = np.hanning(frame_len)
    for i in range(n_frames):
        start = i * hop_len
        frame = x[start : start + frame_len].astype(np.float64) * window
        energy = np.sqrt(np.mean(frame**2))
        if energy < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
        if corr[0] <= 0 or max_lag >= corr.size:
            continue
        segment = corr[min_lag:max_lag]
        if segment.size == 0:
            continue
        peak_idx = int(np.argmax(segment)) + min_lag
        strength = corr[peak_idx] / corr[0]
        if strength > 0.5:
            f0[i] = sr / peak_idx
    return f0


@pytest.mark.skipif(not _installed("separator"), reason="separator engine not installed")
def test_singing_preset_preserves_pitch_contour():
    assert SINGING_FILE.exists(), f"Missing test fixture: {SINGING_FILE}"

    result = run_job(SINGING_FILE, "singing_vocal", export_intermediate=True)

    in_x, in_sr = read_wav_mono(SINGING_FILE)
    out_x, out_sr = read_wav_mono(result.job_dir / "output.wav")
    assert out_sr == in_sr

    f0_in = _track_pitch(in_x, in_sr)
    f0_out = _track_pitch(out_x, out_sr)

    n = min(f0_in.size, f0_out.size)
    valid = ~np.isnan(f0_in[:n]) & ~np.isnan(f0_out[:n])
    voiced_fraction = valid.sum() / n
    assert voiced_fraction > 0.7, (
        f"Only {voiced_fraction:.0%} of frames stayed voiced/periodic after processing; "
        f"the sustained note did not survive intact"
    )

    # Pitch contour shape (including the vibrato wobble) must track closely, not just be
    # "some pitch in the right ballpark" - correlation catches vibrato being smoothed away.
    correlation = np.corrcoef(f0_in[:n][valid], f0_out[:n][valid])[0, 1]
    assert correlation > 0.9, f"Pitch contour correlation only {correlation:.3f}"

    # The held note's average pitch should not have drifted to a different note/octave.
    mean_in = np.nanmean(f0_in[:n][valid])
    mean_out = np.nanmean(f0_out[:n][valid])
    assert abs(mean_in - mean_out) < 10, f"Mean pitch drifted: {mean_in:.1f} Hz -> {mean_out:.1f} Hz"

    # Vibrato depth (std dev of the pitch wobble around its own mean) should survive at a
    # comparable magnitude, not get flattened out by the processing.
    std_in = np.nanstd(f0_in[:n][valid])
    std_out = np.nanstd(f0_out[:n][valid])
    assert std_out > std_in * 0.4, (
        f"Vibrato depth collapsed: input std {std_in:.2f} Hz -> output std {std_out:.2f} Hz"
    )


@pytest.mark.skipif(not _installed("separator"), reason="separator engine not installed")
def test_singing_preset_uses_no_speech_engines(monkeypatch):
    """The singing preset itself must never invoke a speech-oriented engine (denoise via
    deepfilternet/clearervoice, super-resolution, generative restore).
    """
    import app.orchestrator as orch
    from app.presets import get_preset

    calls = []
    original = orch._run_engine_stage

    def spy(x, sr, engine_name, task, *args, **kwargs):
        calls.append((engine_name, task))
        return original(x, sr, engine_name, task, *args, **kwargs)

    monkeypatch.setattr(orch, "_run_engine_stage", spy)

    preset = get_preset("singing_vocal")
    assert preset.super_resolution_mode == "off"
    assert preset.generative_restore_enabled is False

    run_job(SINGING_FILE, "singing_vocal")

    speech_engines = {"deepfilternet", "clearervoice", "resemble"}
    used_engines = {engine for engine, _ in calls}
    assert not (used_engines & speech_engines), f"Singing preset invoked speech engine(s): {used_engines & speech_engines}"
