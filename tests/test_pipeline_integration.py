"""Phase 1 acceptance test: full CLI pipeline on the 10-minute noisy test file.

Requires the deepfilternet engine to be installed; skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio_io import read_wav_mono, write_wav
from app.chunking import plan_chunks
from app.dsp.loudness import measure_lufs
from app.engines import get_engine
from app.orchestrator import run_job

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = REPO_ROOT / "tests" / "audio" / "01_synthetic_noisy_speech_10min.wav"


def _engine_installed() -> bool:
    try:
        return get_engine("deepfilternet").installed
    except KeyError:
        return False


@pytest.mark.skipif(not _engine_installed(), reason="deepfilternet engine not installed")
def test_ten_minute_file_processes_cleanly():
    assert TEST_FILE.exists(), f"Missing test fixture: {TEST_FILE}"

    result = run_job(TEST_FILE, "clean_voiceover")

    in_x, in_sr = read_wav_mono(TEST_FILE)
    out_x, out_sr = read_wav_mono(result.output_wav)

    assert out_sr == in_sr
    # Sample-accurate length match.
    assert out_x.size == in_x.size

    # No NaN/inf.
    assert np.all(np.isfinite(out_x))

    # Loudness within 0.5 LU of the youtube target (-14 LUFS) from the clean_voiceover preset.
    lufs = measure_lufs(out_x, out_sr)
    assert abs(lufs - (-14.0)) < 0.5

    # Sanity net: catch a genuinely broken output (e.g. a dropped sample to full scale),
    # while allowing legitimate loud high-frequency content elsewhere in real speech.
    diffs = np.abs(np.diff(out_x))
    assert diffs.max() < 1.9

    # No chunk-border clicks: the sample-to-sample jump right at each chunk boundary must not
    # be an outlier compared to the jumps found everywhere else in the file.
    engine = get_engine("deepfilternet")
    plan = plan_chunks(in_x.size, in_sr, engine.max_chunk_seconds)
    assert len(plan.starts) > 1, "test file is too short to exercise chunk boundaries"

    baseline = float(np.percentile(diffs, 99.9))
    for boundary in plan.starts[1:]:
        window = diffs[max(0, boundary - 10) : boundary + 10]
        assert window.max() < baseline * 5, (
            f"Chunk boundary discontinuity at sample {boundary}: "
            f"{window.max()} vs baseline {baseline}"
        )


@pytest.mark.skipif(not _engine_installed(), reason="deepfilternet engine not installed")
def test_over_unity_peak_does_not_break_denoiser(tmp_path):
    """Regression test: a source with a true peak above +-1.0 (hot mic, or lossy-codec
    (e.g. AAC/m4a) decode overshoot) previously made DeepFilterNet emit a pathological
    full-scale oscillating burst instead of denoised audio. The pre-denoise safety
    limiter in orchestrator.run_job should prevent that unconditionally.
    """
    sr = 48000
    t = np.linspace(0, 3, sr * 3, endpoint=False, dtype=np.float32)
    x = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    # Inject a true-peak-over burst like AAC decode ringing around a loud transient.
    burst = slice(int(0.5 * sr), int(0.5 * sr) + 40)
    x[burst] = 1.3 * np.sin(np.linspace(0, 8 * np.pi, burst.stop - burst.start))

    src = tmp_path / "hot_mic.wav"
    write_wav(src, x, sr, subtype="FLOAT")

    result = run_job(src, "clean_voiceover", export_intermediate=True)
    denoised, _ = read_wav_mono(result.job_dir / "02_denoise.wav")

    diffs = np.abs(np.diff(denoised))
    assert diffs.max() < 1.0, f"Denoiser produced a full-scale artifact: max diff {diffs.max()}"
