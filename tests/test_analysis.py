"""Unit tests for Stage 0 analysis."""

import numpy as np

from app.analysis import analyze


def test_detects_clipping():
    sr = 48000
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    x = 1.5 * np.sin(2 * np.pi * 440 * t)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)

    result = analyze(x, sr)
    assert result.clipping_percent > 1.0
    assert any("clipping" in w.lower() for w in result.warnings)


def test_no_clipping_on_quiet_sine():
    sr = 48000
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    x = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    result = analyze(x, sr)
    assert result.clipping_percent == 0.0
    assert result.peak_dbfs < -15


def test_bandwidth_estimate_reflects_low_pass_content():
    sr = 48000
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    x = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)  # narrowband low tone

    result = analyze(x, sr)
    assert result.bandwidth_hz < 2000
    assert any("bandwidth" in w.lower() for w in result.warnings)
