"""Unit tests for DSP building blocks that do not require an engine."""

import numpy as np

from app.chunking import plan_chunks, recombine, split
from app.dsp.chain import FinishingParams, run_finishing_chain
from app.dsp.loudness import measure_lufs


def test_chunk_recombine_reconstructs_flat_signal_exactly():
    sr = 48000
    x = np.ones(sr * 130, dtype=np.float32)  # 130s, forces multiple chunks at 60s max
    plan = plan_chunks(x.size, sr, max_chunk_seconds=60, overlap_seconds=1.0)
    assert len(plan.starts) > 1

    chunks = split(x, plan)
    out = recombine(chunks, plan, x.size)

    assert out.size == x.size
    np.testing.assert_allclose(out, x, atol=1e-6)


def test_chunk_recombine_no_discontinuity_on_ramp_signal():
    sr = 48000
    n = sr * 125
    x = np.linspace(0, 1, n, dtype=np.float32)
    plan = plan_chunks(n, sr, max_chunk_seconds=60, overlap_seconds=1.0)
    chunks = split(x, plan)
    out = recombine(chunks, plan, n)

    # No large sample-to-sample jumps anywhere, including at former chunk borders.
    diffs = np.abs(np.diff(out))
    assert diffs.max() < 1e-4


def test_finishing_chain_no_nan_inf_and_hits_loudness_target():
    sr = 48000
    rng = np.random.default_rng(0)
    x = 0.2 * rng.standard_normal(sr * 5).astype(np.float32)

    params = FinishingParams(loudness_target="youtube")
    out = run_finishing_chain(x, sr, params)

    assert out.size == x.size
    assert np.all(np.isfinite(out))
    lufs = measure_lufs(out, sr)
    assert abs(lufs - (-14.0)) < 0.5


def test_finishing_chain_stem_mode_no_normalization_peak_target():
    sr = 48000
    rng = np.random.default_rng(1)
    x = 0.8 * rng.standard_normal(sr * 3).astype(np.float32)

    params = FinishingParams(loudness_target="stem", stem_peak_dbfs=-6.0)
    out = run_finishing_chain(x, sr, params)

    peak_dbfs = 20 * np.log10(np.max(np.abs(out)))
    assert abs(peak_dbfs - (-6.0)) < 0.2
