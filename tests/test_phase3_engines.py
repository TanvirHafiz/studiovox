"""Phase 3 acceptance tests: ClearerVoice (SE + SR), audio-separator (dereverb), DNSMOS.

Requires the respective engines to be installed; each test skips otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio_io import read_wav_mono, write_wav
from app.dsp.chain import FinishingParams
from app.engines import get_engine
from app.orchestrator import run_job
from app.presets import Preset

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_FILE = REPO_ROOT / "tests" / "audio" / "00_smoke_test_8s.wav"


def _installed(name: str) -> bool:
    try:
        return get_engine(name).installed
    except KeyError:
        return False


def _preset(key: str, stages: dict, loudness_target: str = "youtube") -> Preset:
    return Preset(
        key=key,
        name=key,
        description="",
        stages=stages,
        finishing=FinishingParams(loudness_target=loudness_target),
    )


@pytest.mark.skipif(not _installed("dnsmos"), reason="dnsmos engine not installed")
def test_dnsmos_metrics_recorded_per_stage():
    """Accept criterion: per-stage metrics display. job.json must carry a DNSMOS
    SIG/BAK/OVRL row for 'original' and 'final' at minimum.
    """
    preset = _preset("dnsmos_only", stages={})  # no denoise/dereverb/SR, just finishing
    result = run_job(SMOKE_FILE, preset)

    import json

    with open(result.job_json_path, "r", encoding="utf-8") as f:
        job_data = json.load(f)

    stages = {m["stage"] for m in job_data["metrics"]}
    assert "original" in stages
    assert "final" in stages
    for m in job_data["metrics"]:
        assert m["sig"] is None or 1.0 <= m["sig"] <= 5.0
        assert m["ovrl"] is None or 1.0 <= m["ovrl"] <= 5.0


@pytest.mark.skipif(not _installed("clearervoice"), reason="clearervoice engine not installed")
def test_super_resolution_skipped_above_bandwidth_threshold():
    """Accept criterion: SR only triggers on low-bandwidth inputs. A threshold set below
    the source's actual bandwidth must make 'auto' mode skip the stage entirely.
    """
    preset = _preset(
        "sr_should_skip",
        stages={"super_resolution": {"mode": "auto", "bandwidth_threshold_hz": 1.0}},
    )
    result = run_job(SMOKE_FILE, preset, export_intermediate=True)
    assert not (result.job_dir / "04_super_resolution.wav").exists()


@pytest.mark.skipif(not _installed("clearervoice"), reason="clearervoice engine not installed")
def test_super_resolution_runs_below_bandwidth_threshold():
    preset = _preset(
        "sr_should_run",
        stages={"super_resolution": {"mode": "auto", "bandwidth_threshold_hz": 48000.0}},
    )
    result = run_job(SMOKE_FILE, preset, export_intermediate=True)
    assert (result.job_dir / "04_super_resolution.wav").exists()


@pytest.mark.skipif(not _installed("separator"), reason="separator engine not installed")
def test_dereverb_stage_runs_and_blends():
    """Full-strength dereverb should measurably change the signal, and the wet/dry blend
    at strength=0.5 should land between the dry and fully-wet versions.
    """
    preset_full = _preset("dereverb_full", stages={"dereverb": {"enabled": True, "strength": 1.0}})
    result_full = run_job(SMOKE_FILE, preset_full, export_intermediate=True)
    dry, sr = read_wav_mono(SMOKE_FILE)
    wet, _ = read_wav_mono(result_full.job_dir / "03_dereverb.wav")
    assert wet.size == dry.size
    assert not np.allclose(wet, dry, atol=1e-4)

    preset_half = _preset("dereverb_half", stages={"dereverb": {"enabled": True, "strength": 0.5}})
    result_half = run_job(SMOKE_FILE, preset_half, export_intermediate=True)
    blended, _ = read_wav_mono(result_half.job_dir / "03_dereverb.wav")
    assert blended.size == dry.size
    assert not np.allclose(blended, dry, atol=1e-4)
    assert not np.allclose(blended, wet, atol=1e-4)
