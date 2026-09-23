"""Phase 4 acceptance tests: generative restore blend/alignment and the integrity check
that guards it. Each test skips if the engine it needs isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio_io import read_wav_mono
from app.dsp.blend import detect_comb_filtering
from app.dsp.chain import FinishingParams
from app.engines import get_engine
from app.integrity import run_integrity_check
from app.orchestrator import run_job
from app.presets import Preset

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_FILE = REPO_ROOT / "tests" / "audio" / "00_smoke_test_8s.wav"
INTEGRITY_BEFORE = REPO_ROOT / "tests" / "audio" / "09_integrity_before.wav"
INTEGRITY_AFTER = REPO_ROOT / "tests" / "audio" / "09_integrity_after.wav"


def _installed(name: str) -> bool:
    try:
        return get_engine(name).installed
    except KeyError:
        return False


def _preset(key: str, stages: dict) -> Preset:
    return Preset(key=key, name=key, description="", stages=stages, finishing=FinishingParams())


@pytest.mark.skipif(not _installed("whisper"), reason="whisper engine not installed")
def test_integrity_check_flags_genuinely_different_speech(tmp_path):
    """Section 11's hallucination probe, in miniature: 'before' and 'after' audio saying
    different sentences must be flagged by the integrity check (this is what a generative
    model inventing or changing words would look like to this guard).
    """
    before, sr = read_wav_mono(INTEGRITY_BEFORE)
    after, _ = read_wav_mono(INTEGRITY_AFTER)

    result = run_integrity_check(before, after, sr, tmp_path)

    assert result.ran, result.reason
    assert result.wer is not None and result.wer > 0.05
    assert result.flagged
    assert len(result.differing_segments) > 0


@pytest.mark.skipif(not _installed("whisper"), reason="whisper engine not installed")
def test_integrity_check_passes_identical_speech(tmp_path):
    before, sr = read_wav_mono(INTEGRITY_BEFORE)

    result = run_integrity_check(before, before.copy(), sr, tmp_path)

    assert result.ran, result.reason
    assert result.wer == 0.0
    assert not result.flagged
    assert len(result.differing_segments) == 0


def _force_speech_mode(monkeypatch):
    """The mode_guess heuristic (speech vs singing) can misclassify short synthetic test
    clips; these tests are about Stage 5 wiring, not that heuristic's accuracy, so mode is
    pinned deterministically the same way test_generative_restore_skipped_in_singing_mode
    pins it to 'singing'.
    """
    import app.orchestrator as orch

    original_analyze = orch.analyze

    def fake_analyze(x, sr):
        result = original_analyze(x, sr)
        result.mode_guess = "speech"
        return result

    monkeypatch.setattr(orch, "analyze", fake_analyze)


@pytest.mark.skipif(not _installed("resemble"), reason="resemble engine not installed")
def test_generative_restore_band_blend_produces_no_comb_filtering(monkeypatch):
    """band_blend itself must not introduce comb filtering on top of whatever the faithful
    signal already contains. SR is forced off here (bandwidth_threshold_hz=0) because that
    stage can carry its own periodic artifacts on synthetic test audio, which would confound
    a check aimed specifically at the blend step - the DSP-level mechanism is already covered
    directly, on controlled signals, by test_blend.py's comb-detection tests.
    """
    _force_speech_mode(monkeypatch)
    preset = _preset(
        "resemble_band",
        stages={
            "super_resolution": {"mode": "off"},
            "generative_restore": {"enabled": True, "engine": "resemble", "blend_mode": "band", "wet": 0.5},
        },
    )
    result = run_job(SMOKE_FILE, preset, export_intermediate=True)

    faithful, sr = read_wav_mono(result.job_dir / "01_original.wav")
    out_path = result.job_dir / "05_generative_restore.wav"
    assert out_path.exists()
    blended, _ = read_wav_mono(out_path)

    _, faithful_details = detect_comb_filtering(faithful, sr)
    _, blended_details = detect_comb_filtering(blended, sr)
    faithful_strength = faithful_details.get("periodicity_strength", 0.0)
    blended_strength = blended_details.get("periodicity_strength", 0.0)
    assert blended_strength <= faithful_strength + 0.15, (
        f"Band blend increased spectral periodicity beyond the faithful signal's own: "
        f"{faithful_strength} -> {blended_strength}"
    )


@pytest.mark.skipif(not _installed("resemble"), reason="resemble engine not installed")
@pytest.mark.skipif(not _installed("whisper"), reason="whisper engine not installed")
def test_generative_restore_runs_integrity_check_end_to_end(monkeypatch):
    _force_speech_mode(monkeypatch)
    preset = _preset(
        "resemble_with_integrity",
        stages={"generative_restore": {"enabled": True, "engine": "resemble", "blend_mode": "band", "wet": 0.5}},
    )
    result = run_job(SMOKE_FILE, preset)

    import json

    with open(result.job_json_path, "r", encoding="utf-8") as f:
        job_data = json.load(f)

    assert job_data["integrity"] is not None
    assert job_data["integrity"]["ran"] is True
    assert job_data["integrity"]["wer"] is not None


@pytest.mark.skipif(not _installed("resemble"), reason="resemble engine not installed")
def test_generative_restore_skipped_in_singing_mode(monkeypatch):
    """Stage 5 must be hidden/skipped for singing content per the plan's Stage 5 design,
    even when the preset enables it.
    """
    import app.orchestrator as orch

    original_analyze = orch.analyze

    def fake_analyze(x, sr):
        result = original_analyze(x, sr)
        result.mode_guess = "singing"
        return result

    monkeypatch.setattr(orch, "analyze", fake_analyze)

    preset = _preset(
        "resemble_singing_should_skip",
        stages={"generative_restore": {"enabled": True, "engine": "resemble"}},
    )
    result = run_job(SMOKE_FILE, preset, export_intermediate=True)
    assert not (result.job_dir / "05_generative_restore.wav").exists()
