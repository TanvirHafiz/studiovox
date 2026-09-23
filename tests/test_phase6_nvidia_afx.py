"""Phase 6 acceptance tests: the NVIDIA AFX adapter is selectable as a Stage 5 (and
Stage 3/4) engine when its SDK path is configured, and the app works normally - clean,
specific errors, not crashes - when it is not. The actual SDK is proprietary and not
available in this environment, so these tests exercise the gating and error-reporting
machinery, not a real effects_demo.exe run.
"""

from __future__ import annotations

from pathlib import Path

from app.config import config
from app.engines import get_engine
from app.presets import list_presets


def test_nvidia_afx_engine_registered_with_expected_tasks():
    spec = get_engine("nvidia_afx")
    assert set(spec.tasks) == {"generative_restore", "dereverb", "super_resolution"}


def test_nvidia_afx_not_installed_when_sdk_path_unset(monkeypatch):
    monkeypatch.setattr(config, "nvidia_afx_sdk_path", None)
    spec = get_engine("nvidia_afx")
    assert spec.installed is False
    assert spec.not_installed_reason is not None
    assert "nvidia_afx_sdk_path" in spec.not_installed_reason


def test_nvidia_afx_not_installed_when_sdk_path_does_not_exist(monkeypatch):
    monkeypatch.setattr(config, "nvidia_afx_sdk_path", "Z:/definitely/not/a/real/path")
    spec = get_engine("nvidia_afx")
    assert spec.installed is False


def test_nvidia_afx_installed_when_sdk_path_exists(tmp_path, monkeypatch):
    fake_sdk = tmp_path / "afx_sdk"
    fake_sdk.mkdir()
    monkeypatch.setattr(config, "nvidia_afx_sdk_path", str(fake_sdk))
    spec = get_engine("nvidia_afx")
    assert spec.installed is True
    assert spec.not_installed_reason is None


def test_other_engines_and_presets_unaffected_by_nvidia_afx_absence(monkeypatch):
    """The app must work normally without the SDK: no default preset requires nvidia_afx,
    and every other engine's installed status is independent of it.
    """
    monkeypatch.setattr(config, "nvidia_afx_sdk_path", None)

    presets = list_presets()
    assert len(presets) > 0
    for preset in presets.values():
        assert preset.generative_restore_engine != "nvidia_afx" or not preset.generative_restore_enabled
        assert preset.dereverb_engine != "nvidia_afx" or not preset.dereverb_enabled
        assert preset.super_resolution_engine != "nvidia_afx" or preset.super_resolution_mode == "off"

    deepfilternet = get_engine("deepfilternet")
    assert deepfilternet.not_installed_reason is None or "nvidia_afx" not in (deepfilternet.not_installed_reason or "")


def test_nvidia_afx_selected_engine_fails_with_specific_actionable_error(monkeypatch, tmp_path):
    """When a preset does select nvidia_afx without the SDK configured, the job must fail
    with a clear, specific reason - not hang, not silently skip, not a generic traceback.
    """
    import app.orchestrator as orch
    from app.dsp.chain import FinishingParams
    from app.presets import Preset
    from app.worker_runner import WorkerError

    monkeypatch.setattr(config, "nvidia_afx_sdk_path", None)

    original_analyze = orch.analyze

    def fake_analyze(x, sr):
        result = original_analyze(x, sr)
        result.mode_guess = "speech"
        return result

    monkeypatch.setattr(orch, "analyze", fake_analyze)

    preset = Preset(
        key="afx_no_sdk",
        name="afx_no_sdk",
        description="",
        stages={"generative_restore": {"enabled": True, "engine": "nvidia_afx"}},
        finishing=FinishingParams(),
    )

    smoke_file = Path(__file__).resolve().parent / "audio" / "00_smoke_test_8s.wav"
    try:
        orch.run_job(smoke_file, preset)
        assert False, "expected run_job to raise when nvidia_afx is selected without the SDK"
    except WorkerError as e:
        assert "nvidia_afx_sdk_path is not set" in str(e)
