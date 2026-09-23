"""Phase 7 tests: batch processing, durable job history, and custom presets."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.config import config
from app.engines import get_engine
from app.history import job_dir_for, list_jobs, load_job
from app.orchestrator import run_batch, run_job
from app.presets import get_preset, sanitize_key, save_preset

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_FILE = REPO_ROOT / "tests" / "audio" / "00_smoke_test_8s.wav"


def _deepfilternet_installed() -> bool:
    try:
        return get_engine("deepfilternet").installed
    except KeyError:
        return False


# ---- Custom presets ----


def test_sanitize_key_produces_filesystem_safe_lowercase():
    assert sanitize_key("My Custom Preset!") == "my_custom_preset"
    assert sanitize_key("  spaced  ") == "spaced"


def test_sanitize_key_rejects_names_with_no_usable_characters():
    with pytest.raises(ValueError):
        sanitize_key("!!!")


def test_save_and_load_preset_round_trip():
    stages = {"denoise": {"enabled": True, "engine": "deepfilternet", "strength": 0.8}}
    finishing = {"loudness_target": "podcast", "high_pass_hz": 90}

    key = sanitize_key("Test Round Trip Preset")
    path = save_preset(key, "Test Round Trip Preset", "a test preset", stages, finishing)
    try:
        assert path.exists()
        loaded = get_preset(key)
        assert loaded.name == "Test Round Trip Preset"
        assert loaded.denoise_enabled is True
        assert loaded.denoise_strength == 0.8
        assert loaded.finishing.loudness_target == "podcast"
        assert loaded.finishing.high_pass_hz == 90
    finally:
        path.unlink(missing_ok=True)


def test_save_preset_rejects_invalid_key():
    with pytest.raises(ValueError):
        save_preset("Not A Valid Key!", "name", "desc", {}, {})


# ---- Job history ----


def test_list_jobs_and_load_job_after_run(tmp_path, monkeypatch):
    """History must be readable purely from disk (no dependency on any in-memory state),
    since that's what makes it survive a server restart.
    """
    fake_jobs_dir = tmp_path / "jobs"
    fake_jobs_dir.mkdir()
    monkeypatch.setattr(config, "jobs_dir", fake_jobs_dir)

    result = run_job(SMOKE_FILE, "clean_voiceover", job_id="history_test_job")

    summaries = list_jobs()
    assert any(s["job_id"] == "history_test_job" for s in summaries)

    loaded = load_job("history_test_job")
    assert loaded is not None
    assert loaded["preset"] == "clean_voiceover"
    assert job_dir_for("history_test_job") == result.job_dir


def test_load_job_returns_none_for_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "jobs_dir", tmp_path)
    assert load_job("does_not_exist") is None


def test_list_jobs_ignores_underscore_prefixed_and_malformed_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "jobs_dir", tmp_path)
    (tmp_path / "_uploads").mkdir()
    broken = tmp_path / "broken_job"
    broken.mkdir()
    (broken / "job.json").write_text("{not valid json", encoding="utf-8")

    assert list_jobs() == []


# ---- Batch processing ----


@pytest.mark.skipif(not _deepfilternet_installed(), reason="deepfilternet engine not installed")
def test_run_batch_processes_all_files_and_reports_per_file_status(tmp_path):
    ok_file = tmp_path / "ok.wav"
    shutil.copyfile(SMOKE_FILE, ok_file)
    bad_file = tmp_path / "not_audio.wav"
    bad_file.write_bytes(b"this is not a real wav file")

    progress_calls = []

    def on_progress(index, total, filename, stage, frac):
        progress_calls.append((index, total, filename, stage, frac))

    results = run_batch([ok_file, bad_file], "clean_voiceover", on_progress=on_progress)

    assert len(results) == 2
    by_name = {r.input_path.name: r for r in results}
    assert by_name["ok.wav"].status == "done"
    assert by_name["ok.wav"].job_result is not None
    assert by_name["not_audio.wav"].status == "error"
    assert by_name["not_audio.wav"].error is not None

    assert len(progress_calls) > 0
    assert all(call[1] == 2 for call in progress_calls)  # total is always 2
