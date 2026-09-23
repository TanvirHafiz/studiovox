"""Ties together ingest, analysis, chunking, engine workers, and the finishing chain
into one job. This is what both the CLI and (later) the web UI call.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from app.analysis import analyze
from app.audio_io import read_wav_mono, resample, write_wav
from app.chunking import plan_chunks, recombine, split
from app.config import config
from app.dsp.blend import align_to, band_blend, detect_comb_filtering, time_blend
from app.dsp.chain import run_finishing_chain
from app.dsp.loudness import measure_lufs, true_peak_limiter
from app.engines import get_engine
from app.ingest import decode_to_wav, remux_audio_into_video
from app.integrity import run_integrity_check
from app.logging_setup import job_logger
from app.presets import Preset, get_preset
from app.worker_runner import run_worker

OVRL_DROP_WARNING_THRESHOLD = 0.2  # flag a stage that lowers DNSMOS OVRL by more than this


@dataclass
class JobResult:
    job_id: str
    job_dir: Path
    output_wav: Path
    output_video: Path | None
    job_json_path: Path


def new_job_dir(input_path: Path, job_id: str | None = None) -> tuple[str, Path]:
    if job_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"{ts}_{input_path.stem}_{uuid.uuid4().hex[:6]}"
    job_dir = config.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_id, job_dir


def _match_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim to an exact sample count. Rate conversion (resample) can drift by a
    handful of samples, and engine stages must not silently shift the pipeline's length.
    """
    if x.size == target_len:
        return x
    if x.size < target_len:
        return np.pad(x, (0, target_len - x.size))
    return x[:target_len]


def _run_engine_stage(
    x: np.ndarray,
    sr: int,
    engine_name: str,
    task: str,
    params: dict,
    job_dir: Path,
    stage_label: str,
    on_progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Runs one engine (denoise, dereverb, or super-resolution) via its worker subprocess,
    chunking as needed and resampling to/from the engine's native rate.
    """
    engine = get_engine(engine_name)
    engine_sr = engine.sample_rates[0]
    x_engine = resample(x, sr, engine_sr)

    plan = plan_chunks(x_engine.size, engine_sr, engine.max_chunk_seconds)
    chunks_in = split(x_engine, plan)

    chunk_dir = job_dir / "chunks" / f"{stage_label}_{engine_name}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, chunk in enumerate(chunks_in):
        in_path = chunk_dir / f"chunk_{i:04d}_in.wav"
        out_path = chunk_dir / f"chunk_{i:04d}_out.wav"
        write_wav(in_path, chunk, engine_sr, subtype="FLOAT")
        manifest.append({"in": str(in_path), "out": str(out_path)})

    manifest_path = chunk_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    run_worker(engine, task=task, manifest_path=manifest_path, params=params, on_progress=on_progress)

    # Some model architectures (MDXC in particular) pad internally to a fixed window multiple,
    # so a chunk can come back a handful of samples longer or shorter than it went in. recombine()
    # assumes each chunk matches its planned length exactly, so that is enforced here per chunk
    # rather than only on the final concatenated result.
    chunks_out = []
    for item, expected_len in zip(manifest, (e - s for s, e in zip(plan.starts, plan.ends))):
        out_audio, out_sr = read_wav_mono(Path(item["out"]))
        assert out_sr == engine_sr
        chunks_out.append(_match_length(out_audio, expected_len))

    recombined = recombine(chunks_out, plan, x_engine.size)
    result = resample(recombined, engine_sr, sr)
    return _match_length(result, x.size)


def _score_dnsmos(x: np.ndarray, sr: int, job_dir: Path, label: str) -> dict | None:
    """Runs the DNSMOS engine on a stage's audio. Returns None (and logs a warning) if the
    engine isn't installed, since DNSMOS is a diagnostic, not a required part of the pipeline.
    """
    try:
        engine = get_engine("dnsmos")
    except KeyError:
        return None
    if not engine.installed:
        return None

    score_dir = job_dir / "dnsmos"
    score_dir.mkdir(parents=True, exist_ok=True)
    clip_path = score_dir / f"{label}.wav"
    write_wav(clip_path, x, sr, subtype="FLOAT")

    manifest_path = score_dir / f"{label}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([{"in": str(clip_path)}], f)

    try:
        result = run_worker(engine, task="score", manifest_path=manifest_path, params={})
        scores = result["scores"][0]
    except Exception:  # noqa: BLE001
        return None
    finally:
        clip_path.unlink(missing_ok=True)

    return {
        "stage": label,
        "sig": round(scores["sig"], 3) if scores.get("sig") is not None else None,
        "bak": round(scores["bak"], 3) if scores.get("bak") is not None else None,
        "ovrl": round(scores["ovrl"], 3) if scores.get("ovrl") is not None else None,
    }


def run_job(
    input_path: Path,
    preset_key: str | Preset,
    on_progress: Callable[[str, float], None] | None = None,
    export_intermediate: bool = False,
    job_id: str | None = None,
) -> JobResult:
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    preset = get_preset(preset_key) if isinstance(preset_key, str) else preset_key
    job_id, job_dir = new_job_dir(input_path, job_id=job_id)
    logger = job_logger(job_id, job_dir)
    logger.info("Starting job for %s with preset '%s'", input_path, preset.key)

    sr = config.internal_sample_rate

    def report(stage: str, frac: float) -> None:
        logger.info("[%s] %.0f%%", stage, frac * 100)
        if on_progress:
            on_progress(stage, frac)

    # Stage 0: ingest
    report("ingest", 0.0)
    raw_wav = job_dir / "00_ingest.wav"
    ingest_info = decode_to_wav(input_path, raw_wav, sr)
    x, actual_sr = read_wav_mono(raw_wav)
    assert actual_sr == sr
    report("ingest", 1.0)

    # Stage 0b: analysis
    analysis_result = analyze(x, sr)
    logger.info("Analysis: %s", analysis_result.to_dict())
    for w in analysis_result.warnings:
        logger.warning(w)

    current = x
    if export_intermediate:
        # FLOAT, not the default PCM_24: intermediate exports are for diagnosis, and PCM_24
        # silently hard-clips any sample above +-1.0, which would hide the exact bug a
        # true-peak-over source (lossy-codec overshoot, hot mic) is here to help debug.
        write_wav(job_dir / "01_original.wav", current, sr, subtype="FLOAT")

    metrics: list[dict] = []
    score = _score_dnsmos(current, sr, job_dir, "original")
    if score:
        metrics.append(score)

    def score_stage(label: str, audio: np.ndarray) -> None:
        s = _score_dnsmos(audio, sr, job_dir, label)
        if not s:
            return
        if metrics and metrics[-1]["ovrl"] is not None and s["ovrl"] is not None:
            drop = metrics[-1]["ovrl"] - s["ovrl"]
            if drop > OVRL_DROP_WARNING_THRESHOLD:
                logger.warning(
                    "Stage '%s' lowered DNSMOS OVRL by %.2f (%.2f -> %.2f)",
                    label, drop, metrics[-1]["ovrl"], s["ovrl"],
                )
        metrics.append(s)

    # Stage 1: pre-denoise safety limiter. DeepFilterNet (and neural denoisers generally)
    # can produce pathological full-scale oscillating output when fed samples above +-1.0
    # true peak - a real recording defect (hot mic, or overshoot from lossy source codecs
    # like AAC/m4a), not something the 0.1%-clipping-triggered optional declip stage catches
    # since it can be well under that threshold. This runs unconditionally: it is a no-op
    # (no gain change) on audio that never exceeds the ceiling.
    if analysis_result.peak_dbfs > -1.0:
        logger.info("Pre-denoise safety limiting: peak was %.2f dBFS", analysis_result.peak_dbfs)
        current = true_peak_limiter(current, sr, ceiling_dbtp=-1.0)

    # Stage 2: denoise (or vocal isolation, for the singing preset - same slot, different task)
    if preset.denoise_enabled:
        report("denoise", 0.0)
        current = _run_engine_stage(
            current, sr, preset.denoise_engine, preset.denoise_task,
            {"strength": preset.denoise_strength}, job_dir, "denoise",
            on_progress=lambda f: report("denoise", f),
        )
        if export_intermediate:
            write_wav(job_dir / "02_denoise.wav", current, sr, subtype="FLOAT")
        report("denoise", 1.0)
        score_stage("denoise", current)

    # Stage 3: dereverb (optional, wet/dry blended)
    if preset.dereverb_enabled:
        report("dereverb", 0.0)
        dry = current
        wet = _run_engine_stage(
            current, sr, preset.dereverb_engine, "dereverb",
            {}, job_dir, "dereverb",
            on_progress=lambda f: report("dereverb", f),
        )
        strength = preset.dereverb_strength
        current = (strength * wet + (1 - strength) * dry).astype(np.float32) if strength < 1.0 else wet
        if export_intermediate:
            write_wav(job_dir / "03_dereverb.wav", current, sr, subtype="FLOAT")
        report("dereverb", 1.0)
        score_stage("dereverb", current)

    # Stage 4: super-resolution. 'auto' only runs it when the source's effective bandwidth
    # (measured in Stage 0 analysis) is below the threshold; forcing it on already-full-band
    # audio wastes GPU time for no audible gain.
    sr_mode = preset.super_resolution_mode
    run_sr = sr_mode == "always" or (
        sr_mode == "auto" and analysis_result.bandwidth_hz < preset.super_resolution_bandwidth_threshold_hz
    )
    if run_sr:
        report("super_resolution", 0.0)
        logger.info(
            "Running super-resolution: bandwidth was %.0f Hz (threshold %.0f Hz, mode '%s')",
            analysis_result.bandwidth_hz, preset.super_resolution_bandwidth_threshold_hz, sr_mode,
        )
        current = _run_engine_stage(
            current, sr, preset.super_resolution_engine, "super_resolution",
            {}, job_dir, "super_resolution",
            on_progress=lambda f: report("super_resolution", f),
        )
        if export_intermediate:
            write_wav(job_dir / "04_super_resolution.wav", current, sr, subtype="FLOAT")
        report("super_resolution", 1.0)
        score_stage("super_resolution", current)

    # Stage 5: generative restore (optional, off by default except the Rescue preset).
    # Speech mode only - a generative speech model is not meaningful on singing.
    integrity_result = None
    if preset.generative_restore_enabled and analysis_result.mode_guess != "singing":
        report("generative_restore", 0.0)
        faithful = current
        generative_raw = _run_engine_stage(
            current, sr, preset.generative_restore_engine, "generative_restore",
            {}, job_dir, "generative_restore",
            on_progress=lambda f: report("generative_restore", f),
        )
        aligned, lag = align_to(faithful, generative_raw, sr)
        logger.info("Generative restore alignment: shifted by %d samples (%.1f ms)", lag, 1000 * lag / sr)

        blend_mode = preset.generative_restore_blend_mode
        if blend_mode == "band":
            current = band_blend(
                faithful, aligned, sr,
                crossover_hz=preset.generative_restore_crossover_hz,
                wet=preset.generative_restore_wet,
            )
        else:
            current = time_blend(faithful, aligned, wet=preset.generative_restore_wet)
            comb_flagged, comb_details = detect_comb_filtering(current, sr)
            if comb_flagged:
                logger.warning("Comb filtering detected in time blend: %s", comb_details)

        if export_intermediate:
            write_wav(job_dir / "05_generative_restore.wav", current, sr, subtype="FLOAT")
        report("generative_restore", 1.0)
        score_stage("generative_restore", current)

        # Integrity check: did the generative model invent or change words? Compares the
        # faithful pre-restore audio against the blended result, since that is what the
        # listener will actually hear.
        report("integrity_check", 0.0)
        integrity_result = run_integrity_check(faithful, current, sr, job_dir)
        if integrity_result.ran and integrity_result.flagged:
            logger.warning(
                "Content integrity check flagged this job: WER %.1f%% (%d differing segment(s))",
                integrity_result.wer * 100, len(integrity_result.differing_segments),
            )
        report("integrity_check", 1.0)

    # Stage 6: finishing chain
    report("finishing", 0.0)
    finished = run_finishing_chain(current, sr, preset.finishing)
    report("finishing", 1.0)

    # Sample-accurate length: pad or trim to match the input exactly.
    if finished.size < x.size:
        finished = np.pad(finished, (0, x.size - finished.size))
    elif finished.size > x.size:
        finished = finished[: x.size]

    score_stage("final", finished)

    # Stage 7: export
    report("export", 0.0)
    output_wav = job_dir / "output.wav"
    write_wav(output_wav, finished, sr, subtype="PCM_24")

    output_video = None
    if ingest_info.is_video:
        output_video = job_dir / f"output{input_path.suffix}"
        remux_audio_into_video(input_path, output_wav, output_video)

    final_lufs = measure_lufs(finished, sr)
    if not np.isfinite(final_lufs):
        final_lufs = -70.0

    job_json_path = job_dir / "job.json"
    job_data = {
        "job_id": job_id,
        "input_path": str(input_path),
        "preset": preset.key,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": analysis_result.to_dict(),
        "final_lufs": round(final_lufs, 2),
        "metrics": metrics,
        "integrity": integrity_result.to_dict() if integrity_result else None,
        "is_video": ingest_info.is_video,
        "output_wav": str(output_wav),
        "output_video": str(output_video) if output_video else None,
    }
    with open(job_json_path, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2)

    report("export", 1.0)
    logger.info("Job complete: %s", output_wav)

    return JobResult(
        job_id=job_id,
        job_dir=job_dir,
        output_wav=output_wav,
        output_video=output_video,
        job_json_path=job_json_path,
    )
