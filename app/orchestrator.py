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
from app.dsp.chain import run_finishing_chain
from app.dsp.loudness import measure_lufs, true_peak_limiter
from app.engines import get_engine
from app.ingest import decode_to_wav, remux_audio_into_video
from app.logging_setup import job_logger
from app.presets import Preset, get_preset
from app.worker_runner import run_worker


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


def _run_denoise_stage(
    x: np.ndarray,
    sr: int,
    preset: Preset,
    job_dir: Path,
    on_progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    engine = get_engine(preset.denoise_engine)
    engine_sr = engine.sample_rates[0]
    x_engine = resample(x, sr, engine_sr)

    plan = plan_chunks(x_engine.size, engine_sr, engine.max_chunk_seconds)
    chunks_in = split(x_engine, plan)

    chunk_dir = job_dir / "chunks" / preset.denoise_engine
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

    run_worker(
        engine,
        task="denoise",
        manifest_path=manifest_path,
        params={"strength": preset.denoise_strength},
        on_progress=on_progress,
    )

    chunks_out = []
    for item in manifest:
        out_audio, out_sr = read_wav_mono(Path(item["out"]))
        assert out_sr == engine_sr
        chunks_out.append(out_audio)

    recombined = recombine(chunks_out, plan, x_engine.size)
    return resample(recombined, engine_sr, sr)


def run_job(
    input_path: Path,
    preset_key: str,
    on_progress: Callable[[str, float], None] | None = None,
    export_intermediate: bool = False,
    job_id: str | None = None,
) -> JobResult:
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    preset = get_preset(preset_key)
    job_id, job_dir = new_job_dir(input_path, job_id=job_id)
    logger = job_logger(job_id, job_dir)
    logger.info("Starting job for %s with preset '%s'", input_path, preset_key)

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

    # Stage 1: pre-denoise safety limiter. DeepFilterNet (and neural denoisers generally)
    # can produce pathological full-scale oscillating output when fed samples above +-1.0
    # true peak - a real recording defect (hot mic, or overshoot from lossy source codecs
    # like AAC/m4a), not something the 0.1%-clipping-triggered optional declip stage catches
    # since it can be well under that threshold. This runs unconditionally: it is a no-op
    # (no gain change) on audio that never exceeds the ceiling.
    if analysis_result.peak_dbfs > -1.0:
        logger.info("Pre-denoise safety limiting: peak was %.2f dBFS", analysis_result.peak_dbfs)
        current = true_peak_limiter(current, sr, ceiling_dbtp=-1.0)

    # Stage 2: denoise
    if preset.denoise_enabled:
        report("denoise", 0.0)
        current = _run_denoise_stage(
            current, sr, preset, job_dir, on_progress=lambda f: report("denoise", f)
        )
        if export_intermediate:
            write_wav(job_dir / "02_denoise.wav", current, sr, subtype="FLOAT")
        report("denoise", 1.0)

    # Stage 6: finishing chain
    report("finishing", 0.0)
    finished = run_finishing_chain(current, sr, preset.finishing)
    report("finishing", 1.0)

    # Sample-accurate length: pad or trim to match the input exactly.
    if finished.size < x.size:
        finished = np.pad(finished, (0, x.size - finished.size))
    elif finished.size > x.size:
        finished = finished[: x.size]

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
        "preset": preset_key,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": analysis_result.to_dict(),
        "final_lufs": round(final_lufs, 2),
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
