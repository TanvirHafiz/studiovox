"""StudioVox FastAPI app server."""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.analysis import analyze
from app.audio_io import read_wav_mono
from app.config import REPO_ROOT, config
from app.engines import list_engines
from app.health import run_health_check
from app.history import job_dir_for, list_jobs, load_job
from app.ingest import decode_to_wav
from app.job_manager import manager
from app.logging_setup import setup_logging
from app.presets import list_presets, sanitize_key, save_preset

logger = setup_logging()

app = FastAPI(title="StudioVox")

WEB_DIR = REPO_ROOT / "web"


@app.middleware("http")
async def no_cache_for_ui(request: Request, call_next):
    """The UI (index.html/app.js/styles.css) changes across phases while this app is under
    active development. A browser that cached an old page against new JS (or vice versa) can
    silently reference DOM elements that no longer exist, breaking features with no visible
    error. Disabling caching for the UI shell removes that whole class of confusing bugs.
    """
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response
UPLOADS_DIR = config.jobs_dir / "_uploads"


@app.get("/api/health")
def health():
    report = run_health_check()
    return report.to_dict()


@app.get("/api/config")
def get_config():
    return {
        "internal_sample_rate": config.internal_sample_rate,
        "internal_channels": config.internal_channels,
        "gpu_require": config.gpu_require,
    }


@app.get("/api/presets")
def presets_endpoint():
    return {key: p.to_dict() for key, p in list_presets().items()}


class SavePresetRequest(BaseModel):
    name: str
    description: str = ""
    stages: dict
    finishing: dict
    key: str | None = None  # omit to derive from name; include to overwrite an existing preset


@app.post("/api/presets")
def save_preset_endpoint(req: SavePresetRequest):
    try:
        key = req.key or sanitize_key(req.name)
        save_preset(key, req.name, req.description, req.stages, req.finishing)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"key": key}


@app.get("/api/engines")
def engines_endpoint():
    return {
        name: {
            "tasks": spec.tasks,
            "installed": spec.installed,
            "not_installed_reason": spec.not_installed_reason,
            "needs_gpu": spec.needs_gpu,
            "vram_gb_estimate": spec.vram_gb_estimate,
            "license": spec.license,
        }
        for name, spec in list_engines().items()
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload").name
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:10]}_{safe_name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest), "filename": safe_name}


class AnalyzeRequest(BaseModel):
    path: str


@app.post("/api/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    src = Path(req.path)
    if not src.exists():
        raise HTTPException(404, f"File not found: {src}")

    tmp_wav = UPLOADS_DIR / f"{uuid.uuid4().hex[:10]}_analyze.wav"
    try:
        ingest_info = decode_to_wav(src, tmp_wav, config.internal_sample_rate)
        x, sr = read_wav_mono(tmp_wav)
        result = analyze(x, sr)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not analyze file: {e}")
    finally:
        tmp_wav.unlink(missing_ok=True)

    suggested_preset = "singing_vocal" if result.mode_guess == "singing" else "clean_voiceover"
    return {
        "analysis": result.to_dict(),
        "suggested_preset": suggested_preset,
        "is_video": ingest_info.is_video,
    }


class CreateJobRequest(BaseModel):
    path: str
    preset: str = "clean_voiceover"


@app.post("/api/jobs")
def create_job(req: CreateJobRequest):
    src = Path(req.path)
    if not src.exists():
        raise HTTPException(404, f"File not found: {src}")
    if req.preset not in list_presets():
        raise HTTPException(400, f"Unknown preset: {req.preset}")
    job_id = manager.start_job(src, req.preset)
    return {"job_id": job_id}


@app.get("/api/jobs")
def jobs_history_endpoint(limit: int = 200):
    """Job history from disk, so past jobs stay visible across server restarts (the
    in-memory JobManager only knows about jobs started in the current process).
    """
    return list_jobs(limit=limit)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    state = manager.get(job_id)
    if state:
        job_data = None
        if state.result:
            with open(state.result.job_json_path, "r", encoding="utf-8") as f:
                job_data = json.load(f)
        return {
            "status": state.status,
            "stage": state.stage,
            "progress": state.progress,
            "error": state.error,
            "job": job_data,
        }

    # Not tracked by this process (e.g. server restarted since it ran) - fall back to disk.
    job_data = load_job(job_id)
    if job_data is None:
        raise HTTPException(404, "Unknown job")
    return {"status": "done", "stage": "export", "progress": 1.0, "error": None, "job": job_data}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    state = manager.get(job_id)
    if not state:
        raise HTTPException(404, "Unknown job")

    async def event_stream():
        yield f"data: {json.dumps({'type': 'progress', 'stage': state.stage, 'progress': state.progress})}\n\n"
        if state.status != "running":
            yield f"data: {json.dumps({'type': state.status, 'error': state.error})}\n\n"
            return
        while True:
            try:
                item = await asyncio.to_thread(state.events.get, True, 1.0)
            except queue.Empty:
                if state.status != "running":
                    yield f"data: {json.dumps({'type': state.status, 'error': state.error})}\n\n"
                    return
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] in ("done", "error"):
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


AUDIO_FILENAMES = {"original": "01_original.wav", "final": "output.wav"}


def _resolve_job_dir(job_id: str) -> Path:
    """Prefers the in-memory result (covers jobs whose folder uses a different naming
    convention historically), falls back to the disk-derived path everything else uses.
    """
    state = manager.get(job_id)
    if state and state.result:
        return state.result.job_dir
    return job_dir_for(job_id)


@app.get("/api/jobs/{job_id}/audio/{which}")
def job_audio(job_id: str, which: str):
    filename = AUDIO_FILENAMES.get(which)
    if not filename:
        raise HTTPException(400, f"Unknown audio variant: {which}")
    file_path = _resolve_job_dir(job_id) / filename
    if not file_path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(str(file_path), media_type="audio/wav")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str, which: str = "final"):
    job_dir = _resolve_job_dir(job_id)
    job_data = load_job(job_id)
    if which == "video" and job_data and job_data.get("output_video"):
        path = Path(job_data["output_video"])
    else:
        path = job_dir / "output.wav"
    if not path.exists():
        raise HTTPException(404, "Output file not found")
    return FileResponse(str(path), filename=path.name)


class CreateBatchRequest(BaseModel):
    paths: list[str]
    preset: str = "clean_voiceover"


@app.post("/api/batch")
def create_batch(req: CreateBatchRequest):
    if not req.paths:
        raise HTTPException(400, "No files provided")
    if req.preset not in list_presets():
        raise HTTPException(400, f"Unknown preset: {req.preset}")
    missing = [p for p in req.paths if not Path(p).exists()]
    if missing:
        raise HTTPException(404, f"File(s) not found: {missing}")
    batch_id = manager.start_batch([Path(p) for p in req.paths], req.preset)
    return {"batch_id": batch_id}


@app.get("/api/batch/{batch_id}")
def batch_status(batch_id: str):
    state = manager.get_batch(batch_id)
    if not state:
        raise HTTPException(404, "Unknown batch")
    return {
        "status": state.status,
        "total": state.total,
        "current_index": state.current_index,
        "current_filename": state.current_filename,
        "stage": state.stage,
        "stage_progress": state.stage_progress,
        "results": [
            {
                "filename": r.input_path.name,
                "status": r.status,
                "job_id": r.job_result.job_id if r.job_result else None,
                "error": r.error,
            }
            for r in state.results
        ],
    }


@app.get("/api/batch/{batch_id}/events")
async def batch_events(batch_id: str):
    state = manager.get_batch(batch_id)
    if not state:
        raise HTTPException(404, "Unknown batch")

    async def event_stream():
        if state.status != "running":
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        while True:
            try:
                item = await asyncio.to_thread(state.events.get, True, 1.0)
            except queue.Empty:
                if state.status != "running":
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] == "done":
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Serve the single-page UI. Mounted after the API routes so /api/* takes priority.
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.on_event("startup")
def on_startup():
    config.ensure_dirs()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    report = run_health_check()
    logger.info("ffmpeg: %s", "OK" if report.ffmpeg.found else f"MISSING ({report.ffmpeg.error})")
    logger.info("gpu: %s", report.gpu.name if report.gpu.found else f"NOT FOUND ({report.gpu.error})")
    for d in report.disk:
        logger.info("disk %s: %.1f GB free of %.1f GB", d.path, d.free_gb, d.total_gb)
    logger.info("StudioVox ready at http://%s:%s", config.host, config.port)
