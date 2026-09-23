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
from app.health import run_health_check
from app.ingest import decode_to_wav
from app.job_manager import manager
from app.logging_setup import setup_logging
from app.presets import list_presets

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
    return {
        key: {"name": p.name, "description": p.description}
        for key, p in list_presets().items()
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


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    state = manager.get(job_id)
    if not state:
        raise HTTPException(404, "Unknown job")

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


@app.get("/api/jobs/{job_id}/audio/{which}")
def job_audio(job_id: str, which: str):
    state = manager.get(job_id)
    if not state or not state.result:
        raise HTTPException(404, "Job not ready")
    filename = AUDIO_FILENAMES.get(which)
    if not filename:
        raise HTTPException(400, f"Unknown audio variant: {which}")
    file_path = state.result.job_dir / filename
    if not file_path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(str(file_path), media_type="audio/wav")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str, which: str = "final"):
    state = manager.get(job_id)
    if not state or not state.result:
        raise HTTPException(404, "Job not ready")
    if which == "video" and state.result.output_video:
        path = state.result.output_video
    else:
        path = state.result.output_wav
    return FileResponse(str(path), filename=path.name)


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
