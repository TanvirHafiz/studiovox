"""StudioVox FastAPI app server. Phase 0: skeleton, health endpoint, static UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import REPO_ROOT, config
from app.health import run_health_check
from app.logging_setup import setup_logging

logger = setup_logging()

app = FastAPI(title="StudioVox")

WEB_DIR = REPO_ROOT / "web"


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


# Serve the single-page UI. Mounted after the API routes so /api/* takes priority.
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.on_event("startup")
def on_startup():
    config.ensure_dirs()
    report = run_health_check()
    logger.info("ffmpeg: %s", "OK" if report.ffmpeg.found else f"MISSING ({report.ffmpeg.error})")
    logger.info("gpu: %s", report.gpu.name if report.gpu.found else f"NOT FOUND ({report.gpu.error})")
    for d in report.disk:
        logger.info("disk %s: %.1f GB free of %.1f GB", d.path, d.free_gb, d.total_gb)
    logger.info("StudioVox ready at http://%s:%s", config.host, config.port)
