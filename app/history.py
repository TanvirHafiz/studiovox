"""Job history: reads job.json files directly from disk, so past jobs stay visible and
reopenable across server restarts (the in-memory JobManager only tracks the current process's
lifetime).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import config


def job_dir_for(job_id: str) -> Path:
    return config.jobs_dir / job_id


def load_job(job_id: str) -> dict | None:
    path = job_dir_for(job_id) / "job.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_jobs(limit: int = 200) -> list[dict]:
    if not config.jobs_dir.exists():
        return []
    summaries = []
    for entry in config.jobs_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        job_json = entry / "job.json"
        if not job_json.exists():
            continue
        try:
            with open(job_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append({
            "job_id": data.get("job_id", entry.name),
            "input_filename": Path(data.get("input_path", entry.name)).name,
            "preset": data.get("preset"),
            "created_utc": data.get("created_utc"),
            "final_lufs": data.get("final_lufs"),
            "is_video": data.get("is_video", False),
            "integrity_flagged": bool((data.get("integrity") or {}).get("flagged")),
        })
    summaries.sort(key=lambda s: s.get("created_utc") or "", reverse=True)
    return summaries[:limit]
