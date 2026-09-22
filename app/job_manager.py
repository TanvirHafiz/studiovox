"""Runs orchestrator.run_job in a background thread per web job, exposing progress
via a per-job queue that the SSE endpoint drains.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.orchestrator import JobResult, run_job


@dataclass
class JobState:
    status: str = "running"  # running | done | error
    stage: str = "starting"
    progress: float = 0.0
    result: JobResult | None = None
    error: str | None = None
    events: queue.Queue = field(default_factory=queue.Queue)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def start_job(self, input_path: Path, preset_key: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        state = JobState()
        with self._lock:
            self._jobs[job_id] = state

        def on_progress(stage: str, frac: float) -> None:
            state.stage = stage
            state.progress = frac
            state.events.put({"type": "progress", "stage": stage, "progress": frac})

        def worker() -> None:
            try:
                result = run_job(
                    input_path,
                    preset_key,
                    on_progress=on_progress,
                    export_intermediate=True,
                    job_id=job_id,
                )
                state.status = "done"
                state.result = result
                state.events.put({"type": "done", "job_id": job_id})
            except Exception as e:  # noqa: BLE001
                state.status = "error"
                state.error = str(e)
                state.events.put({"type": "error", "error": str(e)})

        threading.Thread(target=worker, daemon=True, name=f"job-{job_id}").start()
        return job_id

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)


manager = JobManager()
