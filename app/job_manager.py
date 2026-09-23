"""Runs orchestrator.run_job (and run_batch) in a background thread per web job/batch,
exposing progress via a per-job queue that the SSE endpoint drains.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.orchestrator import BatchItemResult, JobResult, run_batch, run_job


@dataclass
class JobState:
    status: str = "running"  # running | done | error
    stage: str = "starting"
    progress: float = 0.0
    result: JobResult | None = None
    error: str | None = None
    events: queue.Queue = field(default_factory=queue.Queue)


@dataclass
class BatchState:
    status: str = "running"  # running | done
    total: int = 0
    current_index: int = 0
    current_filename: str = ""
    stage: str = "starting"
    stage_progress: float = 0.0
    results: list[BatchItemResult] = field(default_factory=list)
    events: queue.Queue = field(default_factory=queue.Queue)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._batches: dict[str, BatchState] = {}
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

    def start_batch(self, input_paths: list[Path], preset_key: str) -> str:
        batch_id = uuid.uuid4().hex[:12]
        state = BatchState(total=len(input_paths))
        with self._lock:
            self._batches[batch_id] = state

        def on_progress(index: int, total: int, filename: str, stage: str, frac: float) -> None:
            state.current_index = index
            state.current_filename = filename
            state.stage = stage
            state.stage_progress = frac
            state.events.put({
                "type": "progress",
                "index": index,
                "total": total,
                "filename": filename,
                "stage": stage,
                "progress": frac,
            })

        def worker() -> None:
            results = run_batch(input_paths, preset_key, on_progress=on_progress)
            state.results = results
            state.status = "done"
            state.events.put({
                "type": "done",
                "summary": [
                    {
                        "filename": r.input_path.name,
                        "status": r.status,
                        "job_id": r.job_result.job_id if r.job_result else None,
                        "error": r.error,
                    }
                    for r in results
                ],
            })

        threading.Thread(target=worker, daemon=True, name=f"batch-{batch_id}").start()
        return batch_id

    def get_batch(self, batch_id: str) -> BatchState | None:
        with self._lock:
            return self._batches.get(batch_id)


manager = JobManager()
