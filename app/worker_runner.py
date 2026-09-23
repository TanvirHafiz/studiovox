"""Runs an engine worker subprocess against a manifest of chunk in/out pairs,
parsing PROGRESS/RESULT lines from stdout.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

from app.config import config
from app.engines import EngineSpec


class WorkerError(RuntimeError):
    pass


def run_worker(
    engine: EngineSpec,
    task: str,
    manifest_path: Path,
    params: dict,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    if not engine.installed:
        raise WorkerError(f"Engine '{engine.name}' is not installed (missing {engine.python}).")

    cmd = [
        str(engine.python),
        str(engine.worker),
        "--task",
        task,
        "--chunks-manifest",
        str(manifest_path),
        "--params",
        json.dumps(params),
    ]

    # Some engine libraries (ClearerVoice observed) download weights into a path relative to
    # the process's working directory rather than an absolute location we control, which would
    # otherwise pollute the repo root. Give every worker a per-engine cwd under models/ so any
    # such relative downloads land in the right place regardless of which engine does it.
    engine_cwd = config.models_dir / engine.name
    engine_cwd.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(engine_cwd),
    )

    result: dict | None = None
    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        stdout_lines.append(line)
        if line.startswith("PROGRESS "):
            try:
                value = float(line.split(" ", 1)[1])
                if on_progress:
                    on_progress(value)
            except (IndexError, ValueError):
                pass
        elif line.startswith("RESULT "):
            try:
                result = json.loads(line.split(" ", 1)[1])
            except (IndexError, json.JSONDecodeError):
                pass

    stderr = proc.stderr.read() if proc.stderr else ""
    return_code = proc.wait()

    if return_code != 0 or result is None:
        raise WorkerError(
            f"Engine '{engine.name}' failed (exit {return_code}).\n"
            f"stderr:\n{stderr.strip()}\n"
            f"stdout tail:\n" + "\n".join(stdout_lines[-20:])
        )

    return result
