"""System health checks: ffmpeg, GPU, disk space. Used by /api/health at startup and on demand."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

from app.config import config


@dataclass
class FfmpegStatus:
    found: bool
    path: str | None = None
    version: str | None = None
    error: str | None = None


@dataclass
class GpuStatus:
    found: bool
    name: str | None = None
    vram_total_mb: int | None = None
    driver_version: str | None = None
    error: str | None = None


@dataclass
class DiskStatus:
    path: str
    free_gb: float
    total_gb: float
    low_space_warning: bool


@dataclass
class HealthReport:
    ffmpeg: FfmpegStatus
    gpu: GpuStatus
    disk: list[DiskStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ffmpeg": vars(self.ffmpeg),
            "gpu": vars(self.gpu),
            "disk": [vars(d) for d in self.disk],
        }


def check_ffmpeg() -> FfmpegStatus:
    exe = shutil.which(config.ffmpeg) or config.ffmpeg
    try:
        result = subprocess.run(
            [exe, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return FfmpegStatus(found=False, error=result.stderr.strip()[:500])
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        return FfmpegStatus(found=True, path=exe, version=first_line)
    except FileNotFoundError:
        return FfmpegStatus(found=False, error=f"ffmpeg not found at '{exe}'. Set paths.ffmpeg in config.yaml.")
    except Exception as e:  # noqa: BLE001
        return FfmpegStatus(found=False, error=str(e))


def check_gpu() -> GpuStatus:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return GpuStatus(found=False, error="nvidia-smi not found on PATH. No NVIDIA GPU detected or driver not installed.")
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return GpuStatus(found=False, error=result.stderr.strip()[:500] or "nvidia-smi returned no output")
        line = result.stdout.strip().splitlines()[0]
        name, mem_total, driver = [x.strip() for x in line.split(",")]
        return GpuStatus(
            found=True,
            name=name,
            vram_total_mb=int(float(mem_total)),
            driver_version=driver,
        )
    except Exception as e:  # noqa: BLE001
        return GpuStatus(found=False, error=str(e))


def check_disk(low_space_gb: float = 10.0) -> list[DiskStatus]:
    statuses = []
    seen_drives = set()
    for path in (config.jobs_dir, config.models_dir):
        path.mkdir(parents=True, exist_ok=True)
        drive = path.anchor
        if drive in seen_drives:
            continue
        seen_drives.add(drive)
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        statuses.append(
            DiskStatus(
                path=str(path),
                free_gb=round(free_gb, 1),
                total_gb=round(total_gb, 1),
                low_space_warning=free_gb < low_space_gb,
            )
        )
    return statuses


def run_health_check() -> HealthReport:
    return HealthReport(
        ffmpeg=check_ffmpeg(),
        gpu=check_gpu(),
        disk=check_disk(),
    )
