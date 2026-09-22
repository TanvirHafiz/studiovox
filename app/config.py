"""Loads config.yaml and resolves paths relative to the repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


class Config:
    def __init__(self, data: dict[str, Any]):
        self._data = data

        server = data.get("server", {})
        self.host: str = server.get("host", "127.0.0.1")
        self.port: int = int(server.get("port", 7860))
        self.open_browser: bool = bool(server.get("open_browser", True))

        paths = data.get("paths", {})
        self.ffmpeg: str = paths.get("ffmpeg", "ffmpeg")
        self.ffprobe: str = paths.get("ffprobe", "ffprobe")
        self.models_dir: Path = _resolve(paths.get("models_dir", "models"))
        self.jobs_dir: Path = _resolve(paths.get("jobs_dir", "jobs"))
        self.logs_dir: Path = _resolve(paths.get("logs_dir", "logs"))
        self.presets_dir: Path = _resolve(paths.get("presets_dir", "presets"))
        self.engines_dir: Path = _resolve(paths.get("engines_dir", "engines"))

        gpu = data.get("gpu", {})
        self.gpu_require: bool = bool(gpu.get("require", False))

        audio = data.get("audio", {})
        self.internal_sample_rate: int = int(audio.get("internal_sample_rate", 48000))
        self.internal_channels: str = audio.get("internal_channels", "mono")

        nvidia_afx = data.get("nvidia_afx", {})
        self.nvidia_afx_sdk_path: str | None = nvidia_afx.get("sdk_path")

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.jobs_dir, self.logs_dir, self.presets_dir, self.engines_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> Config:
    config_path = path or (REPO_ROOT / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)


config = load_config()
