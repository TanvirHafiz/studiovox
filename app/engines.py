"""Engine registry: reads engines/<name>/engine.yaml files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config import config


@dataclass
class EngineSpec:
    name: str
    python: Path
    worker: Path
    sample_rates: list[int]
    channels: str
    max_chunk_seconds: int
    needs_gpu: bool
    vram_gb_estimate: float
    license: str
    tasks: list[str]
    raw: dict[str, Any]

    @property
    def not_installed_reason(self) -> str | None:
        """None if installed; otherwise a specific, user-actionable reason why not."""
        if not self.python.exists():
            return f"Python environment missing (expected at {self.python})."
        # Some engines (nvidia_afx) don't get their own env - they wrap an external SDK the
        # user installs separately, so "installed" also depends on a configured path existing.
        requires = self.raw.get("requires_config_path")
        if requires:
            value = getattr(config, requires, None)
            if not value:
                return f"config.yaml's {requires} is not set."
            if not Path(value).exists():
                return f"config.yaml's {requires} ({value}) does not exist."
        return None

    @property
    def installed(self) -> bool:
        return self.not_installed_reason is None


def _load_one(engine_dir: Path) -> EngineSpec | None:
    yaml_path = engine_dir / "engine.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    python = Path(data.get("python", ""))
    if not python.is_absolute():
        python = config.engines_dir.parent / python
    worker = engine_dir / "worker.py"
    return EngineSpec(
        name=data.get("name", engine_dir.name),
        python=python,
        worker=worker,
        sample_rates=data.get("sample_rates", [48000]),
        channels=data.get("channels", "mono"),
        max_chunk_seconds=int(data.get("max_chunk_seconds", 30)),
        needs_gpu=bool(data.get("needs_gpu", False)),
        vram_gb_estimate=float(data.get("vram_gb_estimate", 0)),
        license=data.get("license", "unknown"),
        tasks=data.get("tasks", []),
        raw=data,
    )


def list_engines() -> dict[str, EngineSpec]:
    engines: dict[str, EngineSpec] = {}
    if not config.engines_dir.exists():
        return engines
    for entry in sorted(config.engines_dir.iterdir()):
        if entry.is_dir():
            spec = _load_one(entry)
            if spec:
                engines[spec.name] = spec
    return engines


def get_engine(name: str) -> EngineSpec:
    engines = list_engines()
    if name not in engines:
        raise KeyError(f"Unknown engine '{name}'. Available: {list(engines.keys())}")
    return engines[name]
