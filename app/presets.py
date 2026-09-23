"""Loads presets/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config import config
from app.dsp.chain import FinishingParams


@dataclass
class Preset:
    key: str
    name: str
    description: str
    stages: dict[str, Any]
    finishing: FinishingParams

    @property
    def denoise_enabled(self) -> bool:
        return bool(self.stages.get("denoise", {}).get("enabled", False))

    @property
    def denoise_engine(self) -> str:
        return self.stages.get("denoise", {}).get("engine", "deepfilternet")

    @property
    def denoise_strength(self) -> float:
        return float(self.stages.get("denoise", {}).get("strength", 1.0))

    @property
    def dereverb_enabled(self) -> bool:
        return bool(self.stages.get("dereverb", {}).get("enabled", False))

    @property
    def dereverb_engine(self) -> str:
        return self.stages.get("dereverb", {}).get("engine", "separator")

    @property
    def dereverb_strength(self) -> float:
        """Wet/dry blend: 1.0 = fully dereverbed, 0.0 = original (bypassed)."""
        return float(self.stages.get("dereverb", {}).get("strength", 1.0))

    @property
    def super_resolution_mode(self) -> str:
        """'auto' (only when source bandwidth is low), 'always', or 'off'."""
        return self.stages.get("super_resolution", {}).get("mode", "auto")

    @property
    def super_resolution_engine(self) -> str:
        return self.stages.get("super_resolution", {}).get("engine", "clearervoice")

    @property
    def super_resolution_bandwidth_threshold_hz(self) -> float:
        return float(self.stages.get("super_resolution", {}).get("bandwidth_threshold_hz", 14000))


def _load_one(path: Path) -> Preset:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    finishing_data = data.get("finishing", {})
    finishing = FinishingParams(**{k: v for k, v in finishing_data.items() if k in FinishingParams.__dataclass_fields__})
    return Preset(
        key=path.stem,
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        stages=data.get("stages", {}),
        finishing=finishing,
    )


def list_presets() -> dict[str, Preset]:
    presets: dict[str, Preset] = {}
    if not config.presets_dir.exists():
        return presets
    for path in sorted(config.presets_dir.glob("*.yaml")):
        preset = _load_one(path)
        presets[preset.key] = preset
    return presets


def get_preset(key: str) -> Preset:
    presets = list_presets()
    if key not in presets:
        raise KeyError(f"Unknown preset '{key}'. Available: {list(presets.keys())}")
    return presets[key]
