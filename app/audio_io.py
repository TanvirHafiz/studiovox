"""WAV read/write and resampling helpers shared by the orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    return mono, sr


def write_wav(path: Path, x: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype=subtype)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    return soxr.resample(x.astype(np.float32), sr_in, sr_out, quality="HQ").astype(np.float32)
