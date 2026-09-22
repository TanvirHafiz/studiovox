"""Splits long audio into overlapping chunks for an engine, and recombines results
with equal-power crossfades so no clicks or level jumps appear at chunk borders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChunkPlan:
    starts: list[int]
    ends: list[int]  # exclusive
    overlap_samples: int


def plan_chunks(total_samples: int, sr: int, max_chunk_seconds: int, overlap_seconds: float = 1.0) -> ChunkPlan:
    max_chunk_samples = max_chunk_seconds * sr
    overlap_samples = int(overlap_seconds * sr)

    if total_samples <= max_chunk_samples:
        return ChunkPlan(starts=[0], ends=[total_samples], overlap_samples=overlap_samples)

    starts: list[int] = []
    ends: list[int] = []
    step = max_chunk_samples - overlap_samples
    if step <= 0:
        step = max_chunk_samples

    pos = 0
    while pos < total_samples:
        end = min(pos + max_chunk_samples, total_samples)
        starts.append(pos)
        ends.append(end)
        if end >= total_samples:
            break
        pos += step

    return ChunkPlan(starts=starts, ends=ends, overlap_samples=overlap_samples)


def split(x: np.ndarray, plan: ChunkPlan) -> list[np.ndarray]:
    return [x[s:e] for s, e in zip(plan.starts, plan.ends)]


def recombine(chunks: list[np.ndarray], plan: ChunkPlan, total_samples: int) -> np.ndarray:
    """Linear crossfade over the overlap region between consecutive chunks.

    Linear (not equal-power) fades are correct here because the overlap region holds the
    same underlying content from two chunks, not two independent signals: the fades sum to
    exactly 1 everywhere, so no level bump appears in the middle of the crossfade.
    """
    if len(chunks) == 1:
        return chunks[0][:total_samples]

    out = np.zeros(total_samples, dtype=np.float64)
    weight = np.zeros(total_samples, dtype=np.float64)

    for i, (chunk, start, end) in enumerate(zip(chunks, plan.starts, plan.ends)):
        length = end - start
        chunk = chunk[:length]
        env = np.ones(length, dtype=np.float64)

        overlap = plan.overlap_samples
        if i > 0 and overlap > 0 and overlap <= length:
            env[:overlap] = np.linspace(0, 1, overlap)
        if i < len(chunks) - 1 and overlap > 0 and overlap <= length:
            env[-overlap:] = np.linspace(1, 0, overlap)

        out[start:end] += chunk.astype(np.float64) * env
        weight[start:end] += env

    weight[weight == 0] = 1.0
    result = out / weight
    return result.astype(np.float32)
