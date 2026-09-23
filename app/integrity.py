"""Content integrity check (speech): transcribes audio before and after generative
restoration and compares the text. Catches the failure mode generative models have that
denoise/dereverb/SR do not: inventing or changing words rather than just filtering audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.audio_io import write_wav
from app.engines import get_engine
from app.worker_runner import run_worker

WER_FLAG_THRESHOLD = 0.05  # 5%, per the plan's stated integrity guard


@dataclass
class IntegrityResult:
    ran: bool
    wer: float | None = None
    flagged: bool = False
    reference_text: str = ""
    hypothesis_text: str = ""
    differing_segments: list[dict] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "ran": self.ran,
            "wer": round(self.wer, 4) if self.wer is not None else None,
            "flagged": self.flagged,
            "reference_text": self.reference_text,
            "hypothesis_text": self.hypothesis_text,
            "differing_segments": self.differing_segments,
            "reason": self.reason,
        }


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard word-level WER via Levenshtein edit distance / reference length."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0

    # DP edit distance over word sequences.
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        curr = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(hyp)] / len(ref)


def _find_differing_segments(ref_segments: list[dict], hyp_segments: list[dict], word_overlap_threshold: float = 0.5) -> list[dict]:
    """Pairs each reference segment with the closest-by-time hypothesis segment and flags
    pairs with low word overlap, giving the user timestamps to go listen to.
    """
    differing = []
    for ref_seg in ref_segments:
        ref_mid = (ref_seg["start"] + ref_seg["end"]) / 2
        if not hyp_segments:
            break
        closest = min(hyp_segments, key=lambda h: abs((h["start"] + h["end"]) / 2 - ref_mid))

        ref_words = set(ref_seg["text"].lower().split())
        hyp_words = set(closest["text"].lower().split())
        if not ref_words and not hyp_words:
            continue
        overlap = len(ref_words & hyp_words) / max(1, len(ref_words | hyp_words))

        if overlap < word_overlap_threshold:
            differing.append({
                "start": ref_seg["start"],
                "end": ref_seg["end"],
                "before_text": ref_seg["text"],
                "after_text": closest["text"],
            })
    return differing


def _transcribe(audio, sr, job_dir: Path, label: str, model: str) -> dict | None:
    try:
        engine = get_engine("whisper")
    except KeyError:
        return None
    if not engine.installed:
        return None

    clip_dir = job_dir / "whisper"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / f"{label}.wav"
    write_wav(clip_path, audio, sr, subtype="FLOAT")

    manifest_path = clip_dir / f"{label}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([{"in": str(clip_path)}], f)

    try:
        result = run_worker(engine, task="transcribe", manifest_path=manifest_path, params={"model": model})
        return result["results"][0]
    except Exception:  # noqa: BLE001
        return None
    finally:
        clip_path.unlink(missing_ok=True)


def run_integrity_check(before_audio, after_audio, sr: int, job_dir: Path, model: str = "large-v3") -> IntegrityResult:
    """Transcribes both signals and compares. Never raises: a failure here should not fail
    the whole job, since this is a safety guard, not core pipeline functionality.
    """
    before = _transcribe(before_audio, sr, job_dir, "before", model)
    if before is None:
        return IntegrityResult(ran=False, reason="whisper engine not installed or transcription failed")
    after = _transcribe(after_audio, sr, job_dir, "after", model)
    if after is None:
        return IntegrityResult(ran=False, reason="whisper engine not installed or transcription failed")

    wer = word_error_rate(before["text"], after["text"])
    differing = _find_differing_segments(before["segments"], after["segments"])

    return IntegrityResult(
        ran=True,
        wer=wer,
        flagged=wer > WER_FLAG_THRESHOLD,
        reference_text=before["text"],
        hypothesis_text=after["text"],
        differing_segments=differing,
    )
