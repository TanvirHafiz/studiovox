"""Stage 0: decode any audio or video input to WAV float32 via ffmpeg."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import config

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class IngestInfo:
    is_video: bool
    original_path: Path
    duration_seconds: float
    source_sample_rate: int
    source_channels: int


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            config.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def decode_to_wav(input_path: Path, out_wav: Path, sample_rate: int, channels: int = 1) -> IngestInfo:
    """Decode audio (or the audio track of a video) to a float32 WAV at the given rate."""
    info = probe(input_path)
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise RuntimeError(f"No audio stream found in {input_path}")
    astream = audio_streams[0]
    duration = float(info.get("format", {}).get("duration", astream.get("duration", 0.0)) or 0.0)
    source_sr = int(astream.get("sample_rate", sample_rate))
    source_ch = int(astream.get("channels", 1))
    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "flt",
        "-c:a",
        "pcm_f32le",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(60, int(duration * 2) + 30))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.strip()[-2000:]}")

    return IngestInfo(
        is_video=is_video,
        original_path=input_path,
        duration_seconds=duration,
        source_sample_rate=source_sr,
        source_channels=source_ch,
    )


def remux_audio_into_video(video_path: Path, new_audio_wav: Path, out_path: Path) -> None:
    """Replace the audio track of a video with new_audio_wav, copying the video stream (no re-encode)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(new_audio_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg remux failed: {result.stderr.strip()[-2000:]}")
