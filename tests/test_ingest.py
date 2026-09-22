"""Regression tests for is_video detection: file extension alone is not reliable."""

import subprocess
from pathlib import Path

import pytest

from app.ingest import decode_to_wav

FFMPEG_AVAILABLE = subprocess.run(
    ["ffmpeg", "-version"], capture_output=True
).returncode == 0

pytestmark = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not available")


def _make(tmp_path: Path, args: list[str], name: str) -> Path:
    out = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", *args, str(out)],
        capture_output=True,
        check=True,
    )
    return out


def test_audio_only_mp4_is_not_flagged_as_video(tmp_path):
    """Recorders (e.g. WhatsApp voice notes) sometimes export audio-only content in an
    .mp4 container. is_video must reflect an actual video stream, not the extension.
    """
    src = _make(
        tmp_path,
        ["-f", "lavfi", "-i", "sine=frequency=250:duration=1:sample_rate=48000", "-c:a", "aac"],
        "audio_only.mp4",
    )
    out_wav = tmp_path / "out.wav"
    info = decode_to_wav(src, out_wav, 48000)
    assert info.is_video is False


def test_mp3_with_embedded_cover_art_is_not_flagged_as_video(tmp_path):
    """An attached picture is reported by ffprobe as a video stream; it must not
    trigger the video remux path.
    """
    cover_src = _make(
        tmp_path,
        [
            "-f", "lavfi", "-i", "sine=frequency=300:duration=1:sample_rate=48000",
            "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=1",
            "-map", "0:a", "-map", "1:v",
            "-c:a", "mp3", "-c:v", "mjpeg", "-disposition:v", "attached_pic",
        ],
        "with_art.mp3",
    )
    out_wav = tmp_path / "out.wav"
    info = decode_to_wav(cover_src, out_wav, 48000)
    assert info.is_video is False


def test_real_video_is_flagged_as_video(tmp_path):
    src = _make(
        tmp_path,
        [
            "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=1:r=10",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=1",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
        ],
        "real_video.mp4",
    )
    out_wav = tmp_path / "out.wav"
    info = decode_to_wav(src, out_wav, 48000)
    assert info.is_video is True
