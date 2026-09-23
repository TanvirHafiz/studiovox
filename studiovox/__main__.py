"""CLI: python -m studiovox process in.wav --preset clean_voiceover"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.orchestrator import run_batch, run_job
from app.presets import list_presets

AUDIO_VIDEO_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wma",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
}


def cmd_process(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    def on_progress(stage: str, frac: float) -> None:
        print(f"[{stage}] {frac * 100:5.1f}%")

    try:
        result = run_job(
            input_path,
            args.preset,
            on_progress=on_progress,
            export_intermediate=args.export_intermediate,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Job failed: {e}", file=sys.stderr)
        return 1

    print(f"\nDone. Job: {result.job_id}")
    print(f"Output WAV: {result.output_wav}")
    if result.output_video:
        print(f"Output video: {result.output_video}")
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    for key, preset in list_presets().items():
        print(f"{key}: {preset.name} - {preset.description}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_VIDEO_EXTENSIONS)
    if not files:
        print(f"No audio/video files found in {folder}", file=sys.stderr)
        return 1

    print(f"Found {len(files)} file(s). Processing with preset '{args.preset}'...\n")

    def on_progress(index: int, total: int, filename: str, stage: str, frac: float) -> None:
        print(f"[{index + 1}/{total}] {filename} - {stage} {frac * 100:5.1f}%")

    results = run_batch(files, args.preset, on_progress=on_progress)

    print("\n--- Batch summary ---")
    ok = 0
    for r in results:
        if r.status == "done":
            ok += 1
            print(f"  OK    {r.input_path.name} -> {r.job_result.output_wav}")
        else:
            print(f"  ERROR {r.input_path.name}: {r.error}")
    print(f"\n{ok}/{len(results)} succeeded.")

    return 0 if ok == len(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="studiovox")
    sub = parser.add_subparsers(dest="command", required=True)

    p_process = sub.add_parser("process", help="Process an audio or video file")
    p_process.add_argument("input", help="Path to the input file")
    p_process.add_argument("--preset", default="clean_voiceover", help="Preset key")
    p_process.add_argument(
        "--export-intermediate", action="store_true", help="Export every intermediate stage WAV"
    )
    p_process.set_defaults(func=cmd_process)

    p_presets = sub.add_parser("presets", help="List available presets")
    p_presets.set_defaults(func=cmd_presets)

    p_batch = sub.add_parser("batch", help="Process every audio/video file in a folder")
    p_batch.add_argument("folder", help="Path to the folder of input files")
    p_batch.add_argument("--preset", default="clean_voiceover", help="Preset key")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
