"""CLI: python -m studiovox process in.wav --preset clean_voiceover"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.orchestrator import run_job
from app.presets import list_presets


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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
