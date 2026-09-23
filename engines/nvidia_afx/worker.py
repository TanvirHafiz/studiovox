"""NVIDIA Maxine Audio Effects (AFX) SDK adapter. Shells out to the SDK's effects_demo
sample CLI. Runs in the app's own venv (stdlib only - no SDK bindings needed).

Contract:
  python worker.py --task <generative_restore|dereverb|super_resolution> \
      --chunks-manifest manifest.json --params '{"intensity_ratio": 1.0}'
  manifest.json: [{"in": "path/to/chunk_in.wav", "out": "path/to/chunk_out.wav"}, ...]

The SDK install root comes from the STUDIOVOX_NVIDIA_AFX_SDK_PATH environment variable
(set by app/worker_runner.py from config.yaml's nvidia_afx.sdk_path).

Best-effort: not verified against a real SDK install (proprietary NVIDIA download this
environment does not have). See engine.yaml for what is and isn't confirmed. If this fails
against a real install, the error includes exactly what was searched/tried so the config
format or search patterns can be corrected quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TASK_TO_EFFECT = {
    "generative_restore": "studio_voice_high_quality",
    "dereverb": "dereverb_denoiser",
    "super_resolution": "superres",
}

SAMPLE_RATE = 48000


def _find_sdk_root() -> Path:
    raw = os.environ.get("STUDIOVOX_NVIDIA_AFX_SDK_PATH")
    if not raw:
        raise RuntimeError(
            "NVIDIA AFX SDK path not configured. Set nvidia_afx.sdk_path in config.yaml "
            "to the SDK's install root."
        )
    root = Path(raw)
    if not root.exists():
        raise RuntimeError(f"Configured NVIDIA AFX SDK path does not exist: {root}")
    return root


def _find_file(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def _find_model(root: Path, effect: str) -> Path:
    candidates = [
        p
        for p in root.rglob("*.trtpkg")
        if effect.lower() in p.name.lower() or effect.lower() in str(p.parent).lower()
    ]
    if not candidates:
        raise RuntimeError(
            f"No .trtpkg model found under {root} for effect '{effect}'. "
            f"Searched recursively for '*.trtpkg' with '{effect}' in the path or filename."
        )
    # Prefer a 48k-tagged model since that's this app's internal rate.
    preferred = [p for p in candidates if "48" in p.name]
    return (preferred or candidates)[0]


def _write_config(config_path: Path, effect: str, model_path: Path, in_wav: Path, out_wav: Path, intensity_ratio: float) -> None:
    lines = [
        f"effect {effect}",
        f"sample_rate {SAMPLE_RATE}",
        f"model {model_path}",
        f"input_wav_list {in_wav}",
        f"output_wav_list {out_wav}",
        f"intensity_ratio {intensity_ratio}",
    ]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_TO_EFFECT))
    parser.add_argument("--chunks-manifest", dest="manifest", required=True)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    params = json.loads(args.params)
    intensity_ratio = float(params.get("intensity_ratio", 1.0))
    effect = TASK_TO_EFFECT[args.task]

    with open(args.manifest, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    try:
        sdk_root = _find_sdk_root()
        exe_path = _find_file(sdk_root, "effects_demo.exe")
        if exe_path is None:
            raise RuntimeError(
                f"effects_demo.exe not found anywhere under {sdk_root}. "
                f"Confirm nvidia_afx.sdk_path points at the SDK's install root."
            )
        model_path = _find_model(sdk_root, effect)
    except Exception as e:  # noqa: BLE001
        print(str(e), file=sys.stderr)
        return 1

    total = len(jobs)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, job in enumerate(jobs):
            config_path = Path(tmp_dir) / f"config_{i:04d}.txt"
            in_wav = Path(job["in"])
            out_wav = Path(job["out"])
            _write_config(config_path, effect, model_path, in_wav, out_wav, intensity_ratio)

            result = subprocess.run(
                [str(exe_path), "-c", str(config_path)],
                capture_output=True,
                text=True,
                cwd=str(exe_path.parent),
                timeout=300,
            )
            if result.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
                print(
                    f"effects_demo.exe failed for chunk {i} (effect={effect}, exit={result.returncode}).\n"
                    f"Config written:\n{config_path.read_text(encoding='utf-8')}\n"
                    f"stdout:\n{result.stdout.strip()}\n"
                    f"stderr:\n{result.stderr.strip()}",
                    file=sys.stderr,
                )
                return 1
            print(f"PROGRESS {(i + 1) / total:.4f}", flush=True)

    result_json = {
        "engine": "nvidia_afx",
        "task": args.task,
        "effect": effect,
        "model": str(model_path),
        "chunks_processed": total,
    }
    print(f"RESULT {json.dumps(result_json)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
