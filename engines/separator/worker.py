"""audio-separator worker: vocal isolation and dereverb. Runs inside engines/separator/env.

Contract:
  python worker.py --task vocal_isolation --chunks-manifest manifest.json --params '{}'
  python worker.py --task dereverb --chunks-manifest manifest.json --params '{}'

manifest.json: [{"in": "path/to/chunk_in.wav", "out": "path/to/chunk_out.wav"}, ...]
The model loads once and processes every chunk in the manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

TASK_CONFIG = {
    # model checkpoint, and the single output stem we want kept
    "vocal_isolation": ("vocals_mel_band_roformer.ckpt", "vocals"),
    "dereverb": ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "noreverb"),
}

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "separator"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIG))
    parser.add_argument("--chunks-manifest", dest="manifest", required=True)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    json.loads(args.params)  # validated for a clear error if malformed

    with open(args.manifest, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    try:
        from audio_separator.separator import Separator
    except Exception as e:  # noqa: BLE001
        print(f"Failed to import audio_separator: {e}", file=sys.stderr)
        return 1

    model_filename, stem_name = TASK_CONFIG[args.task]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        separator = Separator(
            model_file_dir=str(MODELS_DIR),
            output_single_stem=stem_name,
            log_level=40,  # logging.ERROR: keep stdout clean for our PROGRESS/RESULT protocol
        )
        separator.load_model(model_filename=model_filename)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load separator model {model_filename}: {e}", file=sys.stderr)
        return 1

    total = len(jobs)
    for i, job in enumerate(jobs):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                separator.output_dir = tmp_dir
                output_files = separator.separate(job["in"])
                if not output_files:
                    raise RuntimeError("separator produced no output files")
                # output_single_stem restricts to one file; take it regardless of naming.
                shutil.copyfile(output_files[0], job["out"])
        except Exception as e:  # noqa: BLE001
            print(f"Chunk {i} failed ({job['in']}): {e}", file=sys.stderr)
            return 1
        print(f"PROGRESS {(i + 1) / total:.4f}", flush=True)

    result = {
        "engine": "separator",
        "task": args.task,
        "model": model_filename,
        "chunks_processed": total,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    print(f"RESULT {json.dumps(result)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
