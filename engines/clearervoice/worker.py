"""ClearerVoice-Studio worker: MossFormer2_SE_48K (denoise) and MossFormer2_SR_48K
(super resolution). Runs inside engines/clearervoice/env.

Contract:
  python worker.py --task denoise --chunks-manifest manifest.json --params '{}'
  python worker.py --task super_resolution --chunks-manifest manifest.json --params '{}'

manifest.json: [{"in": "path/to/chunk_in.wav", "out": "path/to/chunk_out.wav"}, ...]
The model loads once and processes every chunk in the manifest.

stdout: "PROGRESS <0..1>" lines, final "RESULT <json>" line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

TASK_TO_MODEL = {
    "denoise": ("speech_enhancement", "MossFormer2_SE_48K"),
    "super_resolution": ("speech_super_resolution", "MossFormer2_SR_48K"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_TO_MODEL))
    parser.add_argument("--chunks-manifest", dest="manifest", required=True)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    json.loads(args.params)  # no per-call params used yet; validated for a clear error if malformed

    with open(args.manifest, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    try:
        from clearvoice import ClearVoice
    except Exception as e:  # noqa: BLE001
        print(f"Failed to import ClearVoice: {e}", file=sys.stderr)
        return 1

    task_name, model_name = TASK_TO_MODEL[args.task]

    t0 = time.time()
    try:
        cv = ClearVoice(task=task_name, model_names=[model_name])
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load ClearerVoice model {model_name}: {e}", file=sys.stderr)
        return 1

    total = len(jobs)
    for i, job in enumerate(jobs):
        try:
            output_wav = cv(input_path=job["in"], online_write=False)
            cv.write(output_wav, output_path=job["out"])
        except Exception as e:  # noqa: BLE001
            print(f"Chunk {i} failed ({job['in']}): {e}", file=sys.stderr)
            return 1
        print(f"PROGRESS {(i + 1) / total:.4f}", flush=True)

    result = {
        "engine": "clearervoice",
        "task": args.task,
        "model": model_name,
        "chunks_processed": total,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    print(f"RESULT {json.dumps(result)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
