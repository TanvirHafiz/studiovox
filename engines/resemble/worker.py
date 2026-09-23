"""Resemble Enhance generative restoration worker. Runs inside engines/resemble/env.

Contract:
  python worker.py --task generative_restore --chunks-manifest manifest.json --params '{}'
  manifest.json: [{"in": "path/to/chunk_in.wav", "out": "path/to/chunk_out.wav"}, ...]

Optional params: nfe (int, default 32), solver (str, default "midpoint"),
lambd (float, default 0.5), tau (float, default 0.5) - see resemble_enhance docs.

See engine.yaml for the three Windows-specific workarounds baked into this file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# Workaround 2 (see engine.yaml): the bundled model's hparams.yaml embeds
# pathlib.PosixPath objects (saved on Linux); Windows Python cannot construct
# those directly. Must run before resemble_enhance is imported.
pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["generative_restore"])
    parser.add_argument("--chunks-manifest", dest="manifest", required=True)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    params = json.loads(args.params)
    nfe = int(params.get("nfe", 32))
    solver = params.get("solver", "midpoint")
    lambd = float(params.get("lambd", 0.5))
    tau = float(params.get("tau", 0.5))

    with open(args.manifest, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    try:
        import torch
        import torchaudio
        from resemble_enhance.enhancer.inference import enhance
    except Exception as e:  # noqa: BLE001
        print(f"Failed to import resemble_enhance: {e}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    total = len(jobs)
    for i, job in enumerate(jobs):
        try:
            dwav, sr = torchaudio.load(job["in"])
            dwav = dwav.mean(0)
            hwav, out_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)
            torchaudio.save(job["out"], hwav[None], out_sr)
        except Exception as e:  # noqa: BLE001
            print(f"Chunk {i} failed ({job['in']}): {e}", file=sys.stderr)
            return 1
        print(f"PROGRESS {(i + 1) / total:.4f}", flush=True)

    result = {
        "engine": "resemble",
        "task": "generative_restore",
        "chunks_processed": total,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    print(f"RESULT {json.dumps(result)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
