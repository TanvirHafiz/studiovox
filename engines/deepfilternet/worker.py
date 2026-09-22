"""DeepFilterNet3 denoise worker. Runs inside engines/deepfilternet/env.

Contract:
  python worker.py --task denoise --in in.wav --out out.wav --params '{}'
  python worker.py --task denoise --chunks-manifest manifest.json --params '{}'

manifest.json: [{"in": "path/to/chunk_in.wav", "out": "path/to/chunk_out.wav"}, ...]
The model loads once and processes every chunk in the manifest, so long files are not
reloading weights per chunk.

stdout: "PROGRESS <0..1>" lines, final "RESULT <json>" line.
Exit code 0 on success, non-zero with an error on stderr otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types


def _patch_torchaudio_backend_shim() -> None:
    """torchaudio >=2.9 removed the load/save/info I/O backend deepfilternet's df.io needs.
    We pin torch/torchaudio to 2.7.1 (last version with it), so this is a no-op safety net
    in case a future upgrade silently reintroduces the missing module path.
    """
    import torchaudio

    if hasattr(torchaudio, "backend"):
        return
    backend_mod = types.ModuleType("torchaudio.backend")
    common_mod = types.ModuleType("torchaudio.backend.common")
    common_mod.AudioMetaData = getattr(torchaudio, "AudioMetaData", object)
    backend_mod.common = common_mod
    sys.modules["torchaudio.backend"] = backend_mod
    sys.modules["torchaudio.backend.common"] = common_mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["denoise"])
    parser.add_argument("--in", dest="in_path", default=None)
    parser.add_argument("--out", dest="out_path", default=None)
    parser.add_argument("--chunks-manifest", dest="manifest", default=None)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    params = json.loads(args.params)
    strength = float(params.get("strength", 1.0))  # 0..1 wet/dry blend against the input

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            jobs = json.load(f)
    elif args.in_path and args.out_path:
        jobs = [{"in": args.in_path, "out": args.out_path}]
    else:
        print("Must pass --in/--out or --chunks-manifest", file=sys.stderr)
        return 2

    try:
        _patch_torchaudio_backend_shim()
        import torch
        from df.enhance import enhance, init_df, load_audio, save_audio
    except Exception as e:  # noqa: BLE001
        print(f"Failed to import DeepFilterNet: {e}", file=sys.stderr)
        return 1

    t0 = time.time()
    try:
        model, df_state, _ = init_df()
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load DeepFilterNet model: {e}", file=sys.stderr)
        return 1

    total = len(jobs)
    for i, job in enumerate(jobs):
        try:
            audio, _ = load_audio(job["in"], sr=df_state.sr())
            enhanced = enhance(model, df_state, audio)
            if strength < 1.0:
                min_len = min(audio.shape[-1], enhanced.shape[-1])
                enhanced = strength * enhanced[..., :min_len] + (1 - strength) * audio[..., :min_len]
            save_audio(job["out"], enhanced, df_state.sr())
        except Exception as e:  # noqa: BLE001
            print(f"Chunk {i} failed ({job['in']}): {e}", file=sys.stderr)
            return 1
        print(f"PROGRESS {(i + 1) / total:.4f}", flush=True)

    result = {
        "engine": "deepfilternet",
        "task": "denoise",
        "chunks_processed": total,
        "elapsed_seconds": round(time.time() - t0, 2),
        "sample_rate": df_state.sr(),
    }
    print(f"RESULT {json.dumps(result)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
