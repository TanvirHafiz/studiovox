"""faster-whisper transcription worker, used for the content integrity check
(Stage 5 hallucination guard). Runs inside engines/whisper/env.

Contract (transcription, not audio transform, so entries only need "in"):
  python worker.py --task transcribe --chunks-manifest manifest.json --params '{"model": "large-v3"}'
  manifest.json: [{"in": "path/to/clip.wav"}, ...]

stdout: "PROGRESS <0..1>" per file, final "RESULT <json>" with per-file segments.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["transcribe"])
    parser.add_argument("--chunks-manifest", dest="manifest", required=True)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    params = json.loads(args.params)
    model_name = params.get("model", "large-v3")

    with open(args.manifest, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # noqa: BLE001
        print(f"Failed to import faster_whisper: {e}", file=sys.stderr)
        return 1

    try:
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
        except Exception:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load whisper model {model_name}: {e}", file=sys.stderr)
        return 1

    total = len(jobs)
    results = []
    for i, job in enumerate(jobs):
        try:
            segments, info = model.transcribe(job["in"], beam_size=5)
            seg_list = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segments
            ]
            results.append({
                "in": job["in"],
                "language": info.language,
                "text": " ".join(s["text"] for s in seg_list).strip(),
                "segments": seg_list,
            })
        except Exception as e:  # noqa: BLE001
            print(f"Transcription failed for {job['in']}: {e}", file=sys.stderr)
            return 1
        print(f"PROGRESS {(i + 1) / total:.4f}", flush=True)

    print(f"RESULT {json.dumps({'engine': 'whisper', 'task': 'transcribe', 'results': results})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
