"""DNSMOS P.835 quality scoring worker (SIG/BAK/OVRL). CPU-only, tiny models.

Contract (scoring, not audio transform, so entries only need "in"):
  python worker.py --task score --chunks-manifest manifest.json
  manifest.json: [{"in": "path/to/clip.wav"}, ...]

stdout: "PROGRESS <0..1>" per file, final "RESULT <json>" with per-file scores.
Algorithm ported from microsoft/DNS-Challenge DNSMOS/dnsmos_local.py (CC-BY-4.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "dnsmos"


def _audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


def _get_polyfit_val(sig, bak, ovr):
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    return p_sig(sig), p_bak(bak), p_ovr(ovr)


class DNSMOSScorer:
    def __init__(self, primary_model_path: str, p808_model_path: str):
        self.onnx_sess = ort.InferenceSession(primary_model_path, providers=["CPUExecutionProvider"])
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, providers=["CPUExecutionProvider"])

    def score(self, fpath: str) -> dict:
        aud, input_fs = sf.read(fpath)
        if aud.ndim > 1:
            aud = np.mean(aud, axis=1)
        if input_fs != SAMPLING_RATE:
            audio = librosa.resample(y=aud, orig_sr=input_fs, target_sr=SAMPLING_RATE)
        else:
            audio = aud
        audio = audio.astype(np.float32)

        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE

        sig_scores, bak_scores, ovr_scores, p808_scores = [], [], [], []
        for idx in range(max(num_hops, 1)):
            seg = audio[int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(seg) < len_samples:
                continue
            input_features = seg[np.newaxis, :].astype(np.float32)
            p808_features = _audio_melspec(seg[:-160])[np.newaxis, :, :].astype(np.float32)

            p808_mos = self.p808_onnx_sess.run(None, {"input_1": p808_features})[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, {"input_1": input_features})[0][0]
            mos_sig, mos_bak, mos_ovr = _get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw)

            sig_scores.append(mos_sig)
            bak_scores.append(mos_bak)
            ovr_scores.append(mos_ovr)
            p808_scores.append(p808_mos)

        if not sig_scores:
            return {"sig": None, "bak": None, "ovrl": None, "p808_mos": None, "error": "clip too short to score"}

        return {
            "sig": float(np.mean(sig_scores)),
            "bak": float(np.mean(bak_scores)),
            "ovrl": float(np.mean(ovr_scores)),
            "p808_mos": float(np.mean(p808_scores)),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["score"])
    parser.add_argument("--chunks-manifest", dest="manifest", required=True)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    primary_model_path = str(MODELS_DIR / "sig_bak_ovr.onnx")
    p808_model_path = str(MODELS_DIR / "model_v8.onnx")
    if not Path(primary_model_path).exists() or not Path(p808_model_path).exists():
        print(f"DNSMOS models not found under {MODELS_DIR}", file=sys.stderr)
        return 1

    try:
        scorer = DNSMOSScorer(primary_model_path, p808_model_path)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load DNSMOS models: {e}", file=sys.stderr)
        return 1

    total = len(jobs)
    results = []
    for i, job in enumerate(jobs):
        try:
            scores = scorer.score(job["in"])
            scores["in"] = job["in"]
            results.append(scores)
        except Exception as e:  # noqa: BLE001
            print(f"Scoring failed for {job['in']}: {e}", file=sys.stderr)
            return 1
        print(f"PROGRESS {(i + 1) / total:.4f}", flush=True)

    print(f"RESULT {json.dumps({'engine': 'dnsmos', 'task': 'score', 'scores': results})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
