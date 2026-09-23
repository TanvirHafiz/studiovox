# StudioVox

A local, offline vocal enhancement studio for Windows. Feed it a recorded voice file
(speech, singing, or a video containing one) and it chains several state-of-the-art AI
models plus classic DSP finishing to produce studio-quality output, entirely on your own
GPU, with no cloud calls and no per-use cost.

Processing is offline (not real-time), so every stage can afford to use large models with
long look-ahead. Everything runs from a local web UI or the command line.

## Example: before / after

A real run from this app, no cherry-picking — a phone-quality voice recording with
background noise, processed with the **Heavy Voice** preset (denoise, then a deep
low-shelf bass boost for a heavier tone).

- **Before** (original recording, noisy, phone-quality bandwidth): [`examples/example_before.mp3`](examples/example_before.mp3)
- **After** (Heavy Voice: denoised, then bass/depth added): [`examples/example_after_heavy_voice.mp3`](examples/example_after_heavy_voice.mp3)

Click either link to open GitHub's file viewer, which plays audio files inline.

Measured with DNSMOS (an objective, non-intrusive speech quality metric): the background
noise score (BAK, 1–5 scale) went from **2.32 → 4.05** on this clip, and stayed there
through every later stage, including the bass boost.

## What it does

- **Denoise**: DeepFilterNet3 (fast, faithful) or ClearerVoice MossFormer2 (stronger, for
  tough noise)
- **Dereverb**: audio-separator's MelBand Roformer De-Reverb model
- **Super-resolution**: MossFormer2 SR, auto-triggered only when the source's effective
  bandwidth is actually low (a phone recording), never wastes GPU time on already-full-band
  audio
- **Generative restoration** (optional): Resemble Enhance or an NVIDIA AFX SDK adapter,
  blended (never used at full strength) and gated by an automatic content-integrity check
  (transcribes before/after with faster-whisper and flags the job if words changed)
- **Singing mode**: vocal isolation instead of a speech denoiser, so vibrato and sustained
  notes survive
- **Finishing chain**: high-pass, corrective EQ (including a low-shelf for depth/weight),
  de-esser, compressor, breath reduction, loudness normalization to a target (podcast,
  YouTube, broadcast, or an unnormalized mixing stem)
- **Quality guards**: per-stage DNSMOS scoring, level-matched A/B comparison (so a louder
  version never just "sounds better" by being louder), sample-accurate output length,
  video remux with the enhanced audio track
- **Batch queue**: process a whole folder with one preset, from the CLI or the web UI
- **Presets**: Clean Voiceover, Podcast, Rescue (bad recordings), Singing Vocal, Heavy
  Voice (creative), plus save-your-own

## Requirements

- Windows 10/11
- An NVIDIA GPU (most engines need CUDA; tested on an RTX 3090)
- [uv](https://astral.sh/uv) (Python package/env manager)
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (or set `paths.ffmpeg` in `config.yaml`)

## Install

```bash
git clone https://github.com/TanvirHafiz/studiovox.git
cd studiovox
```

Set up the app environment:

```bash
.\scripts\setup_app.ps1
```

Set up the engines (each runs in its own isolated Python environment, so version
conflicts between models never fight each other):

```bash
.\scripts\setup_engines.ps1
```

This installs DeepFilterNet, ClearerVoice, audio-separator, DNSMOS, and faster-whisper.
It's re-runnable and skips anything already working. One engine is opt-in because it needs
extra, somewhat fragile workarounds to run on Windows at all:

```bash
.\scripts\setup_engines.ps1 -IncludeOptional
```

adds Resemble Enhance (used by the "Rescue" preset's generative restoration step).

## Run

```bash
.\run.bat
```

Starts the local web server and opens `http://127.0.0.1:7860` once it's actually ready.
Drag a file in, pick a preset, hit Run. Drop 2+ files to run them as a batch.

## Command line

```bash
uv run python -m studiovox process my_recording.wav --preset clean_voiceover
uv run python -m studiovox batch ./my_recordings_folder --preset podcast
uv run python -m studiovox presets
```

## Presets

| Preset | What it does |
|---|---|
| Clean Voiceover | Faithful, natural cleanup for a decent recording |
| Podcast | Denoise + breath reduction, podcast loudness target |
| Rescue | For genuinely bad recordings: strong denoise, dereverb, bandwidth extension, blended generative restoration (integrity-checked) |
| Singing Vocal | Vocal isolation, not a speech denoiser — preserves vibrato and sustained notes |
| Heavy Voice | Creative effect: deep bass/depth boost for a heavier tone, applied after full denoising |

## Configuration

All paths live in `config.yaml` — nothing is hardcoded to a particular machine. Notably:
`paths.ffmpeg`, `gpu.require`, and `nvidia_afx.sdk_path` (only needed if you've separately
installed NVIDIA's Audio Effects SDK under their own license; the app never bundles it).

## Design notes

The full original design document — architecture, per-engine notes, the pipeline stage by
stage, and the phased build plan this was built against — is in [`plan.md`](plan.md).
