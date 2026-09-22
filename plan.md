# StudioVox: Local Offline Vocal Enhancement Studio

Working title: **StudioVox**. Rename freely.

## 1. Goal

A local Windows desktop app that takes a recorded voice file (speech or singing, or a video containing one) and produces studio-quality output by chaining the best available AI models plus classic DSP finishing. Processing is offline (not real-time), so every stage may use large models with long look-ahead.

Target quality: better than NVIDIA Broadcast on recorded material. Faithful to the original voice by default, with optional generative "studio mic" restoration that is blended in, never forced.

### Target machine

- Windows 10, NVIDIA RTX 3090 (24 GB VRAM), Miniconda installed
- History of PyTorch DLL conflicts between environments (WinError 1114 / c10.dll). The architecture below isolates every heavy model in its own environment to avoid this entirely.

### Non-goals

- Real-time or live microphone processing
- Cloud processing or paid APIs (the app must work fully offline once models are downloaded)
- Music mixing/mastering of full songs (vocal track only)

## 2. Instructions for Claude Code

1. Build phase by phase (section 10). Do not start a phase until the previous phase passes its acceptance checks.
2. Before installing any model package, read its current README on GitHub/PyPI. Install commands, Python versions and weights locations change. Do not rely on memory for them.
3. Never install model packages into the app's main environment. Each engine gets its own isolated environment (section 4).
4. Never "fix" a dependency conflict by upgrading or downgrading torch in a shared environment. If an engine cannot be installed cleanly, stop, report the exact error, and propose options.
5. Keep all paths configurable in `config.yaml`. No hardcoded user paths.
6. Write clear logs to `logs/` for every job.
7. No em dashes anywhere in UI text, docs, or comments.

## 3. Architecture overview

```
[Web UI (localhost)] <-> [FastAPI app server + job orchestrator]
                                  |
                                  | spawns one subprocess per stage
                                  v
        [Engine workers, each in its own isolated Python env]
        deepfilternet | clearervoice | separator | resemble | nvidia_afx | whisper | dnsmos
                                  |
                                  v
               [Job folder on disk: WAV files between stages]
```

- **App server**: Python 3.11, FastAPI, uvicorn. Orchestrates jobs, serves the UI, runs lightweight DSP (resampling, finishing chain, loudness).
- **Engine workers**: small `worker.py` scripts, each run with its own environment's interpreter. They communicate only through files and a JSON contract. A crashed engine never crashes the app.
- **Why subprocesses**: incompatible torch/torchaudio pins between models (Resemble Enhance in particular pins old versions), clean VRAM release after every stage (process exit frees GPU memory), and no DLL conflicts.
- **UI**: single-page web app served by FastAPI at `http://127.0.0.1:7860`, opened automatically in the default browser by `run.bat`. Vanilla JS or lightweight Preact via CDN, plus wavesurfer.js for waveforms/spectrograms.

## 4. Engine environments

Managed with **uv** (fast, per-folder venvs). Fall back to separate conda envs only if an engine requires it.

```
engines/
  deepfilternet/   env/  worker.py  engine.yaml
  clearervoice/    env/  worker.py  engine.yaml
  separator/       env/  worker.py  engine.yaml   # audio-separator (Mel-Band RoFormer etc.)
  resemble/        env/  worker.py  engine.yaml   # optional
  nvidia_afx/      worker.py  engine.yaml         # optional, wraps SDK CLI, user installs SDK
  whisper/         env/  worker.py  engine.yaml   # faster-whisper, for integrity checks
  dnsmos/          env/  worker.py  engine.yaml   # quality metrics (ONNX)
```

`engine.yaml` declares:

```yaml
name: clearervoice
python: engines/clearervoice/env/Scripts/python.exe
sample_rates: [48000]          # rates the engine accepts natively
channels: mono
max_chunk_seconds: 30          # orchestrator chunks longer audio
needs_gpu: true
vram_gb_estimate: 4
license: Apache-2.0
tasks: [denoise, super_resolution]
```

`scripts/setup_engines.ps1` creates each env, installs its packages, downloads weights into `models/<engine>/`, and runs a smoke test. It must be re-runnable and skip engines that are already healthy. Optional engines are only installed when the user opts in.

### Worker contract (all engines)

```
<env python> worker.py --task <task> --in <in.wav> --out <out.wav> --params '<json>'
```

- stdout lines `PROGRESS 0.37` for progress; final line `RESULT {json}` with metrics/notes.
- Exit code 0 on success, non-zero with a human-readable error on stderr.
- Input/output: WAV, float32, at a rate from `sample_rates`. The orchestrator handles resampling (soxr, highest quality) before and after.

## 5. Engines and what they do

| Engine | Tasks | Notes |
|---|---|---|
| DeepFilterNet3 | denoise | Fast, faithful, full-band 48 kHz. Default first pass. Check torchaudio compatibility (known import issues with newer torchaudio); pin inside its own env. |
| ClearerVoice-Studio | denoise (MossFormer2_SE_48K), super_resolution (MossFormer2_SR_48K) | Stronger denoise for difficult noise. SR only when source bandwidth is limited. Apache-2.0. |
| audio-separator | vocal isolation (Mel-Band RoFormer / BS-RoFormer vocal models), dereverb / de-echo models | Primary path for **singing**. Also used for dereverb in speech mode. Verify which dereverb models are currently available in the package's model list and pick the best-rated one. |
| Resemble Enhance | generative restore (enhancer), denoise | Optional. 44.1 kHz. Can hallucinate: always blended and always integrity-checked. |
| NVIDIA AFX SDK | studio_voice_high_quality, dereverb_denoiser, superres | Optional adapter. The user downloads the SDK and models from NVIDIA under NVIDIA's license; the app only wraps its sample CLI. Use the high-quality (offline) variants only. |
| faster-whisper | transcription for integrity check | large-v3 on GPU. Speech mode only. |
| DNSMOS (ONNX) | quality scoring (SIG, BAK, OVRL) | Used for per-stage metrics and auto-comparison. Optionally add UTMOS if easy to install. |

## 6. Processing pipeline

All internal audio: 48 kHz, float32, mono (stereo inputs: offer "downmix" (default for voice) or "process L/R separately").

### Stage 0: Ingest

- Decode any audio or video via ffmpeg (ship or locate `ffmpeg.exe`, configurable path).
- If input is video, remember it and extract audio.
- Analyze and store in `job.json`:
  - peak dBFS, clipping percentage (consecutive samples at or near full scale)
  - noise floor estimate (RMS of quietest 10% of 50 ms frames)
  - estimated effective bandwidth (spectral rolloff of voiced frames), to decide whether super-resolution is needed
  - speech vs singing guess (pitch stability and sustained-note heuristics); the user can override
  - DNSMOS of the original
- Show warnings: heavy clipping ("cannot be fully repaired"), very low bandwidth, very low level.

### Stage 1: Declip (optional, auto-suggested if clipping > 0.1%)

- Classic DSP declipper (e.g. constrained cubic/AR interpolation of clipped runs). Mark in UI as "partial repair".

### Stage 2: Isolate / Denoise

- Speech: DeepFilterNet3 (default) or MossFormer2_SE_48K (for tough noise). Strength control via attenuation limit where the engine supports it, otherwise via wet/dry blend.
- Singing: Mel-Band RoFormer vocal model from audio-separator. Speech denoisers are **disabled** in singing mode (they damage vibrato and sustained notes).

### Stage 3: Dereverb (optional)

- audio-separator dereverb model, or NVIDIA AFX dereverb when installed.
- Strength via wet/dry.

### Stage 4: Bandwidth extension (conditional)

- Run MossFormer2_SR_48K only if Stage 0 bandwidth estimate is below ~14 kHz, or if the user forces it.

### Stage 5: Generative restore (optional, off by default except in "Rescue" preset)

- Engine choice: NVIDIA Studio Voice HQ or Resemble Enhance.
- Output is **blended** with the Stage 4 output. Default blend 40% wet.
- **Phase/timing alignment before blending**: generative outputs are not sample-aligned or phase-coherent with the input. Cross-correlate to remove any latency offset, then offer two blend modes:
  - `time` blend (simple mix, aligned). Warn if comb filtering is detected (spectral notch check).
  - `band` blend (default): below a crossover (default 4 kHz) keep the faithful signal, above it mix in the generative signal. Linkwitz-Riley crossover. This gives the "studio air" without phasey low mids.
- Speech mode only. Hidden in singing mode.

### Stage 6: Finishing chain (DSP, runs in app env)

Implement with numpy/scipy (or Spotify `pedalboard` for speed; note pedalboard is GPL-3.0, fine for personal use, flag it if distributing).

1. High-pass 70 to 90 Hz (preset dependent), 18 dB/oct
2. Gentle corrective EQ: optional mud cut around 250 to 400 Hz, presence lift around 3 to 5 kHz, air shelf above 10 kHz (all small, preset-driven, user adjustable)
3. De-esser: dynamic band 5 to 9 kHz, auto threshold from sibilance analysis
4. Compressor: ~3:1, slow-ish attack for speech, gentle for singing
5. Optional breath reduction (detect breaths by energy + spectral flatness between phrases; attenuate, never delete, default -8 dB)
6. Loudness normalization (pyloudnorm, ITU-R BS.1770) plus true-peak limiter:
   - Podcast: -16 LUFS, -1 dBTP
   - YouTube/voiceover: -14 LUFS, -1 dBTP
   - Broadcast: -23 LUFS, -1 dBTP
   - Vocal stem for mixing: no normalization, peak at -6 dBFS

### Stage 7: Export

- WAV 48 kHz 24-bit (default), FLAC, MP3 320.
- If input was video: remux enhanced audio into the video with ffmpeg (copy video stream, no re-encode), output as a new file.
- Output length must equal input length exactly (sample-accurate) so video sync is preserved. Verify and pad/trim if an engine drifted.
- Optionally export every intermediate stage WAV.

### Chunking for long files

- Orchestrator splits audio into chunks of each engine's `max_chunk_seconds` with 1 s overlap, runs the worker once with a chunk list (models load once per stage, not once per chunk), and recombines with equal-power crossfades.
- Test explicitly that no clicks or level jumps appear at chunk borders.

## 7. Quality guards

These separate a toy from a pro tool.

1. **Level-matched A/B**: when comparing any two versions in the UI, both play at matched integrated loudness. Louder always sounds "better"; this removes that bias.
2. **Per-stage metrics**: DNSMOS SIG/BAK/OVRL after every stage, shown as a small table. If a stage lowers OVRL by more than 0.2, flag it.
3. **Content integrity check (speech)**: transcribe the Stage 2 output and the final output with faster-whisper; compute word error rate between them. If WER > 5% or any segment differs, list the timestamps so the user can listen. Catches generative hallucination (changed or invented syllables).
4. **Timing check**: cross-correlation between input and output; report any offset, auto-compensate.
5. **Never overwrite the original.** Every job lives in its own folder.

## 8. Presets (`presets/*.yaml`)

| Preset | Stages | Notes |
|---|---|---|
| Clean Voiceover (default) | DeepFilterNet (moderate) > dereverb light > finish -14 LUFS | Faithful, natural |
| Podcast | DeepFilterNet > dereverb > finish -16 LUFS + breath reduction | |
| Rescue (bad recording) | MossFormer2 SE > dereverb strong > SR if needed > generative restore 50% band blend > finish | Integrity check mandatory |
| Phone / Low-bandwidth | denoise > SR forced > generative restore 40% > finish | |
| Singing Vocal | RoFormer vocal isolation > dereverb light > finish (stem mode, no loudness norm) | No speech models |
| Custom | user picks | Saved as a new preset |

The preset recommendation in Stage 0 picks one automatically from the analysis; the user can change it.

## 9. UI spec

Single page, dark theme, desktop-first.

1. **Drop zone**: drag files or a folder (batch). Shows analysis summary per file with warnings.
2. **Pipeline panel**: preset dropdown, then a vertical list of stage cards. Each card: enable toggle, engine dropdown (only installed engines), strength/blend slider, and advanced settings collapsed.
3. **Run**: progress bar per stage and overall, current engine name, ETA. Cancel button (kills the worker process).
4. **Results view**:
   - Waveform + spectrogram for Original and Final (toggle to any intermediate stage)
   - A/B toggle (keyboard: `A` / `B`, space to play) with loudness matching, synced playhead
   - Loop region selection for focused listening
   - Metrics table and integrity warnings with clickable timestamps
5. **Export panel**: format, loudness target, "remux into video" checkbox, open output folder.
6. **Engines page**: installed/missing status, VRAM estimate, install button for optional engines (runs setup script), link to NVIDIA SDK download instructions.
7. **Batch queue**: process a folder with one preset; summary table at the end.

## 10. Build phases

### Phase 0: Skeleton and environment
- Repo layout, `config.yaml`, logging, `run.bat`, app venv with uv (Python 3.11).
- ffmpeg detection, GPU detection (nvidia-smi), disk space check.
- **Accept**: `run.bat` opens the empty UI; health endpoint reports GPU and ffmpeg.

### Phase 1: Core CLI pipeline (no UI)
- Ingest + analysis, DeepFilterNet engine, finishing chain, loudness, export, chunking.
- CLI: `python -m studiovox process in.wav --preset clean_voiceover`
- **Accept**: 10-minute noisy test file processes without chunk-border artifacts; output length matches input; loudness within 0.5 LU of target.

### Phase 2: Web UI
- Upload, preset/stage editor, job progress (WebSocket or SSE), results view with A/B, export.
- **Accept**: full run from browser; A/B switching is gapless and loudness-matched.

### Phase 3: More engines
- ClearerVoice (SE + SR), audio-separator (vocal isolation + dereverb), DNSMOS metrics.
- **Accept**: per-stage metrics display; SR only triggers on low-bandwidth inputs.

### Phase 4: Generative restore + guards
- Resemble Enhance engine, alignment, time and band blend, faster-whisper integrity check.
- **Accept**: hallucination test (section 11) is flagged by the integrity check; band blend shows no comb filtering on test files.

### Phase 5: Singing mode
- Speech/singing detection, singing preset, speech engines disabled in singing mode.
- **Accept**: sustained notes and vibrato survive intact on the singing test clip (listening test + pitch contour comparison before/after).

### Phase 6: NVIDIA AFX adapter (optional)
- Wrap the SDK sample CLI for `studio_voice_high_quality`, `dereverb_denoiser`, `superres`. Detect SDK path from config; clear message if missing.
- **Accept**: Studio Voice HQ selectable as a Stage 5 engine when the SDK is installed; app works normally when it is not.

### Phase 7: Batch, polish, packaging
- Batch queue, custom presets, video remux, job history.
- Optional: package the app env with PyInstaller; engines stay as folders beside it.

## 11. Test set

Create `tests/audio/` with these clips (generate synthetic ones by mixing clean speech with noise/impulse responses; the user will add real recordings):

1. Laptop mic speech + fan/AC noise
2. Speech in a reverberant room
3. Phone-quality speech (8 or 16 kHz source, upsampled)
4. Speech with intermittent noise (keyboard, dog, traffic)
5. Clipped speech
6. Singing vocal with room noise and reverb
7. 30-minute speech file (chunking and memory test)
8. Video file (MP4) with noisy dialogue (remux and sync test)
9. Hallucination probe: very noisy speech where words are barely intelligible (integrity check must flag differences)

Automated tests: output length equality, loudness target, no NaN/inf, no chunk-border clicks (detect sample discontinuities), DNSMOS OVRL of final >= original for clips 1 to 4.

## 12. Honest limits (show in the app's Help page)

- Clipping, heavy distortion and missing words cannot be truly recovered; generative restore can only guess.
- Generative engines may change voice timbre or invent syllables. That is why they are blended and checked.
- The best result still comes from a decent recording: 48 kHz, 24-bit, peaks around -12 dBFS, mic close to the mouth, soft room.

## 13. Repo layout

```
studiovox/
  run.bat
  config.yaml
  app/
    main.py            # FastAPI
    orchestrator.py    # jobs, chunking, resampling, worker calls
    analysis.py        # stage 0
    dsp/               # declip, eq, deesser, compressor, breath, loudness, blend, align
    engines.py         # engine registry from engine.yaml files
    presets.py
  engines/<name>/...
  presets/*.yaml
  web/ index.html  app.js  styles.css
  scripts/ setup_app.ps1  setup_engines.ps1
  models/              # weights, gitignored
  jobs/                # per-job folders, gitignored
  logs/
  tests/
```
