# Sets up engine environments under engines/<name>/env. Re-runnable: skips engines whose
# smoke test already passes.
#
# Usage: .\setup_engines.ps1 [-IncludeOptional]
#   -IncludeOptional also installs engines/resemble (Resemble Enhance), needed only for the
#   "Rescue" preset's generative restoration step. It requires several Windows-specific
#   workarounds (see Install-Resemble below and engines/resemble/engine.yaml) and is skipped
#   by default per the plan: "Optional engines are only installed when the user opts in."
param(
    [switch]$IncludeOptional
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# torch/torchaudio pinned to 2.7.1 (cu128) everywhere: torchaudio >=2.9 removed the
# load/save/info I/O backend that deepfilternet and clearvoice both depend on. Once
# deepfilternet's env has it, later engines copy its site-packages instead of
# re-downloading ~3GB each - uv's direct download from download.pytorch.org has been
# observed to stall indefinitely on this connection for large wheels, while a plain
# file copy of an already-verified-working install is fast and reliable.
$TorchVersion = "2.7.1"
$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"

function Install-PinnedTorch($EnvPythonDir) {
    $sourceSitePackages = "engines\deepfilternet\env\Lib\site-packages"
    $targetSitePackages = "$EnvPythonDir\Lib\site-packages"
    if (Test-Path $sourceSitePackages) {
        Write-Host "  Copying pinned torch $TorchVersion from the deepfilternet env ..."
        robocopy $sourceSitePackages $targetSitePackages /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    } else {
        Write-Host "  Downloading torch $TorchVersion directly ..."
        uv pip install --python "$EnvPythonDir\Scripts\python.exe" `
            "torch==$TorchVersion" "torchaudio==$TorchVersion" --index-url $TorchIndexUrl
    }
}

function Test-DeepFilterNet {
    $py = "engines\deepfilternet\env\Scripts\python.exe"
    if (-not (Test-Path $py)) { return $false }
    & $py -c "import torch, torchaudio; from df.enhance import init_df, enhance, load_audio, save_audio; assert torch.cuda.is_available()" 2>$null
    return $LASTEXITCODE -eq 0
}

function Install-DeepFilterNet {
    Write-Host "Setting up engines/deepfilternet ..."
    uv venv --python 3.11 engines/deepfilternet/env
    uv pip install --python engines/deepfilternet/env/Scripts/python.exe `
        "torch==$TorchVersion" "torchaudio==$TorchVersion" --index-url $TorchIndexUrl
    uv pip install --python engines/deepfilternet/env/Scripts/python.exe `
        deepfilternet soundfile numpy

    if (-not (Test-DeepFilterNet)) {
        Write-Error "DeepFilterNet smoke test failed after install."
        exit 1
    }
    Write-Host "deepfilternet OK"
}

function Test-ClearerVoice {
    $py = "engines\clearervoice\env\Scripts\python.exe"
    if (-not (Test-Path $py)) { return $false }
    & $py -c "import torch; from clearvoice import ClearVoice; assert torch.cuda.is_available()" 2>$null
    return $LASTEXITCODE -eq 0
}

function Install-ClearerVoice {
    Write-Host "Setting up engines/clearervoice ..."
    uv venv --python 3.11 engines/clearervoice/env
    Install-PinnedTorch "engines\clearervoice\env"
    "torch==$TorchVersion" | Out-File -Encoding ascii $env:TEMP\studiovox_constraints.txt
    "torchaudio==$TorchVersion" | Out-File -Encoding ascii -Append $env:TEMP\studiovox_constraints.txt
    uv pip install --python engines/clearervoice/env/Scripts/python.exe `
        clearvoice --constraint $env:TEMP\studiovox_constraints.txt

    if (-not (Test-ClearerVoice)) {
        Write-Error "ClearerVoice smoke test failed after install."
        exit 1
    }
    Write-Host "clearervoice OK"
}

function Test-Separator {
    $py = "engines\separator\env\Scripts\python.exe"
    if (-not (Test-Path $py)) { return $false }
    & $py -c "import torch; from audio_separator.separator import Separator; assert torch.cuda.is_available()" 2>$null
    return $LASTEXITCODE -eq 0
}

function Install-Separator {
    Write-Host "Setting up engines/separator ..."
    uv venv --python 3.11 engines/separator/env
    Install-PinnedTorch "engines\separator\env"
    "torch==$TorchVersion" | Out-File -Encoding ascii $env:TEMP\studiovox_constraints.txt
    "torchaudio==$TorchVersion" | Out-File -Encoding ascii -Append $env:TEMP\studiovox_constraints.txt
    uv pip install --python engines/separator/env/Scripts/python.exe `
        "audio-separator[gpu]" --constraint $env:TEMP\studiovox_constraints.txt

    if (-not (Test-Separator)) {
        Write-Error "audio-separator smoke test failed after install."
        exit 1
    }
    Write-Host "separator OK"
    Write-Host "  Note: onnxruntime-gpu here targets CUDA 13; our pinned torch uses CUDA 12," `
        "so any ONNX-format model (not the MelBand/BS-Roformer .ckpt models this app uses" `
        "by default) will fall back to CPU. Functional, just slower for those models."
}

function Test-DNSMOS {
    $py = "engines\dnsmos\env\Scripts\python.exe"
    if (-not (Test-Path $py)) { return $false }
    if (-not (Test-Path "models\dnsmos\sig_bak_ovr.onnx")) { return $false }
    if (-not (Test-Path "models\dnsmos\model_v8.onnx")) { return $false }
    & $py -c "import onnxruntime, librosa, soundfile" 2>$null
    return $LASTEXITCODE -eq 0
}

function Install-DNSMOS {
    Write-Host "Setting up engines/dnsmos ..."
    uv venv --python 3.11 engines/dnsmos/env
    uv pip install --python engines/dnsmos/env/Scripts/python.exe `
        onnxruntime librosa soundfile numpy

    New-Item -ItemType Directory -Force models/dnsmos | Out-Null
    if (-not (Test-Path "models\dnsmos\sig_bak_ovr.onnx")) {
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx" `
            -OutFile models/dnsmos/sig_bak_ovr.onnx
    }
    if (-not (Test-Path "models\dnsmos\model_v8.onnx")) {
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx" `
            -OutFile models/dnsmos/model_v8.onnx
    }

    if (-not (Test-DNSMOS)) {
        Write-Error "DNSMOS smoke test failed after install."
        exit 1
    }
    Write-Host "dnsmos OK"
}

function Test-Whisper {
    $py = "engines\whisper\env\Scripts\python.exe"
    if (-not (Test-Path $py)) { return $false }
    & $py -c "from faster_whisper import WhisperModel" 2>$null
    return $LASTEXITCODE -eq 0
}

function Install-Whisper {
    Write-Host "Setting up engines/whisper ..."
    uv venv --python 3.11 engines/whisper/env
    uv pip install --python engines/whisper/env/Scripts/python.exe faster-whisper

    if (-not (Test-Whisper)) {
        Write-Error "faster-whisper smoke test failed after install."
        exit 1
    }
    Write-Host "whisper OK"
}

function Test-Resemble {
    $py = "engines\resemble\env\Scripts\python.exe"
    if (-not (Test-Path $py)) { return $false }
    & $py -c "import pathlib; pathlib.PosixPath = pathlib.WindowsPath; import deepspeed; from resemble_enhance.enhancer.inference import enhance; import torch; assert torch.cuda.is_available()" 2>$null
    return $LASTEXITCODE -eq 0
}

function Install-Resemble {
    Write-Host "Setting up engines/resemble (optional) ..."
    Write-Host "  This engine needs several Windows-specific workarounds; see engine.yaml."
    uv venv --python 3.11 engines/resemble/env

    $torchWheel = "$env:TEMP\studiovox_torch-2.1.1+cu118.whl"
    $torchaudioWheel = "$env:TEMP\studiovox_torchaudio-2.1.1+cu118.whl"
    if (-not (Test-Path $torchWheel)) {
        Invoke-WebRequest -Uri "https://download.pytorch.org/whl/cu118/torch-2.1.1%2Bcu118-cp311-cp311-win_amd64.whl" -OutFile $torchWheel
    }
    if (-not (Test-Path $torchaudioWheel)) {
        Invoke-WebRequest -Uri "https://download.pytorch.org/whl/cu118/torchaudio-2.1.1%2Bcu118-cp311-cp311-win_amd64.whl" -OutFile $torchaudioWheel
    }
    uv pip install --python engines/resemble/env/Scripts/python.exe $torchWheel $torchaudioWheel

    # deepspeed==0.12.4 (resemble-enhance's own pin) fails to compile on Windows (no async
    # I/O support). This is a community precompiled wheel for inference use; functionally
    # compatible. Its filename version tag doesn't quite match its internal metadata, hence
    # the skip-check env var.
    $env:UV_SKIP_WHEEL_FILENAME_CHECK = "1"
    uv pip install --python engines/resemble/env/Scripts/python.exe `
        "https://github.com/daswer123/xtts-webui/releases/download/deepspeed/deepspeed-0.11.2+cuda118-cp311-cp311-win_amd64.whl"
    Remove-Item Env:\UV_SKIP_WHEEL_FILENAME_CHECK

    # deepspeed 0.11.2 imports numpy.BUFSIZE, removed in numpy 2.0.
    uv pip install --python engines/resemble/env/Scripts/python.exe "numpy==1.26.2"

    uv pip install --python engines/resemble/env/Scripts/python.exe resemble-enhance --no-deps
    "torch==2.1.1" | Out-File -Encoding ascii $env:TEMP\studiovox_resemble_constraints.txt
    "torchaudio==2.1.1" | Out-File -Encoding ascii -Append $env:TEMP\studiovox_resemble_constraints.txt
    "numpy==1.26.2" | Out-File -Encoding ascii -Append $env:TEMP\studiovox_resemble_constraints.txt
    uv pip install --python engines/resemble/env/Scripts/python.exe `
        "librosa==0.10.1" "omegaconf==2.3.0" "resampy==0.4.2" "soundfile==0.12.1" "tabulate>=0.9.0" `
        "celluloid==0.2.0" "ptflops==0.7.1.2" rich tqdm scipy pandas matplotlib `
        --constraint $env:TEMP\studiovox_resemble_constraints.txt

    if (-not (Test-Resemble)) {
        Write-Error "Resemble Enhance smoke test failed after install."
        exit 1
    }
    Write-Host "resemble OK"
}

$engines = @(
    @{ Name = "deepfilternet"; Test = ${function:Test-DeepFilterNet}; Install = ${function:Install-DeepFilterNet} },
    @{ Name = "clearervoice"; Test = ${function:Test-ClearerVoice}; Install = ${function:Install-ClearerVoice} },
    @{ Name = "separator"; Test = ${function:Test-Separator}; Install = ${function:Install-Separator} },
    @{ Name = "dnsmos"; Test = ${function:Test-DNSMOS}; Install = ${function:Install-DNSMOS} },
    @{ Name = "whisper"; Test = ${function:Test-Whisper}; Install = ${function:Install-Whisper} }
)

if ($IncludeOptional) {
    $engines += @{ Name = "resemble"; Test = ${function:Test-Resemble}; Install = ${function:Install-Resemble} }
}

foreach ($e in $engines) {
    if (& $e.Test) {
        Write-Host "$($e.Name) already healthy, skipping."
    } else {
        & $e.Install
    }
}

Write-Host "Engine setup complete."
if (-not $IncludeOptional) {
    Write-Host "Skipped optional engine: resemble (Rescue preset's generative restore). Run with -IncludeOptional to install it."
}
