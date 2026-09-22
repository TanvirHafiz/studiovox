# Sets up engine environments under engines/<name>/env. Re-runnable: skips engines whose
# smoke test already passes. Optional engines are only installed when the user opts in
# (none are optional yet in Phase 1).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Test-DeepFilterNet {
    $py = "engines\deepfilternet\env\Scripts\python.exe"
    if (-not (Test-Path $py)) { return $false }
    & $py -c "import torch, torchaudio; from df.enhance import init_df, enhance, load_audio, save_audio; assert torch.cuda.is_available()" 2>$null
    return $LASTEXITCODE -eq 0
}

function Install-DeepFilterNet {
    Write-Host "Setting up engines/deepfilternet ..."
    uv venv --python 3.11 engines/deepfilternet/env

    # torch/torchaudio pinned to 2.7.1: torchaudio >=2.9 removed the load/save/info I/O
    # backend that deepfilternet's df.io module depends on (see engine.yaml notes).
    uv pip install --python engines/deepfilternet/env/Scripts/python.exe `
        "torch==2.7.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python engines/deepfilternet/env/Scripts/python.exe `
        deepfilternet soundfile numpy

    if (-not (Test-DeepFilterNet)) {
        Write-Error "DeepFilterNet smoke test failed after install."
        exit 1
    }
    Write-Host "deepfilternet OK"
}

if (Test-DeepFilterNet) {
    Write-Host "deepfilternet already healthy, skipping."
} else {
    Install-DeepFilterNet
}

Write-Host "Engine setup complete."
