# Sets up the StudioVox app environment (not the engine environments; see setup_engines.ps1).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found on PATH. Install it from https://astral.sh/uv"
    exit 1
}

uv sync
Write-Host "App environment ready. Run run.bat to start StudioVox."
