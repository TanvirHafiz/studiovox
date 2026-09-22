# Sets up engine environments under engines/<name>/env. Re-runnable; skips engines already healthy.
# Placeholder for Phase 1+. Engines are added one at a time as they are integrated.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "No engines registered yet. This script will grow as engines are added in later phases."
