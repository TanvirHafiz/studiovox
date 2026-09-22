@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found on PATH. Install it from https://astral.sh/uv and re-run.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Setting up app environment...
    uv sync
    if errorlevel 1 (
        echo Failed to set up the environment.
        pause
        exit /b 1
    )
)

start "" cmd /c "timeout /t 2 >nul && start http://127.0.0.1:7860"
uv run uvicorn app.main:app --host 127.0.0.1 --port 7860

endlocal
