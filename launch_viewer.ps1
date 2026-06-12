# FractalForge Viewer Launcher
# Double-click or run from PowerShell to launch the interactive explorer.

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $scriptDir "venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: Virtual environment not found at $venvPython" -ForegroundColor Red
    Write-Host "Run: python -m venv venv && venv\Scripts\pip install -e '.[dev]'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Launching FractalForge Viewer..." -ForegroundColor Cyan
& $venvPython -m fractalforge.cli.main viewer
