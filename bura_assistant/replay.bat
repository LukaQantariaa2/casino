@echo off
REM Replay a screen recording through the overlay:   replay.bat "C:\path\to\recording.mp4"
REM Text-only QA report instead:                     python tools\replay_report.py "recording.mp4" --advice
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if "%~1"=="" (
  echo Usage: replay.bat "path\to\recording.mp4"
  pause
  exit /b 1
)
python run.py --replay "%~1"
if errorlevel 1 pause
