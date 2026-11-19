@echo off
REM One-time setup: creates a virtual environment and installs the dependencies.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ocr.txt || echo (optional Windows OCR packages could not be installed - bet/target reading disabled)
if not exist domino\engine\round_outcomes.json python tools\build_outcomes.py
echo.
echo Setup finished. Start the assistant with run.bat
pause
