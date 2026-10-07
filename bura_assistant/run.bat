@echo off
REM Start the overlay. Put Chrome in fullscreen (F11) on the game table first.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python run.py %*
if errorlevel 1 pause
