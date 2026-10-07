@echo off
REM Optional: build a standalone BuraAssistant.exe (folder build) with PyInstaller.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m pip install pyinstaller
pyinstaller --noconfirm --name BuraAssistant --windowed run.py
xcopy /E /I /Y config dist\BuraAssistant\config >nul
echo.
echo Built dist\BuraAssistant\BuraAssistant.exe  (config copied next to it)
echo Set "cards_dir" in dist\BuraAssistant\config\app.json to your cards folder (absolute path is fine).
pause
