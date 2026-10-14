@echo off
rem Launch Surf-Ace inside the surf-ace conda environment.
call "%LOCALAPPDATA%\miniforge3\Scripts\activate.bat" surf-ace || (
    echo Could not activate the "surf-ace" conda environment. See README.md for setup.
    pause
    exit /b 1
)
cd /d "%~dp0"
python -m surf_ace %*
