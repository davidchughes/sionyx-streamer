@echo off
:: ==============================================================================
::  SiOnyx Streamer -- Windows Installer
::  Double-click this once to install everything needed.
:: ==============================================================================

echo.
echo ============================================================
echo   SiOnyx Streamer -- Windows Installer
echo ============================================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo   Python was not found.
    echo.
    echo   Please install Python 3.10 or newer from:
    echo     https://www.python.org/downloads/
    echo.
    echo   During install, make sure to tick:
    echo     "Add Python to PATH"
    echo.
    echo   Then run this installer again.
    echo.
    pause
    exit /b 1
)

echo   Python found. Installing required packages...
echo.

pip install opencv-python numpy --quiet

if errorlevel 1 (
    echo.
    echo   Something went wrong installing packages.
    echo   Try running this file as Administrator.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Installation complete.
echo.
echo   To process a video file:
echo     Drag any .mp4 file onto "process_video.bat"
echo     Or double-click "process_video.bat" and follow the prompt.
echo ============================================================
echo.
pause
