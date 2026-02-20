@echo off
:: ==============================================================================
::  SiOnyx Streamer -- Video Processor
::  Drag a .mp4 file onto this to process it, or double-click to browse.
:: ==============================================================================

setlocal

set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%local_video_star_enhancement.py

:: ------------------------------------------------------------------------------
:: Check Python
:: ------------------------------------------------------------------------------

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Python was not found. Please run install_windows.bat first.
    echo.
    pause
    exit /b 1
)

:: ------------------------------------------------------------------------------
:: Get input file -- either dragged onto this bat or prompted
:: ------------------------------------------------------------------------------

if "%~1"=="" (
    echo.
    echo   No file was dragged onto this script.
    echo.
    set /p INPUT_FILE="  Enter the full path to your video file: "
) else (
    set INPUT_FILE=%~1
)

if not exist "%INPUT_FILE%" (
    echo.
    echo   File not found: %INPUT_FILE%
    echo.
    pause
    exit /b 1
)

:: ------------------------------------------------------------------------------
:: Ask where to save output
:: ------------------------------------------------------------------------------

echo.
echo   Input:  %INPUT_FILE%
echo.
set /p SAVE_FILE="  Save enhanced video to (leave blank to just preview, no save): "

:: ------------------------------------------------------------------------------
:: Run
:: ------------------------------------------------------------------------------

echo.
echo   Starting -- press Q in the preview window to quit.
echo.

if "%SAVE_FILE%"=="" (
    python "%PYTHON_SCRIPT%" "%INPUT_FILE%"
) else (
    python "%PYTHON_SCRIPT%" "%INPUT_FILE%" -s "%SAVE_FILE%"
)

if errorlevel 1 (
    echo.
    echo   Something went wrong. Make sure install_windows.bat has been run.
    echo.
)

echo.
pause
endlocal
