@echo off
REM SiOnyx Aurora Windows Client Installation Script

echo ================================
echo SiOnyx Aurora Client Installer
echo ================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install opencv-python numpy requests

echo.
echo [2/3] Creating desktop shortcut...
set SCRIPT_DIR=%~dp0
set SHORTCUT_PATH=%USERPROFILE%\Desktop\SiOnyx Aurora.lnk

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_PATH%'); $s.TargetPath = 'python'; $s.Arguments = '%SCRIPT_DIR%windows_client.py'; $s.WorkingDirectory = '%SCRIPT_DIR%'; $s.Save()"

echo.
echo [3/3] Configuration...
echo.
echo Please edit windows_client.py and set your RPi IP address:
echo   SERVER_IP = "192.168.1.100"  # Change this to your RPi IP
echo.

echo ================================
echo Installation Complete!
echo ================================
echo.
echo Desktop shortcut created: SiOnyx Aurora
echo.
echo Or run manually:
echo   python windows_client.py
echo.
pause
