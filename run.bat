@echo off
REM ===  magic-math : double-click to train on your own GPU  ===
REM Creates a virtual environment, installs PyTorch (+CUDA if you have an
REM NVIDIA GPU), and opens the live training dashboard in your browser.
REM The only thing you need installed first is Python 3.10+.

setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py scripts\bootstrap.py %*
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    python scripts\bootstrap.py %*
  ) else (
    echo.
    echo  Python was not found. Install it from https://www.python.org/downloads/
    echo  ^(tick "Add Python to PATH" in the installer^), then run this again.
    echo.
    pause
    exit /b 1
  )
)

echo.
pause
