@echo off
REM ===  magic-math : double-click to train on your own GPU  ===
REM Creates a virtual environment, installs PyTorch (+CUDA if you have an
REM NVIDIA GPU), and opens the live training dashboard in your browser.
REM The only thing you need installed first is Python 3.10+.

setlocal
cd /d "%~dp0"

REM Find a Python launcher (prefer the Windows "py" launcher, then "python").
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py"
if not defined PYEXE where python >nul 2>nul && set "PYEXE=python"

if not defined PYEXE (
  echo.
  echo  Python was not found. Install it from https://www.python.org/downloads/
  echo  ^(tick "Add Python to PATH" in the installer^), then run this again.
  echo.
  pause
  exit /b 1
)

%PYEXE% scripts\bootstrap.py %*

echo.
pause
