@echo off
REM ===  magic-math : double-click to play with your trained model  ===
REM Opens a little playground in your browser where you can prompt the model
REM you trained with run.bat. (Run run.bat first if you haven't trained yet.)
REM Needs only Python 3.10+; it reuses the same environment run.bat set up.

setlocal
cd /d "%~dp0"

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

%PYEXE% scripts\bootstrap.py chat %*

echo.
pause
