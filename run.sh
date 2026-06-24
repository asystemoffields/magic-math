#!/usr/bin/env bash
# ===  magic-math : train on your own GPU (macOS / Linux)  ===
# Creates a virtual environment, installs PyTorch, and opens the live
# training dashboard in your browser. Needs only Python 3.10+ installed.
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python not found. Install Python 3.10+ and try again."
  exit 1
fi

exec "$PY" scripts/bootstrap.py "$@"
