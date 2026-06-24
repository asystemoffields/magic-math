#!/usr/bin/env bash
# ===  magic-math : play with your trained model (macOS / Linux)  ===
# Opens a little playground in your browser for the model you trained with
# ./run.sh. Run ./run.sh first if you haven't trained a model yet.
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

exec "$PY" scripts/bootstrap.py chat "$@"
