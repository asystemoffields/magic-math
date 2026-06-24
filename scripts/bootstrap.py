"""
One-command setup + launch for the local GPU path.

Assumes the machine has *only* Python installed. This script:
  1. creates a private virtual environment in .venv/ (so we touch nothing global)
  2. installs PyTorch — the CUDA build if an NVIDIA GPU is detected, else CPU
  3. installs numpy + tokenizers
  4. launches the live training dashboard and opens your browser

Run it directly:   python scripts/bootstrap.py
or via the wrappers: run.bat  (Windows)   /   ./run.sh  (macOS/Linux)

Anything after the script name is forwarded to the dashboard, e.g.:
  python scripts/bootstrap.py --preset small
"""

import os
import shutil
import subprocess
import sys
import venv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(ROOT, ".venv")

# The CUDA wheel index. cu121 has the broadest driver compatibility; bump this
# if you have a newer driver and want a newer CUDA. (Edit one line, that's it.)
CUDA_INDEX = "https://download.pytorch.org/whl/cu121"


def venv_python(path):
    return os.path.join(path, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(path, "bin", "python")


def have_nvidia_gpu():
    return shutil.which("nvidia-smi") is not None


def pip(py, *args):
    subprocess.check_call([py, "-m", "pip", *args])


def main():
    print("\n=== magic-math setup ===\n")

    if not os.path.exists(VENV):
        print("· creating virtual environment in .venv/ …")
        venv.EnvBuilder(with_pip=True).create(VENV)
    py = venv_python(VENV)

    print("· upgrading pip …")
    pip(py, "install", "-q", "--upgrade", "pip")

    # Only (re)install torch if it isn't already importable in the venv.
    torch_ok = subprocess.call([py, "-c", "import torch"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if not torch_ok:
        if have_nvidia_gpu():
            print(f"· NVIDIA GPU detected — installing CUDA PyTorch from {CUDA_INDEX}")
            try:
                pip(py, "install", "-q", "torch", "--index-url", CUDA_INDEX)
            except subprocess.CalledProcessError:
                print("  ! CUDA wheel failed; falling back to the CPU build.")
                pip(py, "install", "-q", "torch")
        else:
            print("· no NVIDIA GPU detected — installing CPU PyTorch "
                  "(training will be slow; the Colab path is recommended instead)")
            pip(py, "install", "-q", "torch")

    print("· installing numpy + tokenizers …")
    pip(py, "install", "-q", "numpy", "tokenizers")

    print("\n· launching the dashboard …\n")
    # Run the web module from the repo root so `magicmath` imports cleanly.
    env = dict(os.environ, PYTHONPATH=ROOT)
    subprocess.call([py, "-m", "magicmath.web", *sys.argv[1:]], cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
