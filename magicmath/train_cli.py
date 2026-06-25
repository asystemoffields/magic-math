"""
Headless training from the terminal — no browser, just text output.

    python -m magicmath.train_cli
    python -m magicmath.train_cli --max-steps 2000   # a shorter run

This is the same code the notebook and the web dashboard run; it just uses the
plain-text `console_reporter` instead of a chart. Handy on a remote box over SSH.
"""

from __future__ import annotations

import argparse

from .config import get_configs
from .data import prepare_data
from .train import train, pick_device
from .events import console_reporter


def main():
    ap = argparse.ArgumentParser(description="Train a magic-math model in the terminal.")
    ap.add_argument("--max-steps", type=int, default=None, help="override the step count")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    ap.add_argument("--save-checkpoints", action="store_true",
                    help="also save the model's weights at each checkpoint step")
    args = ap.parse_args()

    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.save_checkpoints:
        overrides["save_checkpoints"] = True

    model_cfg, train_cfg = get_configs(**overrides)
    device = args.device or pick_device()
    print(f"device: {device}")

    data = prepare_data(train_cfg, data_dir=args.data_dir,
                        on_event=console_reporter, vocab_size=model_cfg.vocab_size)
    train(model_cfg, train_cfg, data, device=device, on_event=console_reporter)


if __name__ == "__main__":
    main()
