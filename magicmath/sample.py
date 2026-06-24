"""
Sampling — load a trained checkpoint and generate text from a prompt.

Used three ways: from the command line (`python -m magicmath.sample ...`), from
the notebook's final cell, and by the web dashboard's "talk to your model" box.
"""

from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .model import MagicMath
from . import tokenizer as tok_lib


def load_model(ckpt_path: str, device: str | None = None):
    """Rebuild the model from a checkpoint and return (model, tokenizer, device)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = MagicMath(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tok = tok_lib.load_tokenizer(ckpt["tokenizer_path"])
    return model, tok, device


def generate(model, tok, prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 200, device: str | None = None) -> str:
    device = device or next(model.parameters()).device
    ids = tok_lib.encode(tok, prompt) if prompt else [tok_lib.eos_id(tok)]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, eos_id=tok_lib.eos_id(tok))
    return tok_lib.decode(tok, out[0].tolist())


def main():
    ap = argparse.ArgumentParser(description="Generate text from a trained magic-math model.")
    ap.add_argument("--ckpt", default="out/model-default.pt")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    args = ap.parse_args()

    model, tok, device = load_model(args.ckpt)
    print(generate(model, tok, args.prompt, args.tokens, args.temperature, args.top_k, device))


if __name__ == "__main__":
    main()
