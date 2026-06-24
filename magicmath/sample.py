"""
Sampling — load a trained checkpoint and generate text from a prompt.

Used three ways: from the command line (`python -m magicmath.sample ...`), from
the notebook's final cell, and by the web dashboard's "talk to your model" box.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch
import torch.nn.functional as F

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


@torch.no_grad()
def generate_stream(model, tok, prompt: str, max_new_tokens: int = 200,
                    temperature: float = 0.8, top_k: int = 200, device=None):
    """Like `generate`, but a generator: it yields the text one chunk at a time
    as each token is produced, so a UI (or the notebook) can show the model
    writing live instead of waiting for the whole completion.

    We decode the full sequence each step and yield only the new suffix — that
    keeps spacing correct (byte-level tokens don't decode cleanly one at a time).
    """
    m = getattr(model, "_orig_mod", model)
    device = device or next(m.parameters()).device
    eos = tok_lib.eos_id(tok)
    ids = tok_lib.encode(tok, prompt) if prompt else [eos]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    m.eval()

    text_so_far = tok_lib.decode(tok, ids)
    if text_so_far:
        yield text_so_far                 # emit the prompt first, then continue
    for _ in range(max_new_tokens):
        idx_cond = x[:, -m.cfg.max_seq_len:]
        logits, _ = m(idx_cond)
        logits = logits[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        x = torch.cat([x, nxt], dim=1)
        full = tok_lib.decode(tok, x[0].tolist())
        delta, text_so_far = full[len(text_so_far):], full
        if delta:
            yield delta
        if nxt.item() == eos:
            break


def list_checkpoints(out_dir: str, preset: str):
    """Return [(step, path), ...] sorted by step for the per-step checkpoints
    written when save_checkpoints=True (out/model-<preset>-step<N>.pt)."""
    found = []
    for p in glob.glob(os.path.join(out_dir, f"model-{preset}-step*.pt")):
        m = re.search(r"step(\d+)\.pt$", p)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def compare_checkpoints(ckpt_a: str, ckpt_b: str, prompt: str,
                        max_new_tokens: int = 120, temperature: float = 0.8,
                        device: str | None = None):
    """Load two checkpoints and generate from the *same* prompt with each — the
    clearest before/after of what training bought. Returns (text_a, text_b)."""
    model_a, tok, dev = load_model(ckpt_a, device=device)
    text_a = generate(model_a, tok, prompt, max_new_tokens, temperature, device=dev)
    model_b, _, _ = load_model(ckpt_b, device=device)
    text_b = generate(model_b, tok, prompt, max_new_tokens, temperature, device=dev)
    return text_a, text_b


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
