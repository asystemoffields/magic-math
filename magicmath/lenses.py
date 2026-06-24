"""
Lenses — three small windows into what your trained model actually learned.

None of these change training; they're just fun, cheap ways to poke at the
model afterward. Each is a few lines, runs on CPU, and answers a question a
power user actually wonders about:

  tokenize_view     how does my text become the integers the model sees?
  next_token_table  what is the model's *distribution* over the next token?
                    (this is exactly what `temperature` and top-k act on)
  nearest_tokens    which tokens did the model place near each other in its
                    learned vector space — purely from reading text?
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import tokenizer as tok_lib


def _unwrap(model):
    return getattr(model, "_orig_mod", model)  # in case of torch.compile


def tokenize_view(tok, text: str):
    """Return (ids, pieces): the token ids `text` becomes, and the readable
    string each one stands for. Try it on a rare word vs a common one."""
    enc = tok.encode(text)
    pieces = [tok.decode([i]) for i in enc.ids]
    return enc.ids, pieces


@torch.no_grad()
def next_token_table(model, tok, prompt: str, k: int = 10, temperature: float = 1.0):
    """The top-k candidates for the next token after `prompt`, with the model's
    probability for each. Lower the temperature and the probability mass piles
    onto the top few; raise it and it spreads out — that's all temperature is."""
    m = _unwrap(model)
    device = next(m.parameters()).device
    ids = tok_lib.encode(tok, prompt) if prompt else [tok_lib.eos_id(tok)]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = m(x)                      # (1, 1, vocab) — last-position scores
    logits = logits[0, -1].float()
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, min(k, probs.numel()))
    return [(tok.decode([i]), float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]


@torch.no_grad()
def nearest_tokens(model, tok, word: str, k: int = 8):
    """The k tokens whose embedding vectors point most similarly to `word`'s
    (cosine similarity). The model learned these neighbourhoods on its own, just
    by predicting next tokens — no one told it which words are related.

    Tip: include the leading space, e.g. " dog", since most words are a single
    space-prefixed token. Multi-token words use their first token as the anchor.
    """
    m = _unwrap(model)
    emb = m.tok_emb.weight                # (vocab, d_model) — the embedding table
    ids = tok_lib.encode(tok, word)
    if not ids:
        return []
    anchor = ids[0]
    sims = F.cosine_similarity(emb, emb[anchor][None, :], dim=-1)
    top = torch.topk(sims, min(k + 1, sims.numel()))
    out = []
    for i, s in zip(top.indices.tolist(), top.values.tolist()):
        if i == anchor:
            continue                      # skip the word itself
        out.append((tok.decode([i]), float(s)))
    return out[:k]
