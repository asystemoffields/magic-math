"""
The model — a small, *modern* decoder-only transformer.

This is the same architecture family as Llama 3, Mistral and Qwen, just shrunk.
Each component below is the current standard choice; together they're what lets a
model this size train stably and write coherent text. The one-liners say what
each part buys you — the full reasoning is in the comments where it's defined.

  RMSNorm                  keep every vector at a sane size — simple and cheap
  RoPE (rotary)            encode token position by *rotating* vectors -> relative
  grouped-query attention  share key/value heads -> a much smaller memory cache
  SwiGLU MLP               a *gated* feed-forward layer -> more quality per param
  no biases, pre-norm      lean and stable to train
  tied embeddings          one table for input and output -> fewer parameters

Read top to bottom: RMSNorm -> RoPE -> Attention -> SwiGLU -> Block -> Model.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ----------------------------------------------------------------------------
# RMSNorm — normalize the residual stream to a fixed scale
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """LayerNorm divides by the standard deviation and re-centers with a mean.
    RMSNorm drops the mean entirely: it just divides each vector by its own
    root-mean-square length, then rescales per-channel with a learned weight.
    Cheaper, and in practice just as good. (Used by Llama, Mistral, Gemma, ...)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # do the normalization in fp32 for numerical safety
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# ----------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ----------------------------------------------------------------------------
# Attention on its own is order-blind — it sees a *set* of tokens, not a
# sequence — so something must encode where each token sits. RoPE does this by
# *rotating* each query/key vector by an angle proportional to its position. The
# dot product between a query at position m and a key at position n then depends
# only on (m - n) — i.e. attention becomes naturally *relative*. There are no
# position parameters to learn, and it extrapolates to longer contexts better.
def build_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute the cos/sin tables of shape (max_seq_len, head_dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of x (shape (B, n_heads, T, head_dim)).

    We pair dimension i with dimension i+head_dim/2 and rotate each pair by the
    angle for that position/frequency — the standard 'rotate-half' formulation.
    """
    T = x.shape[-2]
    cos = cos[:T].to(x.dtype)[None, None, :, :]   # (1,1,T,head_dim/2)
    sin = sin[:T].to(x.dtype)[None, None, :, :]
    x1, x2 = x.chunk(2, dim=-1)                    # two halves
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


# ----------------------------------------------------------------------------
# Attention — causal, with RoPE and grouped-query heads
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    """Self-attention where the model decides, per token, which earlier tokens
    to read from. 'Causal' = a token can only look left (at the past).

    Grouped-Query Attention (GQA): we keep `n_heads` query heads but only
    `n_kv_heads` key/value heads, and share each K/V head across a group of
    query heads. This shrinks the KV cache (the thing that dominates memory at
    inference time) with almost no quality loss. n_kv_heads == n_heads recovers
    ordinary multi-head attention; n_kv_heads == 1 is multi-query attention.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            # Normalizing Q and K before the dot product keeps attention logits
            # from blowing up — a stability trick from OLMo2 / Gemma2.
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q = q.transpose(1, 2)  # (B, n_heads,    T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        v = v.transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand the K/V heads to match the number of query heads (GQA).
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Fused, memory-efficient attention (FlashAttention on GPU). is_causal
        # applies the lower-triangular mask for us — no token sees the future.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ----------------------------------------------------------------------------
# SwiGLU MLP — the per-token "thinking" layer
# ----------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """The feed-forward layer, where each token's vector is transformed on its
    own. SwiGLU uses a *gate*: one projection is squashed by SiLU and multiplied
    element-wise into another, then projected back down. The gate lets the layer
    pass or suppress information per channel, which empirically gives more
    quality per parameter than a plain activation. (Llama, PaLM, Mistral, ...)"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ----------------------------------------------------------------------------
# Transformer block — pre-norm, two residual sub-layers
# ----------------------------------------------------------------------------
class Block(nn.Module):
    """A token's vector goes through two refinements, each *added back* to it
    (a residual connection). We normalize *before* each sub-layer (pre-norm),
    which is what makes deep transformers train stably without learning-rate
    warmup gymnastics."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)   # mix across tokens
        x = x + self.mlp(self.mlp_norm(x))               # think within each token
        return x


# ----------------------------------------------------------------------------
# The whole model
# ----------------------------------------------------------------------------
class MagicMath(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: the matrix that turns tokens into vectors is the same
        # matrix (transposed) that turns vectors back into token scores. Saves
        # vocab*d_model params and tends to help small models.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # A standard initialization trick: scale down the projections that write
        # *into* the residual stream by 1/sqrt(2*n_layers), so the residual
        # doesn't grow with depth.
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. If targets given, also return the loss."""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss

        # Inference: we only need scores for the *last* position.
        logits = self.lm_head(x[:, -1:, :])
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=None, top_p=0.9,
                 repetition_penalty=1.15, no_repeat_ngram_size=3,
                 repetition_window=64, eos_id=None):
        """Autoregressive sampling. No KV cache — we re-run the full context each
        step. Slow, but dead simple, and plenty fast for short toy generations.

        The decoding knobs are what make a small model *read* well — and, just as
        importantly, keep it from losing the thread of its own story:
          temperature           <1 = safer/more consistent, >1 = wilder
          top_k / top_p          only sample from the most likely tokens
          no_repeat_ngram_size   forbid repeating any verbatim n-gram (e.g. n=3
                                 blocks "...Jeff, who was Jeff, who was..."). This
                                 kills loops *without* banning a word outright, so
                                 a character can keep its name.
          repetition_penalty     a gentle nudge away from recently-used tokens;
          repetition_window      ...but only over the last `window` tokens, so a
                                 name introduced early in the story isn't
                                 suppressed later (which is what makes a small
                                 model swap "Joe" out for a fresh name mid-story).
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = apply_repetition_penalty(logits, idx, repetition_penalty, repetition_window)
            logits = block_repeat_ngrams(logits, idx, no_repeat_ngram_size)
            if temperature <= 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                logits = filter_top_k_top_p(logits, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        if was_training:
            self.train()
        return idx


def apply_repetition_penalty(logits, idx, penalty, window=0):
    """Damp the logits of recently-used tokens so they're less likely to repeat
    (the CTRL-style penalty). penalty=1.0 is a no-op.

    `window` is what keeps this coherence-friendly: with window>0 only the last
    `window` tokens count, so a name introduced at the start of a story isn't
    penalized for the rest of it. (Penalizing the *whole* history — window=0 —
    is exactly what makes a small model drop "Joe" and grab a fresh name once
    "Joe" has appeared a few times.)"""
    if penalty == 1.0:
        return logits
    for b in range(idx.shape[0]):
        recent = idx[b, -window:] if window else idx[b]
        seen = torch.unique(recent)
        vals = logits[b, seen]
        logits[b, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def block_repeat_ngrams(logits, idx, n):
    """Forbid any token that would complete a verbatim repeat of an n-gram already
    in the sequence (the standard 'no-repeat n-gram' rule). Set those logits to
    -inf. Unlike a blanket penalty this stops loops ("who was Jeff, who was Jeff")
    while leaving the model free to reuse a word in a *new* context — so a
    character can keep its name across the whole story. n<=0 disables it."""
    if not n or n <= 0:
        return logits
    for b in range(idx.shape[0]):
        seq = idx[b].tolist()
        if len(seq) < n:
            continue
        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
        banned = {seq[i + n - 1] for i in range(len(seq) - n + 1)
                  if tuple(seq[i:i + n - 1]) == prefix}
        if banned:
            logits[b, list(banned)] = -float("inf")
    return logits


def filter_top_k_top_p(logits, top_k=None, top_p=None):
    """Keep only the most likely tokens: top_k by count, top_p by cumulative
    probability (nucleus sampling). Everything else gets -inf."""
    if top_k:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k)[0][:, [-1]]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p and top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(ordered, dim=-1).cumsum(dim=-1)
        drop = cum - F.softmax(ordered, dim=-1) > top_p   # keep at least the top token
        ordered = ordered.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, order, ordered)
    return logits
