"""
Configuration for the model and the training run.

Everything that defines *what* we build and *how hard* we train lives here, in
two small dataclasses, plus three named presets you can pick from. If you only
read one file to understand the knobs, read this one.

Nothing here imports torch, so it is cheap to import and easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Optional


# ----------------------------------------------------------------------------
# Model shape
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """The architecture. These numbers fully determine the parameter count."""

    vocab_size: int = 8192          # size of the tokenizer's vocabulary
    d_model: int = 384              # the "width" of the model (residual stream)
    n_layers: int = 6               # number of transformer blocks (the "depth")
    n_heads: int = 6                # number of *query* attention heads
    n_kv_heads: int = 2             # number of *key/value* heads  -> GQA when < n_heads
    d_ff: Optional[int] = None      # SwiGLU hidden size; if None, derived from d_model
    max_seq_len: int = 512          # the context window (in tokens)
    rope_base: float = 10000.0      # RoPE frequency base (theta)
    qk_norm: bool = False           # RMSNorm on Q and K (OLMo2 / Gemma2 trick) — off by default
    tie_embeddings: bool = True     # share the input embedding with the output head

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        if self.d_ff is None:
            # SwiGLU has 3 projections instead of 2, so the usual 4*d_model FFN is
            # scaled by 2/3 to keep the parameter budget comparable, then rounded
            # up to a multiple of 64 (kinder to the GPU). This is the Llama recipe.
            hidden = int(8 / 3 * self.d_model)
            self.d_ff = (hidden + 63) // 64 * 64

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ----------------------------------------------------------------------------
# Training recipe
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Everything about the *run*: data, optimizer, schedule, logging."""

    preset: str = "default"

    # --- data ---------------------------------------------------------------
    # We stream this many bytes of TinyStories text off the Hugging Face hub.
    # ~4 bytes per token, so 250 MB ≈ 60M tokens of training data.
    data_bytes: int = 250_000_000
    seq_len: int = 512              # training context length (must be <= max_seq_len)
    batch_size: int = 64            # sequences per micro-batch
    grad_accum: int = 1             # micro-batches per optimizer step (effective batch = batch*accum)

    # --- optimizer / schedule ----------------------------------------------
    max_steps: int = 6000           # number of optimizer steps
    lr: float = 6e-4                # peak learning rate
    min_lr: float = 6e-5            # final learning rate (cosine floor)
    warmup_steps: int = 200         # linear warmup before the cosine decay
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- logging / eval -----------------------------------------------------
    log_interval: int = 10          # emit a 'step' event every N steps
    eval_interval: int = 750        # measure validation loss every N steps
    eval_iters: int = 50            # batches to average for each validation estimate
    sample_interval: int = 750      # generate a text sample every N steps
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 200        # length of those samples

    # --- misc ---------------------------------------------------------------
    seed: int = 1337
    out_dir: str = "out"
    compile: bool = False           # torch.compile — faster on Linux, flaky on Windows; off by default

    def __post_init__(self):
        assert self.seq_len <= 8192


# ----------------------------------------------------------------------------
# Presets — the three sizes you'll actually pick from
# ----------------------------------------------------------------------------
# Times are rough, for the full run, and depend heavily on the GPU.
#   nano    ~1.6M params   quick taste / smoke test     ~2 min A100   ~5 min T4
#   small   ~7M params     coherent-ish, free Colab     ~10 min A100  ~30 min T4
#   default ~12M params    clearly coherent stories     ~25 min A100  ~75 min T4
_PRESETS = {
    "nano": dict(
        model=dict(vocab_size=2048, d_model=128, n_layers=4, n_heads=4,
                   n_kv_heads=2, max_seq_len=128),
        train=dict(data_bytes=20_000_000, seq_len=128, batch_size=32, max_steps=1000,
                   warmup_steps=50, lr=1e-3, min_lr=1e-4,
                   log_interval=10, eval_interval=200, sample_interval=200),
    ),
    "small": dict(
        model=dict(vocab_size=8192, d_model=256, n_layers=6, n_heads=8,
                   n_kv_heads=2, max_seq_len=256),
        train=dict(data_bytes=120_000_000, seq_len=256, batch_size=64, max_steps=4000,
                   warmup_steps=150, lr=8e-4, min_lr=8e-5,
                   log_interval=10, eval_interval=500, sample_interval=500),
    ),
    "default": dict(
        model=dict(vocab_size=8192, d_model=384, n_layers=6, n_heads=6,
                   n_kv_heads=2, max_seq_len=512),
        train=dict(data_bytes=250_000_000, seq_len=512, batch_size=64, max_steps=6000,
                   warmup_steps=200, lr=6e-4, min_lr=6e-5,
                   log_interval=10, eval_interval=750, sample_interval=750),
    ),
}


def get_configs(preset: str = "default", **overrides):
    """Return (ModelConfig, TrainConfig) for a named preset.

    Any keyword overrides are applied to *whichever* config defines that field,
    so e.g. get_configs("nano", max_steps=50, d_model=64) just works.
    """
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(_PRESETS)}")
    spec = _PRESETS[preset]
    model = ModelConfig(**spec["model"])
    train = TrainConfig(preset=preset, **spec["train"])
    # training context length follows the model's window
    train = replace(train, seq_len=model.max_seq_len)

    model_fields = set(ModelConfig.__dataclass_fields__)
    train_fields = set(TrainConfig.__dataclass_fields__)
    for k, v in overrides.items():
        if k in model_fields:
            model = replace(model, **{k: v})
        elif k in train_fields:
            train = replace(train, **{k: v})
        else:
            raise KeyError(f"{k!r} is not a field of ModelConfig or TrainConfig")
    # re-run derivations after overrides
    model = ModelConfig(**{f: getattr(model, f) for f in model_fields if f != "d_ff"})
    if train.seq_len > model.max_seq_len:
        train = replace(train, seq_len=model.max_seq_len)
    return model, train


def config_summary(model: ModelConfig, train: TrainConfig) -> dict:
    """A compact, JSON-friendly dict describing the run (for the UI header)."""
    return {"model": asdict(model), "train": asdict(train)}
