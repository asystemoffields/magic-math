"""
Configuration for the model and the training run.

Everything that defines *what* we build and *how hard* we train lives here, in
two small dataclasses plus `get_configs()`, which returns the model + training
config (with optional keyword overrides). If you only read one file to
understand the knobs, read this one.

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

    preset: str = "default"   # internal tag used in cache / checkpoint filenames

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
    save_checkpoints: bool = False  # also save the model's weights at each sample step
                                    # (out/model-<preset>-step<N>.pt), not just the final one

    def __post_init__(self):
        assert self.seq_len <= 8192


# ----------------------------------------------------------------------------
# The model — one configuration: a ~12M-parameter Llama-style decoder
# ----------------------------------------------------------------------------
def get_configs(**overrides):
    """Return (ModelConfig, TrainConfig) for the model — a ~12M-parameter,
    Llama-style decoder trained on TinyStories.

    Pass keyword overrides to tweak any field of either config, e.g.
    get_configs(save_checkpoints=True) or get_configs(max_steps=3000, d_model=512).
    """
    model_spec = dict(vocab_size=8192, d_model=384, n_layers=6, n_heads=6,
                      n_kv_heads=2, max_seq_len=512)
    train_spec = dict(data_bytes=250_000_000, batch_size=64, max_steps=6000,
                      warmup_steps=200, lr=6e-4, min_lr=6e-5,
                      log_interval=10, eval_interval=750, sample_interval=750)

    # Merge all overrides into the spec dicts *before* constructing the configs,
    # so we never build an intermediate, half-overridden config that trips a
    # consistency check (e.g. d_model set but n_heads not yet).
    model_fields = set(ModelConfig.__dataclass_fields__)
    train_fields = set(TrainConfig.__dataclass_fields__)
    for k, v in overrides.items():
        if k in model_fields:
            model_spec[k] = v
        elif k in train_fields:
            train_spec[k] = v
        else:
            raise KeyError(f"{k!r} is not a field of ModelConfig or TrainConfig")

    model = ModelConfig(**model_spec)
    # training context length follows the model's window unless overridden
    train_spec.setdefault("seq_len", model.max_seq_len)
    train = TrainConfig(**train_spec)
    if train.seq_len > model.max_seq_len:
        train = replace(train, seq_len=model.max_seq_len)
    return model, train


def config_summary(model: ModelConfig, train: TrainConfig) -> dict:
    """A compact, JSON-friendly dict describing the run (for the UI header)."""
    return {"model": asdict(model), "train": asdict(train)}
