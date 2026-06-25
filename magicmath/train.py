"""
The training loop.

Training is a simple idea repeated a few thousand times:
  1. grab a batch of text windows
  2. ask the model to predict the next token at every position
  3. measure how wrong it was (the "loss" — cross-entropy)
  4. nudge every weight a little in the direction that reduces that wrongness
     (backpropagation + the AdamW optimizer)
  5. repeat, slowly lowering the learning rate

Everything else here is bookkeeping: a learning-rate schedule, mixed-precision
for speed on GPUs, gradient clipping for stability, and periodic eval/sampling
so you can watch the model get better in real time.
"""

from __future__ import annotations

import math
import os
import time

import torch

from .config import ModelConfig, TrainConfig, config_summary
from .model import MagicMath
from .data import Batcher
from . import tokenizer as tok_lib


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _amp_settings(device: str):
    """Choose a precision. bf16 on modern GPUs (A100+), fp16 on older ones
    (T4), plain fp32 on CPU. fp16 needs a gradient scaler; bf16 does not."""
    if device != "cuda":
        return False, torch.float32, None
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, None
    from torch.amp import GradScaler
    return True, torch.float16, GradScaler("cuda")


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr down to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def _make_optimizer(model: MagicMath, cfg: TrainConfig, device: str):
    # Weight-decay the matrices (2D params) but not the 1D norm/bias vectors —
    # a small, standard refinement that helps generalization.
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device == "cuda"
    try:
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=fused)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def estimate_val_loss(model, batcher: Batcher, cfg: TrainConfig, autocast_ctx) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = batcher.get_batch(cfg.batch_size)
        with autocast_ctx():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _sample_text(model, tok, cfg: TrainConfig, device: str) -> str:
    ids = tok_lib.encode(tok, cfg.sample_prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=cfg.sample_tokens, temperature=0.7,
                         top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3,
                         eos_id=tok_lib.eos_id(tok))
    return tok_lib.decode(tok, out[0].tolist())


def checkpoint_steps(max_steps: int, cadence: int) -> list[int]:
    """Steps at which to sample/checkpoint: geometrically *dense early* (1, 2, 4,
    8, …) so the fast noise→grammar transition is visible, then every `cadence`
    steps once the gaps would exceed it."""
    steps = {0, max_steps - 1}
    k = 1
    while k < cadence and k < max_steps:
        steps.add(k)
        k *= 2
    s = cadence
    while s < max_steps:
        steps.add(s)
        s += cadence
    return sorted(steps)


class EMA:
    """Exponential moving average of the weights. The smoothed model is usually a
    little better than the final raw one, basically for free: we train normally
    and keep a shadow copy that trails the live weights, then save the shadow."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])


def train(model_cfg: ModelConfig, train_cfg: TrainConfig, data: dict,
          device: str | None = None, on_event=None) -> dict:
    """Train a model. `data` is the dict returned by data.prepare_data().
    Returns a dict with the model, tokenizer, and checkpoint path."""

    device = device or pick_device()
    torch.manual_seed(train_cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = data["tokenizer"]
    model = MagicMath(model_cfg).to(device)
    if on_event:
        cfg = config_summary(model_cfg, train_cfg)
        cfg["n_params"] = model.num_params()
        cfg["device"] = device
        on_event({"type": "config", "config": cfg})

    if train_cfg.compile and device == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    use_amp, amp_dtype, scaler = _amp_settings(device)

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        import contextlib
        return contextlib.nullcontext()

    optimizer = _make_optimizer(model, train_cfg, device)
    train_batcher = Batcher(data["train_bin"], train_cfg.seq_len, device)
    val_batcher = Batcher(data["val_bin"], train_cfg.seq_len, device)

    tokens_per_step = train_cfg.batch_size * train_cfg.seq_len * train_cfg.grad_accum
    sample_steps = set(checkpoint_steps(train_cfg.max_steps, train_cfg.sample_interval))
    ema = EMA(model, train_cfg.ema_decay) if train_cfg.ema_decay > 0 else None
    model.train()
    t0 = time.time()
    last_t = t0
    last_tokens = 0
    last_val = None   # most recent validation loss, attached to each sample

    for step in range(train_cfg.max_steps):
        lr = _lr_at(step, train_cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # --- one optimizer step (with gradient accumulation) ---------------
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = train_batcher.get_batch(train_cfg.batch_size)
            with autocast_ctx():
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if ema is not None:
            ema.update(model)

        # --- logging -------------------------------------------------------
        if on_event and (step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1):
            now = time.time()
            tokens_seen = (step + 1) * tokens_per_step
            tok_per_s = (tokens_seen - last_tokens) / max(1e-6, now - last_t)
            last_t, last_tokens = now, tokens_seen
            eta = (train_cfg.max_steps - step - 1) * tokens_per_step / max(1.0, tok_per_s)
            on_event({"type": "step", "step": step, "loss": loss_accum, "lr": lr,
                      "tok_per_s": tok_per_s, "eta_s": eta,
                      "tokens": tokens_seen, "max_steps": train_cfg.max_steps})

        # --- checkpoint: eval, sample what the model writes, optionally save
        #     weights. Denser early (see sample_steps) so you can watch the model
        #     go from noise to sentences in the first few hundred steps. --------
        if on_event and step in sample_steps:
            last_val = estimate_val_loss(model, val_batcher, train_cfg, autocast_ctx)
            on_event({"type": "eval", "step": step, "val_loss": last_val})
            text = _sample_text(model, tok, train_cfg, device)
            ev = {"type": "sample", "step": step, "text": text, "val_loss": last_val}
            if train_cfg.save_checkpoints:
                ev["ckpt"] = save_checkpoint(model, model_cfg, train_cfg, data, device, step=step)
            on_event(ev)
            last_t = time.time()   # don't count eval/sample time against throughput

    elapsed = time.time() - t0
    if ema is not None:
        ema.copy_to(model)         # the final saved model is the smoothed (EMA) one
    ckpt = save_checkpoint(model, model_cfg, train_cfg, data, device)
    if on_event:
        on_event({"type": "done", "final_loss": loss_accum,
                  "elapsed_s": elapsed, "ckpt": ckpt})

    return {"model": model, "tokenizer": tok, "device": device, "ckpt": ckpt}


def save_checkpoint(model, model_cfg, train_cfg, data, device, step=None) -> str:
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    name = f"model-{train_cfg.preset}.pt" if step is None \
        else f"model-{train_cfg.preset}-step{step}.pt"
    path = os.path.join(train_cfg.out_dir, name)
    raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile if used
    torch.save({
        "model_state": raw.state_dict(),
        "model_config": vars(model_cfg),
        "preset": train_cfg.preset,
        "tokenizer_path": data["tokenizer_path"],
    }, path)
    return path
