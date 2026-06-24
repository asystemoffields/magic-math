"""
The data pipeline — from text on the internet to batches of token ids.

Stages (all driven by `prepare_data` below):
  1. download   stream N bytes of TinyStories off the Hugging Face hub
  2. tokenizer  train a byte-level BPE vocabulary on that text
  3. encode     turn the text into a flat array of uint16 token ids on disk
                (train.bin / val.bin) — the classic "nanoGPT" data format
  4. batch      at train time, grab random windows out of those arrays

TinyStories (Eldan & Li, 2023) is a synthetic dataset of simple children's
stories written with a small vocabulary. It exists precisely so that *tiny*
models can learn to produce fluent, coherent English — perfect for us.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

from . import tokenizer as tok_lib

# TinyStories V2 (regenerated with GPT-4) — plain text, stories separated by
# the <|endoftext|> marker. We stream these and stop after `data_bytes`.
_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN_URL = f"{_BASE}/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{_BASE}/TinyStoriesV2-GPT4-valid.txt"

# uint16 holds ids up to 65535 — plenty for our vocab sizes.
_DTYPE = np.uint16


def stream_download(url: str, out_path: str, max_bytes: int, on_event=None) -> None:
    """Download up to `max_bytes` of `url` to `out_path`, then stop.

    We just read the response a megabyte at a time and quit early — no need for
    the full multi-gigabyte file. If a sufficiently large file already exists we
    skip the download (so re-running is cheap)."""
    if os.path.exists(out_path) and os.path.getsize(out_path) >= max_bytes * 0.98:
        if on_event:
            on_event({"type": "phase", "phase": "download",
                      "msg": f"reusing cached {os.path.basename(out_path)}"})
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if on_event:
        on_event({"type": "phase", "phase": "download",
                  "msg": f"downloading ~{max_bytes // 1_000_000} MB of TinyStories"})

    req = urllib.request.Request(url, headers={"User-Agent": "magic-math/1.0"})
    got = 0
    chunk = 1 << 20  # 1 MB
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
        while got < max_bytes:
            data = resp.read(min(chunk, max_bytes - got))
            if not data:
                break
            f.write(data)
            got += len(data)
            if on_event and got % (16 << 20) < chunk:
                on_event({"type": "progress", "phase": "download",
                          "done": got, "total": max_bytes,
                          "msg": f"{got // 1_000_000} / {max_bytes // 1_000_000} MB"})
    # Trim a possibly-truncated final story so we don't feed half a word.
    _trim_to_last_marker(out_path)


def _trim_to_last_marker(path: str) -> None:
    """Cut the file back to the last complete <|endoftext|> boundary."""
    marker = tok_lib.END_OF_TEXT.encode()
    with open(path, "rb") as f:
        data = f.read()
    idx = data.rfind(marker)
    if idx != -1:
        with open(path, "wb") as f:
            f.write(data[: idx + len(marker)])


def encode_to_bin(text_path: str, tok, out_path: str, on_event=None) -> int:
    """Encode every story in `text_path` and append a flat uint16 stream to
    `out_path`. An end-of-text id is inserted between stories so the model
    learns where stories begin and end. Returns the total token count."""
    eot = tok_lib.eos_id(tok)
    total = 0
    batch: list[str] = []
    BATCH = 1000

    if os.path.exists(out_path):
        os.remove(out_path)
    if on_event:
        on_event({"type": "phase", "phase": "encode",
                  "msg": f"tokenizing {os.path.basename(text_path)} -> {os.path.basename(out_path)}"})

    def flush(stories, f):
        nonlocal total
        if not stories:
            return
        encs = tok.encode_batch(stories)
        ids = []
        for e in encs:
            ids.extend(e.ids)
            ids.append(eot)
        arr = np.asarray(ids, dtype=_DTYPE)
        arr.tofile(f)
        total += arr.size

    with open(out_path, "ab") as f:
        for story in tok_lib.iter_stories(text_path):
            batch.append(story)
            if len(batch) >= BATCH:
                flush(batch, f)
                batch = []
                if on_event and (total // 1_000_000) != ((total - 1) // 1_000_000):
                    on_event({"type": "progress", "phase": "encode",
                              "msg": f"{total // 1_000_000}M tokens"})
        flush(batch, f)

    if on_event:
        on_event({"type": "phase", "phase": "encode",
                  "msg": f"{total:,} tokens -> {out_path}"})
    return total


def prepare_data(train_cfg, data_dir: str = "data", on_event=None,
                 vocab_size: int | None = None) -> dict:
    """Run the whole pipeline and return paths + the tokenizer + token counts.

    `vocab_size` should match the model you're about to train. If omitted, it's
    taken from the preset. Cache filenames include the vocab so two runs with
    different vocabularies never collide.

    Idempotent: cached downloads, tokenizer, and bins are reused on re-run.
    """
    if vocab_size is None:
        vocab_size = _vocab_from_cfg(train_cfg)

    os.makedirs(data_dir, exist_ok=True)
    train_txt = os.path.join(data_dir, "train.txt")
    valid_txt = os.path.join(data_dir, "valid.txt")
    tag = f"{train_cfg.preset}-v{vocab_size}"
    tok_path = os.path.join(data_dir, f"tokenizer-{tag}.json")
    train_bin = os.path.join(data_dir, f"train-{tag}.bin")
    val_bin = os.path.join(data_dir, f"val-{tag}.bin")

    # 1. download (valid set capped small — we only need it to measure loss)
    stream_download(TRAIN_URL, train_txt, train_cfg.data_bytes, on_event)
    stream_download(VALID_URL, valid_txt, min(train_cfg.data_bytes, 10_000_000), on_event)

    # 2. tokenizer — train on a sample of the corpus (cap so it stays fast)
    if os.path.exists(tok_path):
        tok = tok_lib.load_tokenizer(tok_path)
        if on_event:
            on_event({"type": "phase", "phase": "tokenizer", "msg": "reusing cached tokenizer"})
    else:
        sample_cap = min(train_cfg.data_bytes, 120_000_000)
        tok = tok_lib.train_tokenizer(
            _sample_stories(train_txt, sample_cap),
            vocab_size=vocab_size,
            save_path=tok_path,
            on_event=on_event,
        )

    # 3. encode to bins (reuse if present)
    n_train = _bin_tokens(train_bin) or encode_to_bin(train_txt, tok, train_bin, on_event)
    n_val = _bin_tokens(val_bin) or encode_to_bin(valid_txt, tok, val_bin, on_event)

    return {
        "tokenizer": tok,
        "tokenizer_path": tok_path,
        "train_bin": train_bin,
        "val_bin": val_bin,
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
    }


def _vocab_from_cfg(train_cfg) -> int:
    # The vocab size lives on the *model* config, but prepare_data only takes
    # the train config in the notebook flow; we stash it on train_cfg via the
    # preset table. Fall back to a sensible default.
    from .config import get_configs
    model_cfg, _ = get_configs(train_cfg.preset)
    return model_cfg.vocab_size


def _sample_stories(text_path: str, max_bytes: int):
    """Yield stories until we've seen ~max_bytes of text (tokenizer training)."""
    seen = 0
    for story in tok_lib.iter_stories(text_path):
        yield story
        seen += len(story)
        if seen >= max_bytes:
            return


def _bin_tokens(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return os.path.getsize(path) // np.dtype(_DTYPE).itemsize
    return 0


# ----------------------------------------------------------------------------
# Batching
# ----------------------------------------------------------------------------
class Batcher:
    """Serves random (input, target) windows from a .bin file via a memmap, so
    we never load the whole dataset into RAM. target is input shifted by one —
    the model's job at every position is to predict the *next* token."""

    def __init__(self, bin_path: str, seq_len: int, device: str):
        self.data = np.memmap(bin_path, dtype=_DTYPE, mode="r")
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size: int):
        import torch
        hi = len(self.data) - self.seq_len - 1
        ix = np.random.randint(0, hi, size=batch_size)
        x = np.stack([self.data[i: i + self.seq_len].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1: i + 1 + self.seq_len].astype(np.int64) for i in ix])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
