"""
The tokenizer — how raw text becomes the integers the model actually sees.

A language model never sees letters. It sees token ids: integers indexing a
fixed vocabulary. We *train* our own vocabulary on our own data (no pretrained
weights anywhere in this repo) using byte-level Byte-Pair Encoding (BPE):

  1. Start from the 256 raw bytes — so any text is representable, no "unknown".
  2. Repeatedly find the most frequent adjacent pair of tokens and merge it
     into a new token. Common chunks ("the", " said", "ing") become single ids.
  3. Stop at `vocab_size` tokens.

This is exactly how production tokenizers (Llama, Mistral, ...) are built, just
smaller. We lean on the `tokenizers` library (Rust-backed) so training the vocab
takes seconds.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# The single special token that marks a story boundary in TinyStories. We use
# it as both "beginning of text" and "end of text".
END_OF_TEXT = "<|endoftext|>"


def train_tokenizer(text_iter: Iterable[str], vocab_size: int, save_path: str,
                    on_event=None) -> Tokenizer:
    """Train a byte-level BPE tokenizer and save it to `save_path` (JSON)."""
    if on_event:
        on_event({"type": "phase", "phase": "tokenizer",
                  "msg": f"training a {vocab_size}-token BPE vocabulary"})

    tok = Tokenizer(models.BPE(unk_token=None))
    # ByteLevel pre-tokenizer: operate on raw UTF-8 bytes, so every possible
    # input is covered and we never emit an <unk>.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[END_OF_TEXT],
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(text_iter, trainer=trainer)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    tok.save(save_path)
    if on_event:
        on_event({"type": "phase", "phase": "tokenizer",
                  "msg": f"vocabulary ready ({tok.get_vocab_size()} tokens) -> {save_path}"})
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def eos_id(tok: Tokenizer) -> int:
    return tok.token_to_id(END_OF_TEXT)


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids)


def iter_stories(text_path: str) -> Iterator[str]:
    """Yield TinyStories one at a time, splitting on the end-of-text marker,
    without loading the whole file into memory at once."""
    buf = []
    with open(text_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if END_OF_TEXT in line:
                head, _, tail = line.partition(END_OF_TEXT)
                buf.append(head)
                story = "".join(buf).strip()
                if story:
                    yield story
                buf = [tail]
            else:
                buf.append(line)
    tail = "".join(buf).strip()
    if tail:
        yield tail
