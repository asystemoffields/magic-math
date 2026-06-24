"""
Generates notebooks/magic_math.ipynb from plain Python here, so the notebook
stays diff-able and easy to edit. Run:  python scripts/build_notebook.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "magic_math.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines, form=False):
    meta = {"cellView": "form"} if form else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None,
            "outputs": [], "source": "\n".join(lines)}


CELLS = [
    md(
        "# magic&#8209;math — train a language model from scratch 🪄",
        "",
        "This notebook trains a **small but genuinely modern** language model "
        "from nothing — no pretrained weights anywhere — until it can write its "
        "own little stories. Same architecture family as Llama 3 and Mistral, "
        "just shrunk to ~12M parameters so a full run finishes in minutes.",
        "",
        "**To run it:** in the menu go to **Runtime → Change runtime type → "
        "T4 GPU** (free), then **Runtime → Run all**. That's the whole thing. "
        "Scroll down and watch the loss curve fall and the samples get more "
        "coherent.",
        "",
        "You don't need any machine-learning background. Each step explains "
        "what it's doing and why, building up from tokens to a trained model.",
    ),
    md(
        "## How a language model works, before we build one",
        "",
        "Six ideas. That's the whole game:",
        "",
        "1. **Tokens.** Text is chopped into a few thousand recurring chunks "
        "(`the`, ` said`, `ing`). Each chunk is just an integer id. The model "
        "only ever sees integers.",
        "2. **Vectors.** Every token id is looked up as a vector — a point in a "
        "few-hundred-dimensional space. Similar tokens end up near each other. "
        "(This is the part you already know.)",
        "3. **Attention.** At each position, the model looks back over the "
        "earlier tokens and pulls in the ones that are relevant — *this* is how "
        "it uses context.",
        "4. **Thinking.** Between attention steps, a little feed-forward network "
        "transforms each token's vector on its own.",
        "5. **Predict the next token.** Stack those a few times, and the final "
        "vector at each position is turned into a probability over *what token "
        "comes next*.",
        "6. **Training.** Show it real text, measure how surprised it was by the "
        "true next token (the *loss*), and nudge every number a hair to be less "
        "surprised. Repeat a few thousand times. That's it.",
        "",
        "Everything below is those six ideas in code.",
    ),
    code(
        "# --- setup: pull the code and install the one missing package -------------",
        "import os, sys",
        "if not os.path.exists('magic-math'):",
        "    !git clone -q https://github.com/asystemoffields/magic-math.git",
        "else:",
        "    !git -C magic-math pull -q",
        "sys.path.insert(0, os.path.abspath('magic-math'))",
        "%pip install -q tokenizers",
        "",
        "import torch",
        "print('PyTorch', torch.__version__)",
        "if torch.cuda.is_available():",
        "    print('GPU:', torch.cuda.get_device_name(0))",
        "else:",
        "    print('No GPU! Go to Runtime > Change runtime type > T4 GPU, then Run all again.')",
    ),
    md(
        "## Pick a size",
        "",
        "| preset | params | what you get | ~time on T4 | ~time on A100 |",
        "|---|---|---|---|---|",
        "| `nano` | ~1.6M | a quick taste; semi-words | ~5 min | ~2 min |",
        "| `small` | ~7M | short coherent fragments | ~30 min | ~10 min |",
        "| `default` | ~12M | actual little stories | ~75 min | ~25 min |",
        "",
        "On a **free T4**, start with `small`. If you have **Colab Pro with an "
        "A100**, use `default`. `nano` is great just to see the whole thing run "
        "through in a couple of minutes.",
    ),
    code(
        "from magicmath.config import get_configs, config_summary",
        "import json",
        "",
        'PRESET = "small"  # @param ["nano", "small", "default"]',
        "",
        "# save_checkpoints=True keeps the model's weights at every checkpoint, so",
        "# Step 6 can compare an early one against a late one.",
        "model_cfg, train_cfg = get_configs(PRESET, save_checkpoints=True)",
        "print(json.dumps(config_summary(model_cfg, train_cfg), indent=2))",
        form=True,
    ),
    md(
        "## Step 1 — build a vocabulary and prepare the data",
        "",
        "We stream a chunk of **TinyStories** (a dataset of simple, synthetic "
        "children's stories, built precisely so small models can learn fluent "
        "English), then train our *own* tokenizer on it with byte-pair encoding "
        "— starting from raw bytes and repeatedly merging the most common "
        "adjacent pair into a new token. Finally we encode everything into a "
        "flat array of token ids on disk.",
        "",
        "This cell downloads, builds the vocabulary, and tokenizes. Cached on "
        "re-run.",
    ),
    code(
        "from magicmath.data import prepare_data",
        "from magicmath.notebook import NotebookReporter",
        "",
        "reporter = NotebookReporter()   # we reuse this for the live chart later",
        "data = prepare_data(train_cfg, on_event=reporter, vocab_size=model_cfg.vocab_size)",
        "print('train tokens:', f\"{data['n_train_tokens']:,}\")",
    ),
    md(
        "## Step 2 — meet the model",
        "",
        "Here's the actual network we're about to train. Notice how small it is "
        "— a handful of repeated **blocks**, each with an attention layer and a "
        "feed-forward (`SwiGLU`) layer, wrapped in `RMSNorm`. The full, heavily "
        "commented source is in [`magicmath/model.py`]"
        "(https://github.com/asystemoffields/magic-math/blob/main/magicmath/model.py).",
    ),
    code(
        "from magicmath.model import MagicMath",
        "_m = MagicMath(model_cfg)",
        "print(_m)",
        "print(f'\\n{_m.num_params()/1e6:.2f}M parameters "
        "({_m.num_params(non_embedding=True)/1e6:.2f}M outside the embedding table)')",
        "del _m  # train() builds its own; this was just to look",
    ),
    md(
        "## Step 3 — train, and watch it learn",
        "",
        "Now the loop from idea #6. The chart updates live: **blue** is training "
        "loss (lower = less surprised by the next token), **orange** is validation "
        "loss on held-out text.",
        "",
        "**Below the chart**, the model's writing is sampled at each *checkpoint* — "
        "starting at **step 0 (pure noise)** and every few hundred steps after — and "
        "the snapshots *accumulate*. When it finishes, scroll that list to see the "
        "whole arc from gibberish to little sentences. That transition is the single "
        "most instructive thing here.",
        "",
        "*(Want the model's actual weights saved at each checkpoint too, so you can "
        "reload an early one? Re-make the configs as "
        "`get_configs(PRESET, save_checkpoints=True)` and each snapshot also writes "
        "`out/model-…-step<N>.pt`.)*",
    ),
    code(
        "from magicmath.train import train",
        "result = train(model_cfg, train_cfg, data, on_event=reporter)",
    ),
    md(
        "## Step 4 — talk to your model",
        "",
        "It's trained. Give it a prompt and let it continue. Try changing the "
        "prompt, or the `temperature` (higher = more random/creative, lower = "
        "more predictable).",
    ),
    code(
        "from magicmath.sample import generate_stream",
        "",
        'prompt = "Once upon a time, a little robot"  # @param {type:"string"}',
        "",
        "# stream the tokens in as the model writes them",
        "for delta in generate_stream(result['model'], result['tokenizer'], prompt,",
        "                             max_new_tokens=250, temperature=0.8):",
        "    print(delta, end='', flush=True)",
        form=True,
    ),
    md(
        "## Step 5 — look inside your model",
        "",
        "Three quick lenses into what it actually learned (each is just a few "
        "lines, in `magicmath/lenses.py`):",
        "",
        "- **tokens** — how your text is chopped into the integers the model sees.",
        "- **the model's options** — its top candidates for the *next* token and "
        "their probabilities. Lowering `temperature` piles the probability onto "
        "the top few; that's all temperature does.",
        "- **embedding neighbours** — which tokens the model placed *near each "
        "other* in its vector space. It worked these out purely from predicting "
        "text; nobody told it which words are related.",
    ),
    code(
        "from magicmath import lenses",
        "tok = result['tokenizer']; model = result['model']",
        "",
        'ids, pieces = lenses.tokenize_view(tok, "Once upon a time, a dragon")',
        'print(f"{len(ids)} tokens:", " | ".join(repr(p) for p in pieces))',
        "",
        "print(\"\\nTop next-token guesses after 'Once upon a':\")",
        'for t, p in lenses.next_token_table(model, tok, "Once upon a", k=8):',
        '    print(f"  {p:6.1%}  {t!r}")',
        "",
        'print("\\nTokens nearest the word dog in embedding space:")',
        'for t, s in lenses.nearest_tokens(model, tok, " dog", k=8):',
        '    print(f"  {s:+.3f}  {t!r}")',
    ),
    md(
        "## Step 6 — compare two checkpoints (early vs late)",
        "",
        "Because we saved the model at every checkpoint, you can load an **early** "
        "one and a **late** one and give them the *same* prompt — the clearest "
        "before/after of what all that training actually bought.",
    ),
    code(
        "from magicmath import sample",
        "ckpts = sample.list_checkpoints(train_cfg.out_dir, PRESET)",
        'print("saved checkpoints at steps:", [s for s, _ in ckpts])',
        "",
        'prompt = "The cat sat on the"  # @param {type:"string"}',
        "early_step, early_path = ckpts[0]    # first saved — near step 0, ~noise",
        "late_step,  late_path  = ckpts[-1]   # last saved — fully trained",
        "a, b = sample.compare_checkpoints(early_path, late_path, prompt, max_new_tokens=120)",
        'print(f"\\n=== step {early_step} (early) ===\\n{a}")',
        'print(f"\\n=== step {late_step} (late) ===\\n{b}")',
        form=True,
    ),
    md(
        "## What makes this model *modern*",
        "",
        "This is the same architecture family as today's open models (Llama 3, "
        "Mistral, Qwen), just small. Each component is the current standard "
        "choice, and each one buys real quality:",
        "",
        "- **RMSNorm** — normalize each vector by its length (no mean, no bias). "
        "Keeps training stable, cheaply.",
        "- **RoPE** (rotary position embeddings) — encode a token's position by "
        "*rotating* its query/key vectors, which makes attention naturally "
        "relative and length-flexible.",
        "- **Grouped-Query Attention** — share key/value heads across query "
        "heads, which is what lets models keep a small memory cache at inference.",
        "- **SwiGLU** — a *gated* feed-forward layer that gets more out of each "
        "parameter.",
        "- **No biases, pre-normalization, tied embeddings** — small modern "
        "hygiene that makes training stable and parameter-efficient.",
        "",
        "All of it lives, commented, in `magicmath/model.py`.",
        "",
        "## Things to try (hand these to Claude)",
        "",
        "This repo is meant to be poked at. Open it in **Claude Code**, or paste "
        "files into a Claude conversation, and try prompts like:",
        "",
        "- *“In `magicmath/config.py`, make a new preset that's deeper (more "
        "layers) but narrower. What happens to the loss?”*",
        "- *“Explain what `apply_rope` in `model.py` is doing, line by line.”*",
        "- *“Turn off weight tying and qk-norm and rerun — did it matter?”*",
        "- *“Add top-p (nucleus) sampling to `sample.py`.”*",
        "",
        "Re-run this notebook after any change. Full repo: "
        "[github.com/asystemoffields/magic-math]"
        "(https://github.com/asystemoffields/magic-math).",
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print("wrote", OUT, "with", len(CELLS), "cells")
