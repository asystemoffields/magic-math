# magic-math 🪄

**Train a small but genuinely modern language model from scratch — until it
writes its own little stories.** No pretrained weights, no magic downloads of
someone else's model. You start from random numbers and a pile of text, and a
few minutes later you have a model that strings together coherent English.

It's the same architecture family as Llama 3 and Mistral — just shrunk to about
**12 million parameters** so a full training run finishes in minutes, not days.
And it's built to be *read*: every file explains what it's doing and why,
assuming you know what a vector is but nothing else about machine learning.

Two ways to run it. Pick one:

---

## 🟢 Path A — in your browser (recommended, zero setup)

Everything happens on a free Google Colab GPU. You need nothing installed.

1. Click this badge:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asystemoffields/magic-math/blob/main/notebooks/magic_math.ipynb)
2. In the menu: **Runtime → Change runtime type → T4 GPU** (it's free), then
   **Runtime → Run all**.
3. Scroll down and watch. A loss curve falls, and every minute or so the model
   prints a new story sample that's a little more coherent than the last.

That's the whole thing. When it finishes, the last cell lets you type a prompt
and read what your model writes back.

> On the free **T4**, use the `small` preset (the notebook's default, ~30 min).
> With Colab Pro's **A100**, switch to `default` for noticeably better stories
> (~25 min).

---

## 🔵 Path B — on your own GPU (a live web dashboard)

If you have a machine with an NVIDIA GPU, you can train locally and watch the
exact same live dashboard in your browser. The **only** thing you need installed
first is **Python 3.10+** ([download here](https://www.python.org/downloads/) —
on Windows, tick *“Add Python to PATH”*).

**Windows:** download this repo (green **Code → Download ZIP** button on GitHub,
then unzip), and **double-click `run.bat`**.

**macOS / Linux:** `./run.sh`

That script does everything else for you: it creates a private virtual
environment, installs PyTorch with CUDA (it detects your GPU automatically),
installs the two other small packages, then opens
[http://localhost:8000](http://localhost:8000) with a live training dashboard —
loss chart, throughput, story samples, and a box to chat with your model when
it's done.

Want a quick taste first? `run.bat --preset nano` (a couple of minutes).

<details>
<summary>Prefer to drive it by hand?</summary>

```bash
pip install -r requirements.txt          # torch, numpy, tokenizers
python -m magicmath.web --preset default # opens the dashboard
# or, no web UI, just the terminal:
python -m magicmath.train_cli --preset default
```
(If you have a GPU, install the CUDA build of torch from
[pytorch.org](https://pytorch.org/get-started/locally/) first.)
</details>

---

## What you're actually building (the 60-second version)

A language model is six ideas stacked together. No background needed:

1. **Tokens** — text is split into a few thousand recurring chunks (`the`,
   ` said`, `ing`); each is just an integer id. The model only ever sees ints.
2. **Vectors** — each token id is looked up as a vector, a point in a
   few-hundred-dimensional space. (The part you already know.)
3. **Attention** — at each position the model looks back over earlier tokens and
   pulls in the relevant ones. This is how it uses context.
4. **A little feed-forward network** transforms each token's vector on its own.
5. **Predict the next token** — stack #3 and #4 a few times, and the final vector
   becomes a probability distribution over *what comes next*.
6. **Training** — show it real text, measure how surprised it was by the true
   next token (the **loss**), nudge every number to be less surprised, repeat a
   few thousand times.

The whole repo is those six ideas, written to be read top-to-bottom.

## What makes it *modern*

This is the same architecture family as today's open models (Llama 3, Mistral,
Qwen), shrunk down. Each component is the current standard choice, and each is
implemented and **commented in plain English** in
[`magicmath/model.py`](magicmath/model.py):

| component | what it does | what it buys |
|---|---|---|
| **RMSNorm** | normalize each vector by its length | stable training, cheaply |
| **RoPE** (rotary) | encode position by *rotating* query/key vectors | positions become relative & length-flexible |
| **Grouped-Query Attention** | share key/value heads across query heads | a much smaller memory cache at inference |
| **SwiGLU** | a *gated* feed-forward layer | more quality per parameter |
| **no biases · pre-norm · tied embeddings** | lean, stable, parameter-efficient | small modern hygiene |

For the deeper, component-by-component walkthrough, see
[`docs/architecture.md`](docs/architecture.md).

## Where things live

```
magicmath/
  config.py     all the knobs + the nano/small/default presets   ← start here
  model.py      the transformer: RMSNorm, RoPE, GQA, SwiGLU       ← the good part
  tokenizer.py  train our own byte-level BPE vocabulary
  data.py       stream TinyStories → token ids on disk → batches
  train.py      the training loop (loss, AdamW, schedule, eval)
  sample.py     load a checkpoint and generate text
  web.py        the local dashboard (stdlib http.server + SSE, no framework)
  notebook.py   the live matplotlib chart for Colab
app/index.html  the dashboard front-end (vanilla JS + canvas, no dependencies)
notebooks/      the Colab "Run all" notebook
run.bat / run.sh  one-click local setup + launch
```

The entire dependency list is **three packages**: `torch`, `numpy`,
`tokenizers`. The local dashboard uses only Python's standard library.

## The three presets

| preset | params | data | ~T4 | ~A100 | result |
|---|---:|---:|---:|---:|---|
| `nano` | ~1.6M | 20 MB | ~5 min | ~2 min | semi-words; proves the pipeline |
| `small` | ~7M | 120 MB | ~30 min | ~10 min | short coherent fragments |
| `default` | ~12M | 250 MB | ~75 min | ~25 min | actual little stories |

Change the preset in the notebook's dropdown, or pass `--preset` to `run.bat`.

## Watching it learn (checkpoints)

The whole point of this repo is *visible* progress. During training, both paths
sample the model's writing from *“Once upon a time”* at a series of
**checkpoints** — starting at **step 0** (random noise) and every few hundred
steps after — and those snapshots **accumulate** so you can scroll the full arc:

```
── step 0 ──     prtmk oyye th ,a  e  oo nt...        (random weights)
── step 750 ──   the boy was a dog and the the play   (words, no grammar yet)
── step 3000 ──  Once upon a time there was a little   (sentences appear)
                 girl named Lily. She liked to play...
```

By default only the *final* model is written to `out/`. To also save the
**weights at every checkpoint** (so you can reload an early model and compare),
add `save_checkpoints=True` — `get_configs(PRESET, save_checkpoints=True)` in the
notebook, or `run.bat --save-checkpoints` locally. Each snapshot then writes
`out/model-<preset>-step<N>.pt`.

## Make it yours — hand the repo to Claude

This is a toy you're meant to take apart, and it's small enough that Claude can
hold the whole thing in its head. Open the folder in **Claude Code**, or just
paste files into a Claude chat, and try asking:

- *“In `magicmath/config.py`, add a preset that's deeper but narrower, then tell
  me what you'd expect to happen to the loss.”*
- *“Walk me through `apply_rope` in `model.py` line by line.”*
- *“Turn off weight tying (`tie_embeddings`) and re-run. Did it matter?”*
- *“Add top-p / nucleus sampling to `sample.py`.”*
- *“The samples are repetitive — what in the training recipe would I change?”*

Then re-run (Colab: *Run all*; local: `run.bat`) and see what happened. That
loop — change one thing, watch the loss and the samples — is how the intuitions
actually land.

## FAQ

**Do I need a GPU?** For Path A, no — Colab gives you one. For Path B, you can
run on CPU but it's slow; the script will tell you and install the CPU build
anyway if you insist. Path A is the easy button.

**Is this really from scratch?** Yes. The model starts from random weights and
the tokenizer is trained on the data here. Nothing pretrained is downloaded —
only raw TinyStories text.

**Why TinyStories?** It's a dataset of simple synthetic children's stories,
designed so that *tiny* models can learn fluent, grammatical English. It's what
makes “coherent sentences in minutes” possible at this size.

**It said something weird / repeated itself.** It's a 12M-parameter model that
trained for a few minutes — being occasionally weird is expected and kind of the
charm. Train `default` for longer, or raise/lower `temperature` when sampling.

---

MIT licensed. Built to be read, changed, and re-run.
