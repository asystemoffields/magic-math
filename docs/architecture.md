# The architecture, in depth

This is the longer companion to the comments in
[`magicmath/model.py`](../magicmath/model.py). It assumes you're comfortable with
vectors and have used language models a lot, but haven't built one. We'll go
component by component. Every choice here is the **current standard** — the same
design used by today's open models like Llama 3, Mistral and Qwen — and for each
we'll explain what it does and why it helps.

A decoder-only transformer is a function that takes a sequence of token ids and,
for every position, outputs a probability distribution over the next token. The
"residual stream" is the running vector for each token as it flows through the
network; every layer *reads* from it and *adds* a correction back into it.

```
ids ──embed──► [ block ]×N ──RMSNorm──► lm_head ──► next-token logits
                  │
                  ├─ RMSNorm → Attention (RoPE, GQA) → + (add to residual)
                  └─ RMSNorm → SwiGLU MLP            → + (add to residual)
```

---

## 1. Tokenizer — byte-level BPE

A model can't consume characters, only integers. We build the integer
vocabulary ourselves (`magicmath/tokenizer.py`):

- Start from the 256 raw bytes, so **any** input is representable — there is no
  "unknown token".
- Repeatedly find the most frequent adjacent pair of tokens and merge it into a
  new token. Frequent chunks (`the`, ` said`, `ing`) collapse to single ids.
- Stop at `vocab_size`.

This is exactly how production tokenizers (Llama, Mistral, …) are made; ours is
just smaller (2k–8k tokens) and trained on TinyStories. A smaller vocabulary
means a smaller embedding table, which matters when the whole model is 27M
parameters.

## 2. Embedding — ids become vectors

`nn.Embedding(vocab_size, d_model)` is a lookup table: row *i* is the vector for
token *i*. These vectors are **learned** during training. With **weight tying**
(`tie_embeddings=True`) the very same table is reused, transposed, as the final
output layer that scores next-token candidates — halving the parameters spent on
vocabulary and usually helping small models.

## 3. RMSNorm — keeping vectors a sane size

Deep networks are unstable if the numbers flowing through them drift in scale.
Normalization fixes the scale at each layer.

**RMSNorm** does the minimal version: divide each vector by its own
root-mean-square length and apply a single learned per-channel scale — no mean
subtraction, no bias. Fewer operations and fewer parameters than the classic
alternative (which also re-centers around the mean), with no quality loss in
practice. It's the default in Llama, Mistral, Gemma and Qwen.

We also normalize **before** each sub-layer ("pre-norm") rather than after
("post-norm"). Pre-norm keeps a clean residual highway down the network, which
is what lets it train stably without the learning-rate warmup tricks early
transformers needed.

## 4. RoPE — positions without position vectors

Attention on its own is order-blind: it sees a *set* of tokens, not a sequence.
Something has to inject "this token is at position 5."

The simplest approach is to *add* a learned vector for each absolute position
(0, 1, 2, …), but that caps the context length and only "knows" positions seen
in training. **Rotary Position Embeddings (RoPE)** instead **rotate** each query
and key vector by an angle proportional to its position. The dot product between
a query at position *m* and a key at position *n* then depends only on their
relative offset *(m − n)*. Attention becomes naturally **relative**, and the
model extrapolates to longer contexts far better.

In code (`apply_rope`), we pair dimension *i* with dimension *i + d/2* and rotate
each pair. The angles are precomputed once into cosine/sine tables.

## 5. Attention — with Grouped-Query heads

Attention lets each position gather information from earlier positions. Each
token emits a **query**; every token exposes a **key** and a **value**. A query
attends to the keys it matches, and reads a weighted blend of their values. We
mask so a token can only see the past (**causal** attention) — that's what makes
it a *predict-the-next-token* model.

Multiple **heads** do this in parallel in different subspaces. In the classic
form ("multi-head attention"), every head has its own key and value.

**Grouped-Query Attention (GQA)** keeps all `n_heads` query heads but only
`n_kv_heads` key/value heads, shared across groups of query heads. At inference,
the keys/values are what you must cache for every past token, so shrinking them
is the single biggest memory win in modern LLMs — at almost no quality cost.
`n_kv_heads == n_heads` recovers ordinary multi-head attention; `n_kv_heads == 1`
is multi-query attention.

The actual score-and-blend is one call to PyTorch's
`scaled_dot_product_attention`, which uses a fused FlashAttention kernel on GPU.

*(Optional, off by default:* `qk_norm` *applies RMSNorm to the queries and keys
before the dot product — a stability trick from OLMo2 and Gemma2.)*

## 6. SwiGLU — the per-token feed-forward layer

Between attention steps, each token's vector is transformed on its own by a small
MLP. This is where most of the parameters (and a lot of the "knowledge") live.

A plain MLP projects up to a larger hidden size, applies an activation, and
projects back down. **SwiGLU** adds a *gate*: it projects to *two* hidden
tensors, squashes one with SiLU and multiplies it element-wise into the other,
then projects down. That's three matrices, so the hidden size is scaled by 2/3
to keep the parameter count comparable. The gate lets the layer pass or suppress
information per channel, which empirically gives more quality per parameter. Used
by Llama, PaLM and Mistral.

## 7. Output head and the loss

After the final block and a last RMSNorm, the `lm_head` (the tied embedding
matrix) turns each position's vector into a score for every token in the
vocabulary. **Cross-entropy loss** measures how much probability the model put on
the *true* next token — minimizing it is exactly "be less surprised by real
text." A loss around `ln(vocab_size)` is random guessing; watch it fall well
below that.

## 8. The training recipe (`train.py`)

- **AdamW** optimizer, with weight decay applied only to the 2D weight matrices,
  not the 1D norm vectors.
- **Learning-rate schedule:** linear warmup, then cosine decay to a small floor.
- **Mixed precision:** `bfloat16` on modern GPUs (A100+), `float16` (with a
  gradient scaler) on older ones (T4), plain `float32` on CPU — chosen
  automatically.
- **Gradient clipping** at norm 1.0 for stability.
- **Gradient accumulation** lets you simulate a big batch on a small GPU.

Everything reports through a single `on_event` callback, which is why the same
loop drives both the Colab chart and the local web dashboard.

---

### Suggested reading order in the code

`config.py` (the knobs) → `model.py` (top to bottom) → `data.py` →
`train.py`. Each file is short and commented. Then change one thing and re-run.
