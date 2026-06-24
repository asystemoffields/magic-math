"""
Events — the one channel everything reports progress through.

The training loop doesn't know whether it's running inside a Colab notebook or
the local web dashboard. It just calls `on_event(some_dict)` at every
interesting moment. Each "front end" supplies its own `on_event`:

  * the notebook supplies one that draws a matplotlib loss curve (notebook.py)
  * the web app supplies one that pushes the dict to the browser (web.py)
  * here we provide a plain-text one for terminals / debugging

This decoupling is the whole reason the same training code powers both paths.

Event shapes (all are plain dicts with a "type" key):
  {"type":"config",  "config":{...}}                         once, at the start
  {"type":"phase",   "phase":"download", "msg":"..."}        coarse stage changes
  {"type":"progress","phase":"download", "msg":"...", ...}   fine progress within a stage
  {"type":"step",    "step":n, "loss":..., "lr":..., "tok_per_s":..., "eta_s":...}
  {"type":"eval",    "step":n, "val_loss":...}
  {"type":"sample",  "step":n, "text":"..."}                 a generated text sample
  {"type":"done",    "final_loss":..., "elapsed_s":..., "ckpt":"..."}
"""

from __future__ import annotations

import sys


def console_reporter(event: dict) -> None:
    """A no-frills reporter that prints to stdout. Used by CLIs and as a default."""
    t = event.get("type")
    if t == "config":
        m = event["config"]["model"]
        print(f"[config] d_model={m['d_model']} layers={m['n_layers']} "
              f"heads={m['n_heads']}/{m['n_kv_heads']}kv vocab={m['vocab_size']}")
    elif t in ("phase", "progress"):
        print(f"[{event.get('phase','')}] {event.get('msg','')}")
    elif t == "step":
        print(f"step {event['step']:>6}  loss {event['loss']:.3f}  "
              f"lr {event['lr']:.2e}  {event.get('tok_per_s',0):,.0f} tok/s  "
              f"eta {event.get('eta_s',0)/60:.1f} min")
    elif t == "eval":
        print(f"  -> val loss {event['val_loss']:.3f} at step {event['step']}")
    elif t == "sample":
        print(f"\n--- sample @ step {event['step']} ---\n{event['text']}\n---\n")
    elif t == "done":
        print(f"\ndone. final loss {event['final_loss']:.3f} in "
              f"{event['elapsed_s']/60:.1f} min. checkpoint: {event.get('ckpt')}")
    sys.stdout.flush()


def fan_out(*reporters):
    """Combine several reporters into one (e.g. console + web)."""
    def _combined(event: dict):
        for r in reporters:
            if r is not None:
                r(event)
    return _combined
