"""
A little playground for a model you've already trained.

`magicmath.web` runs the training dashboard; this serves a focused "prompt the
model" page on top of a saved checkpoint — no training, just interaction. It
reuses the same streaming endpoint, so completions arrive token by token.

    python -m magicmath.chat                      # newest model in out/
    python -m magicmath.chat --ckpt out/model-default.pt
    python -m magicmath.chat --preset small       # prefer the 'small' model

Important framing (and the page says this loudly): this is a *story-continuation*
model trained on TinyStories, not a chat assistant. It continues whatever text
you give it. It can't hold a back-and-forth conversation, answer questions, or
follow instructions — it was never trained on any of that.
"""

from __future__ import annotations

import argparse
import glob
import os
import threading
import webbrowser
from http.server import ThreadingHTTPServer

from .web import EventHub, AppState, make_handler
from .sample import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT_HTML = os.path.join(os.path.dirname(HERE), "app", "chat.html")


def find_checkpoint(ckpt: str | None = None, preset: str | None = None,
                    out_dir: str = "out") -> str:
    """Resolve which checkpoint to load: an explicit path, else the newest final
    model in out/ (preferring the given preset, and final models over per-step
    ones)."""
    if ckpt:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt

    candidates = []
    if preset:
        candidates += glob.glob(os.path.join(out_dir, f"model-{preset}.pt"))
    candidates += glob.glob(os.path.join(out_dir, "model-*.pt"))
    finals = [c for c in candidates if "-step" not in os.path.basename(c)]
    pool = finals or candidates
    if not pool:
        raise FileNotFoundError(
            f"No trained model found in '{out_dir}/'. Train one first "
            f"(double-click run.bat, or `python -m magicmath.train_cli`).")
    return max(pool, key=os.path.getmtime)


def serve_chat(ckpt: str | None = None, preset: str | None = None,
               host: str = "127.0.0.1", port: int = 8000,
               open_browser: bool = True, out_dir: str = "out"):
    path = find_checkpoint(ckpt, preset, out_dir)
    print(f"loading {path} …")
    model, tok, device = load_model(path)

    state = AppState(EventHub())
    state.model, state.tokenizer, state.device = model, tok, device
    state.model_ready.set()
    state.page = CHAT_HTML

    server = ThreadingHTTPServer((host, port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://{host}:{port}"
    print(f"\n  magic-math playground:  {url}\n"
          f"  model: {os.path.basename(path)} "
          f"({model.num_params()/1e6:.1f}M params, {device})\n  Ctrl+C to quit.\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        print("\n  bye.")


def main():
    ap = argparse.ArgumentParser(description="Playground for a trained magic-math model.")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: newest in out/)")
    ap.add_argument("--preset", default=None, help="prefer the model trained for this preset")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    serve_chat(args.ckpt, args.preset, args.host, args.port,
               not args.no_browser, args.out_dir)


if __name__ == "__main__":
    main()
