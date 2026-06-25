"""
The local web dashboard — the same live experience as the Colab notebook, but
for when you're training on your own GPU.

No web framework: this is Python's built-in http.server plus Server-Sent Events
(SSE), a dead-simple "server pushes a stream of text to the browser" protocol.
The only third-party packages the whole project needs are torch, numpy and
tokenizers.

What it does when you run `python -m magicmath.web`:
  1. starts a tiny HTTP server and opens your browser at http://localhost:8000
  2. kicks off data prep + training in a background thread
  3. streams every training event (loss, samples, throughput) to the page
  4. when training finishes, the page's prompt box lets you chat with your model

Architecture: a single EventHub fans every event out to (a) a rolling history
(so a freshly-opened/refreshed page can replay the whole run) and (b) every
connected browser's live stream.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(os.path.dirname(HERE), "app", "index.html")


class EventHub:
    """Fans events out to all connected browsers, and remembers history."""

    def __init__(self):
        self.history: list[dict] = []
        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.done = False

    def publish(self, event: dict):
        with self.lock:
            self.history.append(event)
            if event.get("type") == "done":
                self.done = True
            for q in self.subscribers:
                q.put(event)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            snapshot = list(self.history)
            self.subscribers.append(q)
        for ev in snapshot:        # replay history so a new page renders the full run
            q.put(ev)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


# A small holder the request handler can reach for the trained model.
class AppState:
    def __init__(self, hub: EventHub):
        self.hub = hub
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.lock = threading.Lock()
        # set once the trained model is exposed for generation. The 'done' event
        # fires a hair earlier (from inside train()), so a generate request that
        # arrives in that gap waits briefly on this instead of failing.
        self.model_ready = threading.Event()
        # which HTML page to serve at "/" (the training dashboard by default;
        # the chat playground overrides this).
        self.page = INDEX_HTML


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):   # silence the default request logging
            pass

        def _send(self, code, body: bytes, ctype="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                with open(state.page, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/events":
                self._stream_events()
            elif path == "/generate":
                self._generate()
            elif path == "/generate_stream":
                self._generate_stream()
            else:
                self._send(404, b"not found")

        def _ready_model(self):
            """Return (model, tok, device) once available, briefly waiting out
            the train->serve handoff; None if still training."""
            with state.lock:
                m = (state.model, state.tokenizer, state.device)
            if m[0] is None and state.model_ready.wait(timeout=5):
                with state.lock:
                    m = (state.model, state.tokenizer, state.device)
            return m if m[0] is not None else None

        def _stream_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = state.hub.subscribe()
            try:
                while True:
                    try:
                        ev = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")   # comment ping
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state.hub.unsubscribe(q)

        def _params(self):
            qs = parse_qs(urlparse(self.path).query)
            return (qs.get("prompt", [""])[0],
                    int(qs.get("tokens", ["200"])[0]),
                    float(qs.get("temperature", ["0.8"])[0]))

        def _generate(self):
            ready = self._ready_model()
            if ready is None:
                self._send(409, json.dumps({"error": "model still training"}).encode(),
                           "application/json")
                return
            model, tok, device = ready
            prompt, tokens, temperature = self._params()
            from .sample import generate as _gen
            with state.lock:
                text = _gen(model, tok, prompt, max_new_tokens=tokens,
                            temperature=temperature, device=device)
            self._send(200, json.dumps({"text": text}).encode(), "application/json")

        def _generate_stream(self):
            """Stream the completion token-by-token as plain text, so the page
            can show the model writing live."""
            ready = self._ready_model()
            if ready is None:
                self._send(409, json.dumps({"error": "model still training"}).encode(),
                           "application/json")
                return
            model, tok, device = ready
            prompt, tokens, temperature = self._params()
            # echo=0 suppresses the prompt echo (the playground shows the prompt
            # itself, so it only wants the model's continuation streamed back).
            echo = parse_qs(urlparse(self.path).query).get("echo", ["1"])[0] != "0"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            from .sample import generate_stream
            try:
                # one generation at a time (the model isn't thread-safe)
                with state.lock:
                    first = True
                    for delta in generate_stream(model, tok, prompt, max_new_tokens=tokens,
                                                 temperature=temperature, device=device):
                        if first:               # the first chunk is the prompt echo
                            first = False
                            if not echo:
                                continue
                        self.wfile.write(delta.encode("utf-8"))
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def _run_training(state: AppState, overrides: dict):
    """Runs in a background thread: prepare data, train, expose the model."""
    from .config import get_configs
    from .data import prepare_data
    from .train import train, pick_device
    from .events import fan_out, console_reporter

    hub = state.hub
    reporter = fan_out(hub.publish, console_reporter)
    try:
        model_cfg, train_cfg = get_configs(**overrides)
        device = pick_device()
        data = prepare_data(train_cfg, on_event=reporter, vocab_size=model_cfg.vocab_size)
        result = train(model_cfg, train_cfg, data, device=device, on_event=reporter)
        with state.lock:
            state.model = result["model"]
            state.tokenizer = result["tokenizer"]
            state.device = result["device"]
        state.model_ready.set()
    except Exception as e:   # surface failures to the page instead of dying silently
        import traceback
        traceback.print_exc()
        hub.publish({"type": "phase", "phase": "error", "msg": f"{type(e).__name__}: {e}"})


def serve(host: str = "127.0.0.1", port: int = 8000,
          open_browser: bool = True, overrides: dict | None = None):
    hub = EventHub()
    state = AppState(hub)
    server = ThreadingHTTPServer((host, port), make_handler(state))

    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://{host}:{port}"
    print(f"\n  magic-math dashboard:  {url}\n  (training the model — watch it in your browser)\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    train_thread = threading.Thread(
        target=_run_training, args=(state, overrides or {}), daemon=True)
    train_thread.start()

    try:
        train_thread.join()
        print("\n  training finished — the dashboard stays live so you can chat with "
              "your model.\n  press Ctrl+C to quit.\n")
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        print("\n  bye.")


def main():
    ap = argparse.ArgumentParser(description="magic-math local training dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="override the step count")
    ap.add_argument("--save-checkpoints", action="store_true",
                    help="also save the model's weights at each checkpoint step")
    args = ap.parse_args()

    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.save_checkpoints:
        overrides["save_checkpoints"] = True
    serve(host=args.host, port=args.port,
          open_browser=not args.no_browser, overrides=overrides)


if __name__ == "__main__":
    main()
