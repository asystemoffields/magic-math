"""
The notebook front end — a reporter that draws a live loss curve in Colab.

`prepare_data` / `train` only know about the `on_event` callback. This module
supplies one tailored to a Jupyter/Colab cell: it redraws a matplotlib chart of
the training & validation loss in place, and prints text samples as they appear.

Usage inside the notebook:

    from magicmath.notebook import NotebookReporter
    reporter = NotebookReporter()
    train(model_cfg, train_cfg, data, on_event=reporter)
"""

from __future__ import annotations


class NotebookReporter:
    def __init__(self, refresh_every: int = 1):
        self.refresh_every = refresh_every
        self.steps, self.losses = [], []
        self.val_steps, self.val_losses = [], []
        self.config = None
        self._redraws = 0
        self._last_sample = ""

    def __call__(self, event: dict):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        t = event.get("type")
        if t == "config":
            self.config = event["config"]
            n = event["config"].get("n_params", 0)
            print(f"Model: {n/1e6:.1f}M parameters on {event['config'].get('device')}")
        elif t in ("phase", "progress"):
            if event.get("msg"):
                print(f"  · {event['msg']}")
        elif t == "step":
            self.steps.append(event["step"])
            self.losses.append(event["loss"])
            self._redraws += 1
            if self._redraws % self.refresh_every == 0:
                self._draw(plt, clear_output, event)
        elif t == "eval":
            self.val_steps.append(event["step"])
            self.val_losses.append(event["val_loss"])
        elif t == "sample":
            self._last_sample = event["text"]
            self._draw(plt, clear_output, event)
            print(f"\nsample @ step {event['step']}:\n{event['text']}\n")
        elif t == "done":
            self._draw(plt, clear_output, event)
            print(f"\n✓ trained in {event['elapsed_s']/60:.1f} min · "
                  f"final loss {event['final_loss']:.3f} · saved {event.get('ckpt')}")
            if self._last_sample:
                print(f"\nfinal sample:\n{self._last_sample}")

    def _draw(self, plt, clear_output, event):
        clear_output(wait=True)
        plt.figure(figsize=(9, 4))
        if self.steps:
            plt.plot(self.steps, self.losses, lw=1, label="train loss", color="#4f8cff")
        if self.val_steps:
            plt.plot(self.val_steps, self.val_losses, "o-", lw=1.5,
                     label="val loss", color="#ff7a59")
        plt.xlabel("step")
        plt.ylabel("loss (cross-entropy)")
        title = "training"
        if event.get("type") == "step":
            title = (f"step {event['step']} / {event.get('max_steps','?')}  ·  "
                     f"loss {event['loss']:.3f}  ·  {event.get('tok_per_s',0):,.0f} tok/s  ·  "
                     f"eta {event.get('eta_s',0)/60:.1f} min")
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
