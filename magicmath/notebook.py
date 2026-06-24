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
        self.samples = []     # [{step, text, val_loss}] — the progression, kept
        self.config = None
        self.cur = None       # latest 'step' event, for the chart title
        self._since = 0

    def __call__(self, event: dict):
        t = event.get("type")
        if t == "config":
            self.config = event["config"]
            n = event["config"].get("n_params", 0)
            print(f"Model: {n/1e6:.1f}M parameters on {event['config'].get('device')} — training…")
        elif t in ("phase", "progress"):
            if event.get("msg"):
                print(f"  · {event['msg']}")
        elif t == "step":
            self.steps.append(event["step"])
            self.losses.append(event["loss"])
            self.cur = event
            self._since += 1
            if self._since >= self.refresh_every:
                self._since = 0
                self._render()
        elif t == "eval":
            self.val_steps.append(event["step"])
            self.val_losses.append(event["val_loss"])
            self._render()
        elif t == "sample":
            self.samples.append({"step": event["step"], "text": event["text"],
                                 "val_loss": event.get("val_loss")})
            self._render()
        elif t == "done":
            self._render(done=event)

    def _render(self, done=None):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        clear_output(wait=True)

        # --- the live loss chart ---
        fig = plt.figure(figsize=(9, 4))
        if self.steps:
            plt.plot(self.steps, self.losses, lw=1, label="train loss", color="#4f8cff")
        if self.val_steps:
            plt.plot(self.val_steps, self.val_losses, "o-", lw=1.5,
                     label="val loss", color="#ff7a59")
        plt.xlabel("step")
        plt.ylabel("loss (cross-entropy)")
        if done:
            plt.title(f"done · final loss {done['final_loss']:.3f} · "
                      f"{done['elapsed_s']/60:.1f} min")
        elif self.cur:
            c = self.cur
            plt.title(f"step {c['step']} / {c.get('max_steps','?')}  ·  loss {c['loss']:.3f}"
                      f"  ·  {c.get('tok_per_s',0):,.0f} tok/s  ·  "
                      f"eta {c.get('eta_s',0)/60:.1f} min")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close(fig)   # free the figure so repeated redraws don't pile up

        # --- the sample progression (this is the part that persists) ---
        if self.samples:
            print('The model’s writing at each checkpoint (prompt: '
                  '“Once upon a time”) — watch it go from noise to sentences:\n')
            for s in self.samples:
                vl = f" · val loss {s['val_loss']:.2f}" if s.get("val_loss") is not None else ""
                print(f"── step {s['step']}{vl} " + "─" * 22)
                print(s["text"].strip() + "\n")
        if done:
            print(f"✓ trained in {done['elapsed_s']/60:.1f} min · "
                  f"final loss {done['final_loss']:.3f} · saved {done.get('ckpt')}")
