"""
The notebook front end — a flicker-free live reporter for Colab.

`prepare_data` / `train` only know about the `on_event` callback. This supplies
one tailored to a Jupyter/Colab cell. The trick that keeps it from strobing: we
**never** clear the cell. Instead the loss chart lives in a single output slot
that we update *in place* (and only every couple of seconds), while text samples
are printed once each and simply accumulate below it — so you can actually read
the checkpoints as they stream in while the chart keeps ticking above them.

Usage inside the notebook:

    from magicmath.notebook import NotebookReporter
    reporter = NotebookReporter()
    train(model_cfg, train_cfg, data, on_event=reporter)
"""

from __future__ import annotations


class NotebookReporter:
    def __init__(self, chart_every_s: float = 2.0):
        self.chart_every_s = chart_every_s     # min seconds between chart redraws
        self.steps, self.losses = [], []
        self.val_steps, self.val_losses = [], []
        self.cur = None                        # latest 'step' event, for the title
        self._chart = None                     # the in-place display handle
        self._last_chart = 0.0
        self._intro_printed = False

    def __call__(self, event: dict):
        import time
        t = event.get("type")
        if t == "config":
            n = event["config"].get("n_params", 0)
            print(f"Model: {n/1e6:.1f}M parameters on "
                  f"{event['config'].get('device')} — training…\n")
        elif t in ("phase", "progress"):
            if event.get("msg"):
                print(f"  · {event['msg']}")
        elif t == "step":
            self.steps.append(event["step"])
            self.losses.append(event["loss"])
            self.cur = event
            # throttle: only redraw the chart every few seconds, not every step
            if time.time() - self._last_chart >= self.chart_every_s:
                self._draw_chart()
        elif t == "eval":
            self.val_steps.append(event["step"])
            self.val_losses.append(event["val_loss"])
        elif t == "sample":
            self._draw_chart()                 # refresh the chart at the checkpoint
            self._print_sample(event)
        elif t == "done":
            self._draw_chart()
            print(f"\n✓ trained in {event['elapsed_s']/60:.1f} min · "
                  f"final loss {event['final_loss']:.3f} · saved {event.get('ckpt')}")

    def _print_sample(self, event: dict):
        # printed once, never cleared — so the progression stays readable
        if not self._intro_printed:
            self._intro_printed = True
            print("\nThe model’s writing at each checkpoint (prompt: "
                  "“Once upon a time”) — watch it go from noise to sentences:")
        vl = f" · val loss {event['val_loss']:.2f}" if event.get("val_loss") is not None else ""
        print(f"\n── step {event['step']}{vl} " + "─" * 22)
        print(event["text"].strip())

    def _draw_chart(self):
        import time
        import matplotlib.pyplot as plt
        from IPython.display import display
        self._last_chart = time.time()

        fig, ax = plt.subplots(figsize=(9, 4))
        if self.steps:
            ax.plot(self.steps, self.losses, lw=1, label="train loss", color="#4f8cff")
        if self.val_steps:
            ax.plot(self.val_steps, self.val_losses, "o-", lw=1.5,
                    label="val loss", color="#ff7a59")
        ax.set_xlabel("step")
        ax.set_ylabel("loss (cross-entropy)")
        if self.cur:
            c = self.cur
            ax.set_title(f"step {c['step']} / {c.get('max_steps','?')}  ·  "
                         f"loss {c['loss']:.3f}  ·  {c.get('tok_per_s',0):,.0f} tok/s  ·  "
                         f"eta {c.get('eta_s',0)/60:.1f} min")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        # one persistent output slot, updated in place — never clear_output()
        if self._chart is None:
            self._chart = display(fig, display_id=True)
        else:
            self._chart.update(fig)
        plt.close(fig)
