"""Problem setup plots for the ICL toy: y*=|x*g|, train on x>0, test on x<0.
toy_setup.png: setup only. toy_setup_feedback.png: adds an example initial prediction y_0 + feedback (legend lists only feedback items)."""
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path("logs/toy"); out.mkdir(parents=True, exist_ok=True)
x = np.linspace(-1, 1, 400)


def make(feedback: bool, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.axvspan(0, 1, color="C2", alpha=.12); ax.axvspan(-1, 0, color="C3", alpha=.12)
    ax.text(0.5, 1.02, "TRAIN  (x>0)\ny = x·g", ha="center", va="bottom", fontsize=12, color="C2")
    ax.text(-0.5, 1.02, "TEST  (x<0)\ny = −x·g", ha="center", va="bottom", fontsize=12, color="C3")
    for g, c in zip((0.3, 0.6, 0.9), plt.cm.Blues([0.45, 0.7, 0.95])):
        ax.plot(x, np.abs(x * g), color=c, lw=2.5, label=None if feedback else f"true label |x·g|, g={g}")
        ax.plot(x[x < 0], (x * g)[x < 0], color=c, lw=1.5, ls="--")
    if not feedback:
        ax.plot([], [], "k--", lw=1.5, label="x·g extrapolated (what BC predicts)")
    else:
        # example failed attempts (x=±0.7, g=0.6, y*=0.42); initial prediction y_0 chosen off every drawn line
        for k, (xe, y0, tx, ha) in enumerate(((-0.7, -0.32, -0.66, "left"), (0.7, 0.30, 0.66, "right"))):
            ys = abs(xe * 0.6)
            ax.plot(xe, y0, "X", color="k", ms=10, zorder=5, label="initial prediction $y_0$" if k == 0 else None)
            ax.plot(xe, ys, "o", color="k", ms=8, zorder=5, label="target $y^*$" if k == 0 else None)
            ax.annotate("", xy=(xe, ys), xytext=(xe, y0), arrowprops=dict(arrowstyle="->", color="k", lw=2), zorder=4)
            ax.text(tx, (ys + y0) / 2 if k else 0.05, "feedback = $y^* - y_0$", fontsize=10, va="center", ha=ha,
                    bbox=dict(fc="white", ec="k", alpha=.85, boxstyle="round,pad=.25"))
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlim(-1, 1); ax.set_ylim(-0.95, 1.0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=.3); ax.legend(loc="lower center", fontsize=9)
    ax.set_title("Toy setup: y* = |x·g|, g∈[0,1]; train only on x>0", pad=36)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


make(False, out / "toy_setup.png")
make(True, out / "toy_setup_feedback.png")
