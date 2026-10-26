"""PPO expert training curves (eval success vs environment steps) for the three expert tasks, from the TensorBoard logs.
usage: python plot_ppo_experts.py"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_logging import LOG_ROOT
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TASKS = [  # (run dir, title)
    ("pr-full", "PushTGoal-v1 (goal +-3 cm, +-45 deg), 64 eval episodes"),
    ("pick-expert", "PickCube-v1, 64 eval episodes (stopped at 13M)"),
    ("peg-expert-fast", "PegInsertionSide-v1, 16 eval episodes"),
]
SERIES = [("eval/success_once", "success at any step", "#2a78d6")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load(run, tag):
    ea = EventAccumulator(glob.glob(f"{LOG_ROOT}/runs/{run}/events*")[0], size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    v = ea.Scalars(tag)
    return np.array([x.step for x in v]) / 1e6, np.array([x.value for x in v])


def smooth(y, k=5):
    if len(y) < 30:  # too few evals to average (pick)
        return y
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


fig, axs = plt.subplots(1, 3, figsize=(15, 4), facecolor=SURFACE, sharey=True)
for ax, (run, title) in zip(axs, TASKS):
    ax.set_facecolor(SURFACE)
    for tag, label, c in SERIES:
        d = load(run, tag)
        if d is None:
            continue
        x, y = d
        ax.plot(x, y, color=c, lw=0.8, alpha=0.3)
        ax.plot(x, smooth(y), color=c, lw=2, label=label)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("environment steps (millions)", color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
axs[0].set_ylabel("eval success rate", color=MUTED)
fig.suptitle("PPO expert training: thin lines are raw evals, thick lines a 5-eval moving average (none for the 14-eval pick run)", x=0.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = LOG_ROOT / "renders" / "ppo-experts" / "expert_training.png"
os.makedirs(out.parent, exist_ok=True)
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(out)
