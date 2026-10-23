"""Training curves of the PushTGoal-v1 expert PPO runs (one per goal-randomization range), from the TensorBoard logs."""
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = [  # (run dir, label, color = validated categorical slots 1-4 in order)
    ("diag-goal-pr", "no randomization (25M steps)", "#2a78d6"),
    ("pr-small", "goal ±1.5 cm, ±20°", "#eb6834"),
    ("pr-mid", "goal ±2 cm, ±30°", "#1baf7a"),
    ("pr-full", "goal ±3 cm, ±45° (expert used)", "#eda100"),
]
PANELS = [("eval/success_once", "success at any step"), ("eval/success_at_end", "success at final step")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

def load(run, tag):
    ea = EventAccumulator(glob.glob(f"runs/{run}/events*")[0], size_guidance={"scalars": 0}); ea.Reload()
    v = ea.Scalars(tag)
    return np.array([x.step for x in v]) / 1e6, np.array([x.value for x in v])

def smooth(y, k=3):
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")

fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb", sharey=True)
for ax, (tag, title) in zip(axs, PANELS):
    ax.set_facecolor("#fcfcfb")
    for run, label, c in RUNS:
        x, y = load(run, tag)
        ax.plot(x, y, color=c, lw=0.8, alpha=0.3)  # raw evals (64 episodes each, noisy)
        ax.plot(x, smooth(y), color=c, lw=2, label=label)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("environment steps (millions)", color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
axs[0].set_ylabel("eval success rate (64 episodes)", color=MUTED)
axs[0].legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
fig.suptitle("PushTGoal-v1 expert PPO: thin lines are raw evals, thick lines a 3-eval moving average", x=0.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.94))
mkdir = __import__("os").makedirs("renders", exist_ok=True)
fig.savefig("renders/goal_expert_training.png", dpi=140)
print("wrote renders/goal_expert_training.png")
