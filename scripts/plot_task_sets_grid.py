"""Side-by-side hole poses: M train holes vs held-out, M in 4/16/64/256. Usage: plot_task_sets_grid.py <out.png>"""
import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "examples/baselines/ppo")
from icl_tasks import checker_sample, SPLITS
sp = SPLITS["wide_corner"]
held = checker_sample(128, "all", "cpu", torch.Generator(device="cpu").manual_seed(777), "wide_corner").numpy()
BLUE, ORANGE = "#124690", "#B84A0A"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Ms = [4, 16, 64, 256]
fig, axs = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
for ax, M in zip(axs, Ms):
    tr = checker_sample(M, "all", "cpu", torch.Generator(device="cpu").manual_seed(10_000), "wide_corner").numpy()
    ax.scatter(held[:, 0], held[:, 1], s=14, color=ORANGE, alpha=0.5, zorder=2, label="held-out (128)")
    ax.quiver(held[:, 0], held[:, 1], 0.02 * np.cos(held[:, 2]), 0.02 * np.sin(held[:, 2]), color=ORANGE, alpha=0.4, angles="xy", scale_units="xy", scale=1, width=0.003, zorder=1)
    s = {4: 120, 16: 60, 64: 24, 256: 12}[M]
    ax.scatter(tr[:, 0], tr[:, 1], s=s, color=BLUE, edgecolor="white", linewidth=0.8 if M <= 16 else 0.3, zorder=4, label=f"train ({M})")
    ax.quiver(tr[:, 0], tr[:, 1], 0.03 * np.cos(tr[:, 2]), 0.03 * np.sin(tr[:, 2]), color=BLUE, angles="xy", scale_units="xy", scale=1, width=0.004, zorder=3)
    ax.scatter([0], [0], marker="s", color="k", s=40, zorder=5)
    ax.set(xlabel="hole box x (m)", xlim=(sp["x"][0] - .02, sp["x"][1] + .02), ylim=(-0.03, sp["y"][1] + .03), aspect="equal", title=f"{M} train holes")
    ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.32))
axs[0].set_ylabel("hole box y (m)")
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=160)
