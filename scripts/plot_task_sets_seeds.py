"""Hole poses per seed: M train holes (pool_seed = seed) vs the shared held-out set (Generator 777, 512 holes). Usage: plot_task_sets_seeds.py <out.png> [Ms=4,8] [seeds=1,2,3]"""
import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "examples/baselines/ppo")
from icl_tasks import checker_sample, SPLITS
DEV = "cuda"  # must match the training scripts (cuda generators)
Ms = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "4,8").split(",")]
seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]
sp = SPLITS["wide_corner"]
held = checker_sample(512, "all", DEV, torch.Generator(device=DEV).manual_seed(777), "wide_corner").cpu().numpy()
BLUE, ORANGE = "#124690", "#B84A0A"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(len(Ms), len(seeds), figsize=(4.6 * len(seeds), 4.9 * len(Ms)), sharex=True, sharey=True, squeeze=False)
for i, M in enumerate(Ms):
    for j, s in enumerate(seeds):
        ax = axs[i, j]
        tr = checker_sample(M, "all", DEV, torch.Generator(device=DEV).manual_seed(10_000 + s), "wide_corner").cpu().numpy()
        ax.scatter(held[:, 0], held[:, 1], s=7, color=ORANGE, alpha=0.4, zorder=2, label="held-out (512)")
        ax.quiver(held[:, 0], held[:, 1], 0.015 * np.cos(held[:, 2]), 0.015 * np.sin(held[:, 2]), color=ORANGE, alpha=0.3, angles="xy", scale_units="xy", scale=1, width=0.002, zorder=1)
        ax.scatter(tr[:, 0], tr[:, 1], s=110, color=BLUE, edgecolor="white", linewidth=1.2, zorder=4, label=f"train ({M})")
        ax.quiver(tr[:, 0], tr[:, 1], 0.045 * np.cos(tr[:, 2]), 0.045 * np.sin(tr[:, 2]), color=BLUE, angles="xy", scale_units="xy", scale=1, width=0.006, zorder=3)
        ax.scatter([0], [0], marker="s", color="k", s=40, zorder=5)
        ax.set(xlim=(sp["x"][0] - .02, sp["x"][1] + .02), ylim=(-0.03, sp["y"][1] + .03), aspect="equal", title=f"{M} train holes, seed {s}")
        ax.grid(alpha=0.2)
        if j == 0: ax.set_ylabel("hole box y (m)")
        if i == len(Ms) - 1: ax.set_xlabel("hole box x (m)")
axs[0, 0].legend(frameon=False, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=150)
for M in Ms:
    for s in seeds: print(M, s, np.round(checker_sample(M, "all", DEV, torch.Generator(device=DEV).manual_seed(10_000 + s), "wide_corner").cpu().numpy(), 3).tolist()[:2], "...")
