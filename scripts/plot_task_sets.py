"""Visualise the hole-pose task sets used by ppo_tasks.py (4 train holes vs held-out vs full space). Usage: plot_task_sets.py <out.png> [n_train=4]"""
import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "examples/baselines/ppo")
from icl_tasks import checker_sample, SPLITS
dev = "cuda"  # training scripts use cuda generators; cpu generators give different poses
gp = torch.Generator(device=dev).manual_seed(10_000)            # n_tasks=4, pool_seed=0 (ppo_tasks.py)
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
train = checker_sample(M, "all", dev, gp, "wide_corner").cpu().numpy()
ge = torch.Generator(device=dev).manual_seed(777)               # eval sets share one generator: heldout first, then fullspace
held = checker_sample(128, "all", dev, ge, "wide_corner").cpu().numpy()
full = checker_sample(256, "all", dev, ge, "wide_corner").cpu().numpy()
sp = SPLITS["wide_corner"]
BLUE, ORANGE, PINK = "#124690", "#B84A0A", "#E97BA4"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]
for P, c, lab, s in [(full, PINK, "full space (256)", 18), (held, ORANGE, "held-out (128)", 18)]:
    ax.scatter(P[:, 0], P[:, 1], s=s, color=c, alpha=0.55, label=lab, zorder=2)
    ax.quiver(P[:, 0], P[:, 1], 0.022 * np.cos(P[:, 2]), 0.022 * np.sin(P[:, 2]), color=c, alpha=0.5, angles="xy", scale_units="xy", scale=1, width=0.003, zorder=1)
ax.scatter(train[:, 0], train[:, 1], s=140 if M <= 4 else 70, color=BLUE, edgecolor="white", linewidth=1.5, label=f"train holes ({M})", zorder=4)
ax.quiver(train[:, 0], train[:, 1], 0.05 * np.cos(train[:, 2]), 0.05 * np.sin(train[:, 2]), color=BLUE, angles="xy", scale_units="xy", scale=1, width=0.007, zorder=3)
for i, p in enumerate(train): ax.annotate(f"{i+1}", (p[0], p[1]), xytext=(8, 6), textcoords="offset points", color=BLUE, fontweight="bold")
ax.scatter([0], [0], marker="s", color="k", s=60, zorder=5); ax.annotate("robot base", (0, 0), xytext=(6, -12), textcoords="offset points", fontsize=8)
ax.set(xlabel="hole box x (m)", ylabel="hole box y (m)", xlim=(sp["x"][0] - .02, sp["x"][1] + .02), ylim=(-0.03, sp["y"][1] + .03), aspect="equal",
       title="Hole poses (arrows = box yaw)")
ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.28))
ax = axs[1]
yaw = np.degrees
bins = np.linspace(yaw(sp["yaw"][0]), yaw(sp["yaw"][1]), 13)
ax.hist(yaw(full[:, 2]), bins=bins, color=PINK, alpha=0.6, density=True, label="full space")
ax.hist(yaw(held[:, 2]), bins=bins, color=ORANGE, alpha=0.5, density=True, label="held-out")
for i, v in enumerate(yaw(train[:, 2])): ax.axvline(v, color=BLUE, lw=2.2, label="train holes" if i == 0 else None)
ax.set(xlabel="hole yaw (deg)", ylabel="density", title="Hole yaw"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=160)
print("train", np.round(train, 3).tolist()); print("ranges", sp["x"], sp["y"], np.degrees(sp["yaw"]))
